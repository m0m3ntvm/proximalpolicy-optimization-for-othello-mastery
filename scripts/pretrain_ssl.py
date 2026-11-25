"""Masked-encoder pretraining with zero engine input (generate_ssl_positions.py).

Two self-supervised objectives, both pure game structure:

  legality  the network fills in the valid moves: given a position, predict
            which squares a stone could land on. Trained through the SAME
            masked-softmax policy head PPO uses (target = uniform over legal
            squares), so the trunk + policy head start structured.
  recon     BERT-style masked cells: ~15% of board cells are zeroed (and the
            under-attack channel is randomly dropped), and a small
            auxiliary head reconstructs the original cells from the trunk
            features. Discarded after pretraining — it only exists to force
            the shared trunk to learn Othello geometry.
  value     (bonus, from the generator's random-play games) MSE on the game
            outcome — warms up the value head without any expert signal.

The saved checkpoint is the plain state_dict of the OthelloRL architecture
(no auxiliary heads), so train.py --init-path loads it directly — the
architecture flags must match.

Usage:
  uv run --no-sync python scripts/pretrain_ssl.py --data data/ssl --epochs 2 \
      --internal-channels 128 --policy-net-depth 16 --out checkpoints/ssl_pretrain.pth
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from othello import OthelloRL


class SSLWrapper(nn.Module):
    """OthelloRL trunk + policy head, plus a discardable reconstruction head."""

    def __init__(self, internal_channels, res_depth):
        super().__init__()
        self.base = OthelloRL(internal_channels=internal_channels,
                              res_depth=res_depth, name='ssl', return_val=True)
        self.recon_head = nn.Conv2d(in_channels=internal_channels, out_channels=3,
                                    kernel_size=(1, 1), bias=True)

    def forward(self, input, mask, masked_input=None):
        """Return (policy_logits, policy, recon_logits, value)."""
        trunk = self.base.ResidualBlock(self.base.InConv(input))
        logits = self.base.OutConv(trunk)
        masked = logits * mask + (1 - mask) * -1e32
        policy = self.base.softmax(masked.view(masked.size(0), -1)).view(masked.size())
        recon = self.recon_head(trunk)
        value = self.base.value_net(torch.cat((masked_input, policy), axis=1)) \
            if masked_input is not None else None
        return logits, policy, recon, value


def load_shards(data_dir, max_samples=None):
    xs, masks, outs = [], [], []
    for shard in sorted(Path(data_dir).glob('positions_*.npz')):
        d = np.load(shard)
        xs.append(d['x'])
        masks.append(d['mask'])
        outs.append(d['outcome'])
    x = np.concatenate(xs)
    mask = np.concatenate(masks)
    out = np.concatenate(outs)
    if max_samples:
        idx = np.random.default_rng(0).permutation(len(x))[:max_samples]
        x, mask, out = x[idx], mask[idx], out[idx]
    return x, mask, out


def batches(x, mask, out, idx, batch_size):
    for start in range(0, len(idx), batch_size):
        i = idx[start:start + batch_size]
        yield (torch.from_numpy(x[i]), torch.from_numpy(mask[i]),
               torch.from_numpy(out[i]), i)


def legality_target(mask):
    """Uniform distribution over legal squares (CE target for the policy head)."""
    t = mask.view(len(mask), -1)
    return t / t.sum(dim=1, keepdim=True).clamp(min=1.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', type=str, default='data/ssl', help='shard directory')
    ap.add_argument('--epochs', type=int, default=2)
    ap.add_argument('--batch-size', type=int, default=1024)
    ap.add_argument('--learning-rate', type=float, default=1e-3)
    ap.add_argument('--recon-coef', type=float, default=0.25)
    ap.add_argument('--value-coef', type=float, default=0.5)
    ap.add_argument('--mask-frac', type=float, default=0.15, help='cells zeroed per sample')
    ap.add_argument('--attack-drop', type=float, default=0.5,
                    help='prob. of dropping the under-attack channel (forces the '
                         'trunk to read the board itself)')
    ap.add_argument('--max-samples', type=int, default=None)
    ap.add_argument('--internal-channels', type=int, default=128,
                    help='must match train.py launch flags')
    ap.add_argument('--policy-net-depth', type=int, default=16,
                    help='must match train.py launch flags')
    ap.add_argument('--out', type=str, default='checkpoints/ssl_pretrain.pth')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')

    x, mask, out = load_shards(args.data, args.max_samples)
    n = len(x)
    n_val = max(1, n // 20)
    perm = np.random.default_rng(args.seed).permutation(n)
    tr_idx, va_idx = perm[n_val:], perm[:n_val]
    has_outcome = out != 0.0
    print(f'{n} positions ({len(tr_idx)} train / {len(va_idx)} val, '
          f'{has_outcome.mean() * 100:.0f}% with outcome labels).')

    model = SSLWrapper(args.internal_channels, args.policy_net_depth).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'architecture: {args.internal_channels} channels x {args.policy_net_depth} '
          f'blocks ({n_params / 1e6:.1f}M params incl. aux heads).')

    optim = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=max(1, args.epochs // 2),
                                                gamma=0.5)
    ce = nn.CrossEntropyLoss(reduction='none')

    def run_epoch(epoch, train):
        model.train(train)
        idx = tr_idx if train else va_idx
        if train:
            np.random.shuffle(idx)
        tot = pol = rec = val = acc_leg = acc_rec = 0.0
        b = 0
        for xb, mb, ob, bi in batches(x, mask, out, idx, args.batch_size):
            xb, mb, ob = xb.to(device), mb.to(device), ob.to(device)
            # per-sample random masking of board cells (channel 0 only)
            cell_mask = (torch.rand_like(xb[:, 0:1]) > args.mask_frac).float()
            xb_masked = xb * cell_mask
            if args.attack_drop and train:
                drop = (torch.rand(len(xb), device=device) < args.attack_drop)
                xb_masked[drop, 1] = 0.0
            # under-attack channel always fed to the trunk (hint channel);
            # only the board cells are masked
            trunk_in = torch.cat([xb_masked[:, 0:1], xb[:, 1:2]], dim=1)
            if train:
                optim.zero_grad(set_to_none=True)
            logits, policy, recon, value = model(trunk_in, mb, masked_input=xb_masked)
            # legality: CE on masked-softmax logits vs uniform-over-legal target
            policy_loss = ce(logits.view(len(xb), -1),
                             legality_target(mb).view(len(xb), -1)).mean()
            # recon: CE on the masked cells only (class = -1/0/+1 -> 0/1/2)
            cell_flat = cell_mask.view(len(xb), -1) == 0
            target = (xb[:, 0].view(len(xb), -1).long() + 1)[cell_flat]
            recon_loss = ce(recon.view(len(xb), 3, -1).transpose(1, 2).reshape(-1, 3)[cell_flat.view(-1)],
                            target).mean()
            loss = policy_loss + args.recon_coef * recon_loss
            # value: MSE on outcomes, only where the generator provided one
            val_loss = 0.0
            if value is not None and has_outcome[bi].any():
                good = has_outcome[bi]
                val_loss = nn.functional.mse_loss(value.squeeze(-1)[good],
                                                  ob[good], reduction='mean')
                loss = loss + args.value_coef * val_loss
            if train:
                loss.backward()
                optim.step()
            with torch.no_grad():
                # is the top raw-logit square legal? (unmasked preference)
                acc_leg += mb.view(len(xb), -1).gather(
                    1, logits.view(len(xb), -1).argmax(dim=1, keepdim=True)).mean().item()
                rec_pred = recon.argmax(dim=1).view(len(xb), -1).long() - 1
                rec_gold = xb[:, 0].view(len(xb), -1).long()
                acc_rec += (rec_pred[cell_flat] == rec_gold[cell_flat]).float().mean().item()
            tot += loss.item()
            pol += policy_loss.item()
            rec += recon_loss.item()
            val += float(val_loss)
            b += 1
        n_b = max(1, b)
        print(f'epoch {epoch}  {"train" if train else "val  "}: '
              f'loss {tot / n_b:.4f} (legal {pol / n_b:.4f}, recon {rec / n_b:.4f}, '
              f'value {val / n_b:.4f}), top-move-legal {100 * acc_leg / n_b:.1f}%, '
              f'masked-cell acc {100 * acc_rec / n_b:.1f}%')
        return tot / n_b

    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        run_epoch(epoch, train=True)
        run_epoch(epoch, train=False)
        scheduler.step()
    print(f'total time: {time.time() - t0:.0f}s')

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.base.state_dict(), out_path)
    print(f'saved base state_dict ({out_path.stat().st_size / 1e6:.1f} MB) to {out_path}')
    print('load into train.py with --init-path (same architecture flags).')


if __name__ == '__main__':
    main()
