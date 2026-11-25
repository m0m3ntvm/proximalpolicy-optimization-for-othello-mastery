"""Masked-autoencoding pretraining on trajectory chunks (generate_ssl_chunks.py).

Zero engine input. Each chunk is a sequence of consecutive random-play
positions; the input is heavily masked and the model reconstructs the full
sequence:

  tube masking   each of the 4 board quarters is dropped with probability
                 --block-drop (0.5) independently PER CHUNK, and the same
                 quarters are masked across ALL frames — so adjacent
                 near-duplicate frames cannot leak the answer, while some
                 chunks end fully masked and others untouched
  future frames  the last --future-frames frames are masked completely:
                 reconstructing them forces the encoder to learn the game's
                 MOVE DYNAMICS, not just per-frame inpainting

The encoder is the exact OthelloRL trunk (shared by policy and value heads
after the change to a shared value head). Two auxiliaries warm up the RL
heads at the same time: legality CE on the policy head (uniform-over-legal
target) and value MSE on the recorded game outcomes.

The saved checkpoint is the encoder's full state_dict (policy head + shared
value head included) — load it with train.py --init-path. Architecture
flags must match.

Usage:
  uv run --no-sync python scripts/pretrain_mae.py --data data/ssl_chunks --epochs 2 \
      --internal-channels 128 --policy-net-depth 16 --out checkpoints/mae_pretrain.pth
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from othello.models.factory import make_agent


class MAEModel(nn.Module):
    """Encoder (any architecture from the factory) + a small 3D-conv decoder
    that sees all frames. The encoder is only touched through its public
    contract: features(x) -> (trunk map, pre-softmax logits) and
    forward(x, mask) -> (masked-softmax policy, value)."""

    def __init__(self, arch, internal_channels, res_depth, model_dim, n_layers):
        super().__init__()
        self.encoder = make_agent(arch=arch, internal_channels=internal_channels,
                                  res_depth=res_depth, model_dim=model_dim,
                                  n_layers=n_layers, name='mae', return_val=True)
        C = getattr(self.encoder, 'trunk_channels', None) \
            or getattr(self.encoder, 'internal_channels', None)
        self.decoder = nn.Sequential(
            nn.Conv3d(C + 1, C, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=4, num_channels=C, affine=True),
            nn.LeakyReLU(),
            nn.Conv3d(C, C, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=4, num_channels=C, affine=True),
            nn.LeakyReLU(),
            nn.Conv3d(C, 3, kernel_size=1),
        )

    def forward(self, x, mask, keep):
        """x: (B, L, 2, 8, 8), mask: (B, L, 1, 8, 8), keep: (B, L, 1, 8, 8) 1=visible.
        Returns (logits, policy, value, recon_logits)."""
        B, L = x.shape[:2]
        x_masked = x * keep
        xf = x_masked.view(B * L, 2, 8, 8)
        mf = mask.view(B * L, 1, 8, 8)
        trunk, logits = self.encoder.features(xf)
        policy, value = self.encoder.forward(xf, mf)
        # Conv3d layout (B, C, L, H, W): move the frame dim into channel position
        feats3 = trunk.view(B, L, -1, 8, 8).permute(0, 2, 1, 3, 4)
        keep3 = keep.permute(0, 2, 1, 3, 4)
        recon = self.decoder(torch.cat((feats3, keep3), axis=1)).permute(0, 2, 1, 3, 4)
        return logits.view(B * L, 1, 8, 8), policy.view(B * L, 1, 8, 8), \
            value.view(B * L, 1), recon


def load_chunks(data_dir, max_samples=None):
    xs, masks, outs = [], [], []
    for shard in sorted(Path(data_dir).glob('chunks_*.npz')):
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
               torch.from_numpy(out[i]))


def build_keep(B, L, block_drop, future_frames):
    """Tube-masked keep indicator: per chunk, each of the 4 quarter-blocks is
    dropped with probability `block_drop` and masked in EVERY frame (tube
    masking, so adjacent near-duplicate frames cannot leak the answer — some
    chunks come out fully masked, others untouched), plus the last
    `future_frames` frames masked completely. 1 = visible."""
    keep = torch.ones(B, L, 1, 8, 8)
    drop = torch.rand(B, 4) < block_drop          # (B, 4) per-chunk quarter coins
    for blk in range(4):
        r0, c0 = (blk // 2) * 4, (blk % 2) * 4
        keep[drop[:, blk], :, :, r0:r0 + 4, c0:c0 + 4] = 0.0
    if future_frames > 0:
        keep[:, -future_frames:] = 0.0
    return keep


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', type=str, default='data/ssl_chunks', help='shard directory')
    ap.add_argument('--epochs', type=int, default=2)
    ap.add_argument('--batch-size', type=int, default=16, help='chunks per batch')
    ap.add_argument('--learning-rate', type=float, default=1e-3)
    ap.add_argument('--legality-coef', type=float, default=0.5)
    ap.add_argument('--value-coef', type=float, default=0.5)
    ap.add_argument('--block-drop', type=float, default=0.5,
                    help='per-chunk dropout probability for each quarter-block (tube-masked)')
    ap.add_argument('--future-frames', type=int, default=3,
                    help='trailing frames of each chunk masked completely')
    ap.add_argument('--max-samples', type=int, default=None)
    ap.add_argument('--arch', type=str, default='convnext',
                    choices=('convnext', 'vim', 'xlstm'),
                    help='encoder architecture (must match train.py --arch)')
    ap.add_argument('--internal-channels', type=int, default=128,
                    help='convnext arch: must match train.py launch flags')
    ap.add_argument('--policy-net-depth', type=int, default=16,
                    help='convnext arch: must match train.py launch flags')
    ap.add_argument('--model-dim', type=int, default=192,
                    help='vim/xlstm arch: token embedding width')
    ap.add_argument('--layers', type=int, default=8,
                    help='vim/xlstm arch: number of SSM/LSTM blocks')
    ap.add_argument('--out', type=str, default='checkpoints/mae_pretrain.pth')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')

    x, mask, out = load_chunks(args.data, args.max_samples)
    n, L = x.shape[:2]
    n_val = max(1, n // 20)
    perm = np.random.default_rng(args.seed).permutation(n)
    tr_idx, va_idx = perm[n_val:], perm[:n_val]
    print(f'{n} chunks of {L} frames ({len(tr_idx)} train / {len(va_idx)} val).')

    model = MAEModel(args.arch, args.internal_channels, args.policy_net_depth,
                     args.model_dim, args.layers).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    arch_desc = (f'{args.internal_channels} channels x {args.policy_net_depth} blocks'
                 if args.arch == 'convnext' else
                 f'{args.model_dim} dim x {args.layers} layers')
    print(f'architecture: {args.arch} ({arch_desc}, {n_params / 1e6:.1f}M params '
          f'incl. decoder).')

    optim = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=max(1, args.epochs // 2),
                                                gamma=0.5)
    ce = nn.CrossEntropyLoss(reduction='none')

    def run_epoch(epoch, train):
        model.train(train)
        idx = tr_idx if train else va_idx
        if train:
            np.random.shuffle(idx)
        tot = rec = leg = val = acc_rec = acc_fut = acc_leg = 0.0
        b = 0
        for xb, mb, ob in batches(x, mask, out, idx, args.batch_size):
            B = len(xb)
            xb, mb, ob = xb.to(device), mb.to(device), ob.to(device)
            keep = build_keep(B, L, args.block_drop, args.future_frames).to(device)
            visible = keep.view(B, L, -1).sum(dim=2) > 0   # (B, L) frames with info left
            if train:
                optim.zero_grad(set_to_none=True)
            logits, policy, value, recon = model(xb, mb, keep)
            # recon: CE on masked cells only (tube blocks on visible frames +
            # everything on the fully-masked future frames)
            target = (xb[:, :, 0].long() + 1)                                  # (B, L, 8, 8)
            masked_pos = (keep.view(B, L, 8, 8) == 0)
            recon_loss = ce(recon.view(B, L, 3, 8, 8).permute(0, 1, 3, 4, 2).reshape(-1, 3)[masked_pos.view(-1)],
                            target.view(-1)[masked_pos.view(-1)]).mean()
            loss = recon_loss
            # legality aux: CE on raw logits vs uniform-over-legal (visible frames)
            vis = visible.view(-1)
            if vis.any():
                tgt = mb.view(B * L, 1, 8, 8).view(B * L, 64)
                tgt = tgt / tgt.sum(dim=1, keepdim=True).clamp(min=1.0)
                legality_loss = ce(logits.view(B * L, 64)[vis], tgt[vis]).mean()
                loss = loss + args.legality_coef * legality_loss
                val_loss = nn.functional.mse_loss(value.squeeze(-1)[vis],
                                                  ob.view(B * L)[vis])
                loss = loss + args.value_coef * val_loss
            else:
                legality_loss = val_loss = torch.zeros(())
            if train:
                loss.backward()
                optim.step()
            with torch.no_grad():
                pred = recon.argmax(dim=2).view(B, L, 8, 8).long() - 1
                gold = xb[:, :, 0].long()
                acc_rec += (pred[masked_pos] == gold[masked_pos]).float().mean().item()
                fut = torch.zeros(B, L, 8, 8, dtype=torch.bool, device=device)
                fut[:, -args.future_frames:] = True
                acc_fut += (pred[fut] == gold[fut]).float().mean().item() if fut.any() else 0.0
                acc_leg += mb.view(B * L, 64).gather(
                    1, logits.view(B * L, 64).argmax(dim=1, keepdim=True))[vis].mean().item() \
                    if vis.any() else 0.0
            tot += loss.item()
            rec += recon_loss.item()
            leg += float(legality_loss)
            val += float(val_loss)
            b += 1
        n_b = max(1, b)
        print(f'epoch {epoch}  {"train" if train else "val  "}: '
              f'loss {tot / n_b:.4f} (recon {rec / n_b:.4f}, legal {leg / n_b:.4f}, '
              f'value {val / n_b:.4f}), masked acc {100 * acc_rec / n_b:.1f}% '
              f'(future {100 * acc_fut / n_b:.1f}%), top-move-legal {100 * acc_leg / n_b:.1f}%')
        return tot / n_b

    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        run_epoch(epoch, train=True)
        run_epoch(epoch, train=False)
        scheduler.step()
    print(f'total time: {time.time() - t0:.0f}s')

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.encoder.state_dict(), out_path)
    print(f'saved encoder state_dict ({out_path.stat().st_size / 1e6:.1f} MB) to {out_path}')
    print('load into train.py with --init-path (same architecture flags).')


if __name__ == '__main__':
    main()
