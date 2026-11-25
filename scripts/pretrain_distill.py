"""Warm-start pretraining from engine-distillation data (generate_expert_games.py).

Trains the SAME OthelloRL policy+value architecture used by PPO on two
supervised objectives per labeled position:

  policy: cross-entropy against the engine's move (through the masked softmax)
  value:  MSE against the game outcome from the mover's perspective

so train.py can start from sensible weights via --init-path instead of random
init: early evals start well above 50% vs random and the value head starts
calibrated, which lets the PPO phase spend its budget on real improvement.

The saved checkpoint is a plain state_dict — load it with train.py's
--init-path (architecture flags must match the ones used here).

Usage:
  uv run --no-sync python scripts/pretrain_distill.py --data data/distill --epochs 2 \
      --internal-channels 128 --policy-net-depth 16 --out checkpoints/distill_pretrain.pth
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch

from othello import OthelloRL


def load_shards(data_dir, max_samples=None):
    """Merge all games_*.npz shards into (x, mask, move, outcome) arrays."""
    xs, masks, moves, outs = [], [], [], []
    for shard in sorted(Path(data_dir).glob('games_*.npz')):
        d = np.load(shard)
        xs.append(d['x'])
        masks.append(d['mask'])
        moves.append(d['move'])
        outs.append(d['outcome'])
    x = np.concatenate(xs)
    mask = np.concatenate(masks)
    move = np.concatenate(moves).astype(np.int64)  # CE targets need int64 on CUDA
    out = np.concatenate(outs)
    if max_samples:
        idx = np.random.default_rng(0).permutation(len(x))[:max_samples]
        x, mask, move, out = x[idx], mask[idx], move[idx], out[idx]
    return x, mask, move, out


def to_batches(x, mask, move, out, idx, batch_size):
    for start in range(0, len(idx), batch_size):
        i = idx[start:start + batch_size]
        yield (torch.from_numpy(x[i]), torch.from_numpy(mask[i]),
               torch.from_numpy(move[i]), torch.from_numpy(out[i]))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', type=str, default='data/distill', help='shard directory')
    ap.add_argument('--epochs', type=int, default=2)
    ap.add_argument('--batch-size', type=int, default=512)
    ap.add_argument('--learning-rate', type=float, default=1e-3)
    ap.add_argument('--value-coef', type=float, default=0.5, help='PPO uses 0.5 too')
    ap.add_argument('--max-samples', type=int, default=None,
                    help='cap dataset size (for quick experiments)')
    ap.add_argument('--internal-channels', type=int, default=128,
                    help='must match train.py launch flags')
    ap.add_argument('--policy-net-depth', type=int, default=16,
                    help='must match train.py launch flags')
    ap.add_argument('--out', type=str, default='checkpoints/distill_pretrain.pth')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')

    x, mask, move, out = load_shards(args.data, args.max_samples)
    n = len(x)
    n_val = max(1, n // 20)
    perm = np.random.default_rng(args.seed).permutation(n)
    tr_idx, va_idx = perm[n_val:], perm[:n_val]
    print(f'{n} labeled positions ({len(tr_idx)} train / {len(va_idx)} val).')

    model = OthelloRL(internal_channels=args.internal_channels,
                      res_depth=args.policy_net_depth, name='distill',
                      return_val=True).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'architecture: {args.internal_channels} channels x {args.policy_net_depth} '
          f'blocks ({n_params / 1e6:.1f}M params).')

    optim = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=max(1, args.epochs // 2),
                                                gamma=0.5)
    ce = torch.nn.CrossEntropyLoss()

    def run_epoch(step, train):
        model.train(train)
        idx = tr_idx if train else va_idx
        if train:
            np.random.shuffle(idx)
        total = policy = value = acc = batches = 0
        for xb, mb, mv, ob in to_batches(x, mask, move, out, idx, args.batch_size):
            xb, mb, mv, ob = xb.to(device), mb.to(device), mv.to(device), ob.to(device)
            if train:
                optim.zero_grad(set_to_none=True)
            pol, val = model(xb, mb)
            val = val.squeeze(-1)
            policy_loss = ce(pol.view(len(xb), -1), mv[:, 0] * 8 + mv[:, 1])
            value_loss = torch.nn.functional.mse_loss(val, ob)
            loss = policy_loss + args.value_coef * value_loss
            if train:
                loss.backward()
                optim.step()
            with torch.no_grad():
                pred = pol.view(len(xb), -1).argmax(dim=1)
                acc += (pred == mv[:, 0] * 8 + mv[:, 1]).float().mean().item()
            total += loss.item()
            policy += policy_loss.item()
            value += value_loss.item()
            batches += 1
        n_b = max(1, batches)
        return total / n_b, policy / n_b, value / n_b, acc / n_b

    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        step = epoch
        tr = run_epoch(step, train=True)
        print(f'epoch {epoch}/{args.epochs}  '
              f'train: loss {tr[0]:.4f} (policy {tr[1]:.4f}, value {tr[2]:.4f}), '
              f'acc {100 * tr[3]:.2f}%  [{time.time() - t0:.0f}s]')
        scheduler.step()
    va = run_epoch(args.epochs + 1, train=False)
    print(f'val:   loss {va[0]:.4f} (policy {va[1]:.4f}, value {va[2]:.4f}), acc {100 * va[3]:.2f}%')

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_path)
    print(f'saved state_dict ({out_path.stat().st_size / 1e6:.1f} MB) to {out_path}')
    print('load into train.py with --init-path (same architecture flags).')


if __name__ == '__main__':
    main()
