"""Full-scale GPU training launcher for the SHA-ZA PPO pipeline.

Picks GPU-appropriate self-play settings, probes throughput for an honest
ETA, keeps the machine awake during the run, streams GPU telemetry to a
CSV, and forwards the real work to scripts/train.py (any train.py flag can
be passed after a bare ``--``).

Usage:
  uv run python scripts/launch_training.py                        # full notebook scale: 6000 steps x 8192 games
  uv run python scripts/launch_training.py --steps 1000 --games-per-step 4096
  uv run python scripts/launch_training.py --resume --save-dir runs/2026-11-24_213000
  uv run python scripts/launch_training.py -- --epsilon-max 0.1 --ref-path checkpoints/8.pth

Outputs in the save dir:
  config.json   the resolved launch configuration
  gpu_log.csv   GPU telemetry sampled every --monitor-interval seconds
  (plus everything train.py saves: checkpoints, log.npy, ...)

Defaults match the published notebook run (16 workers x 512 games/step,
6000 steps, batch 256, test every 25 steps). On a GPU the launcher switches
self-play to the vectorized lockstep environment (--batch-games, ~21x faster
per game on an RTX 3050 laptop) instead of per-process workers.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import torch
from tabulate import tabulate

REPO = Path(__file__).resolve().parent.parent


def detect_gpu():
    """Return (kind, info) for the compute device."""
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info = {
            'name': props.name,
            'vram_gb': round(props.total_memory / 2 ** 30, 1),
            'compute_capability': f'{props.major}.{props.minor}',
        }
        nvidia_smi = shutil.which('nvidia-smi')
        if nvidia_smi:
            out = subprocess.run([nvidia_smi, '--query-gpu=driver_version,name',
                                  '--format=csv,noheader'],
                                 capture_output=True, text=True, timeout=15).stdout.strip()
            if out:
                info['driver'], info['smi_name'] = (x.strip() for x in out.split(',', 1))
        return 'cuda', info
    return 'cpu', {'cores': os.cpu_count() or 1}


def gpu_busy():
    """Other processes currently using the GPU ('' if none, None if unknown)."""
    nvidia_smi = shutil.which('nvidia-smi')
    if not nvidia_smi:
        return None
    out = subprocess.run([nvidia_smi, '--query-compute-apps=pid,used_memory',
                          '--format=csv,noheader'],
                         capture_output=True, text=True, timeout=15).stdout.strip()
    return [line for line in out.splitlines() if line.strip()]


def probe_throughput(batch_games, games=64):
    """Measure vectorized self-play throughput with the real model (games/s,
    transitions/game). Includes numba/cudnn warmup at the real batch size so
    the GPU has reached steady clocks before timing; ~30-60 s on first run."""
    from othello import OthelloRL
    from othello.rl.vecenv import batchedSelfPlay

    device = torch.device('cuda')
    model = OthelloRL(internal_channels=64, res_depth=8, name='probe',
                      return_val=False).to(device).eval()
    # one full-size env first: numba compile, conv/cudnn warmup, and GPU boost
    # clocks all settle before we time the second env
    batchedSelfPlay(model, device, games, Epsilon=0.05, boltzmann=True,
                    batch=min(batch_games, games))
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    data = batchedSelfPlay(model, device, games, Epsilon=0.05, boltzmann=True,
                           batch=min(batch_games, games))
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    trans_per_game = len(data) / games if data else 0.0
    power_w = None
    nvidia_smi = shutil.which('nvidia-smi')
    if nvidia_smi:
        out = subprocess.run([nvidia_smi, '--query-gpu=power.draw',
                              '--format=csv,noheader,nounits'],
                             capture_output=True, text=True, timeout=15).stdout.strip()
        try:
            power_w = float(out.split()[0])
        except (ValueError, IndexError):
            power_w = None
    del model, data
    torch.cuda.empty_cache()
    return games / dt, trans_per_game, power_w


def estimate_test_seconds(test_num, engine_depth, ref):
    """Rough cost of one test round: vs random and vs best run on CPU at
    ~2 model-vs-model games/s; the minimax engine scales with depth."""
    s = test_num * 0.5                          # vs random
    s += test_num * 0.5                         # vs best model
    if ref:
        s += test_num * 0.5                     # vs reference model
    s += (test_num // 2) * min(4.0, 0.3 + 0.2 * engine_depth)  # vs engine
    return s


def monitor_gpu(log_path, interval, stop):
    """Sample GPU telemetry to a CSV until stop is set."""
    nvidia_smi = shutil.which('nvidia-smi')
    if not nvidia_smi:
        return
    query = '--query-gpu=timestamp,utilization.gpu,memory.used,memory.total,temperature.gpu,clocks.sm,power.draw'
    with open(log_path, 'w', newline='') as f:
        f.write('timestamp,util.gpu,mem.used.mb,mem.total.mb,temp.c,clocks.sm.mhz,power.w\n')
        while not stop.wait(interval):
            try:
                out = subprocess.run([nvidia_smi, query, '--format=csv,noheader,nounits'],
                                     capture_output=True, text=True, timeout=15).stdout.strip()
                if out:
                    f.write(out + '\n')
                    f.flush()
            except Exception:
                continue


class PreventSleep:
    """Pin Windows AC standby/hibernate timeouts to 0 for the run; restore
    the original values on exit (a lid-close nap silently stalls training)."""

    def __init__(self, enabled=True):
        self.enabled = enabled and sys.platform == 'win32' and bool(shutil.which('powercfg'))
        self.original = {}
        self.note = ''

    def _read(self, subgroup):
        out = subprocess.run(['powercfg', '/query', 'SCHEME_CURRENT', 'SUB_SLEEP', subgroup],
                             capture_output=True, text=True, timeout=15).stdout
        m = re.search(r'Current AC Power Setting Index:\s*(0x[0-9a-fA-F]+)', out)
        return int(m.group(1), 16) if m else None

    def __enter__(self):
        if not self.enabled:
            self.note = 'WARNING: not preventing sleep; the machine may suspend mid-training.'
            return self
        for subgroup, alias in (('STANDBYIDLE', 'standby-timeout-ac'),
                                ('HIBERNATEIDLE', 'hibernate-timeout-ac')):
            secs = self._read(subgroup)
            self.original[alias] = secs
            if secs is not None:
                subprocess.run(['powercfg', '/change', alias, '0'],
                               capture_output=True, timeout=15)
        missing = [a for a, s in self.original.items() if s is None]
        if missing:
            self.note = f'WARNING: could not read {missing} timeouts; sleep not fully disabled.'
        else:
            self.note = 'Standby/hibernate disabled for this run (restored at exit).'
        return self

    def __exit__(self, *exc):
        for alias, secs in self.original.items():
            if secs is not None:
                subprocess.run(['powercfg', '/change', alias, str(secs)],
                               capture_output=True, timeout=15)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--steps', type=int, default=6000, help='training steps (notebook: 6000)')
    ap.add_argument('--games-per-step', type=int, default=8192,
                    help='self-play games per step (notebook: 16 workers x 512)')
    ap.add_argument('--batch-games', type=int, default=None,
                    help='games per batched forward on GPU (default: auto by VRAM, 64-256)')
    ap.add_argument('--batch-size', type=int, default=256)
    ap.add_argument('--test-num', type=int, default=250)
    ap.add_argument('--test-freq', type=int, default=25)
    ap.add_argument('--epochs-per-step', type=int, default=1)
    ap.add_argument('--engine-depth', type=int, default=9)
    ap.add_argument('--eval-workers', type=int, default=None,
                    help='parallel eval game processes (default: min(8, cores//2))')
    ap.add_argument('--workers', type=int, default=None,
                    help='classic per-process workers (CPU path only; default min(16, cores))')
    ap.add_argument('--seed', type=int, default=None,
                    help='random seed for a reproducible run')
    ap.add_argument('--ref-path', type=str, default=None,
                    help='optional reference model for the eval ladder')
    ap.add_argument('--save-dir', type=str, default=None,
                    help='default: runs/run_<timestamp>')
    ap.add_argument('--resume', action='store_true',
                    help='pass --load-checkpoint to train.py (requires --save-dir)')
    ap.add_argument('--no-benchmark', action='store_true',
                    help='skip the throughput probe (no ETA estimate)')
    ap.add_argument('--monitor-interval', type=float, default=15.0,
                    help='GPU telemetry sampling period in seconds')
    ap.add_argument('--no-prevent-sleep', action='store_true',
                    help='do not touch Windows power timeouts')
    ap.add_argument('extra', nargs=argparse.REMAINDER,
                    help='extra scripts/train.py arguments (put them after a bare --)')
    args = ap.parse_args()

    if args.resume and args.save_dir is None:
        ap.error('--resume requires an explicit --save-dir')
    save_dir = (Path(args.save_dir) if args.save_dir
                else REPO / 'runs' / f'run_{datetime.now():%Y%m%d_%H%M%S}')
    # launch.sh may tee a launch.log here before we start; only train.py
    # artifacts (checkpoints, log.npy, ...) make a dir "used"
    launcher_artifacts = {'launch.log', 'config.json', 'gpu_log.csv'}
    used = save_dir.exists() and any(p.name not in launcher_artifacts
                                     for p in save_dir.iterdir())
    if used and not args.resume:
        sys.exit(f'{save_dir} already contains a training run; use --resume or a fresh --save-dir.')
    save_dir.mkdir(parents=True, exist_ok=True)

    kind, dev = detect_gpu()
    probe = {}
    eta_s = None

    if kind == 'cuda':
        batch_games = args.batch_games
        if batch_games is None:
            batch_games = 256 if dev['vram_gb'] >= 8 else 128 if dev['vram_gb'] >= 4 else 64
        batch_games = min(batch_games, max(1, args.games_per_step))
        workers, game_per_worker = 1, args.games_per_step

        busy = gpu_busy()
        if busy:
            print(f'WARNING: other process(es) already using the GPU:')
            for line in busy:
                print('  ', line)
        if not args.no_benchmark:
            print('Probing vectorized self-play throughput (2 x 64 games, ~30-60 s first time)...')
            games_s, trans_game, power_w = probe_throughput(batch_games)
            probe = {'games_per_s': round(games_s, 1),
                     'transitions_per_game': round(trans_game, 1),
                     'power_draw_w': power_w}
            print(f'  {games_s:.1f} games/s, {trans_game:.0f} transitions/game.')
            if power_w is not None and power_w < 40 and 'laptop' in dev.get('name', '').lower():
                print(f'  NOTE: GPU drew only {power_w:.0f} W during the probe — plug in and/or '
                      f'use a high-performance power plan for full throughput.')
    else:
        batch_games = 0
        workers = args.workers or min(16, os.cpu_count() or 1)
        if args.games_per_step % workers:
            sys.exit(f'--games-per-step {args.games_per_step} is not divisible by '
                     f'{workers} classic workers.')
        game_per_worker = args.games_per_step // workers

    if probe:
        eval_workers = args.eval_workers if args.eval_workers is not None \
            else min(8, max(1, (os.cpu_count() or 1) // 2))
        selfplay_s = args.games_per_step / probe['games_per_s']
        test_s = estimate_test_seconds(args.test_num, args.engine_depth, args.ref_path)
        eta_s = args.steps * (selfplay_s + test_s / args.test_freq / eval_workers)

    train_args = [sys.executable, str(REPO / 'scripts' / 'train.py'),
                  '--steps', str(args.steps),
                  '--batch-size', str(args.batch_size),
                  '--test-num', str(args.test_num),
                  '--test-freq', str(args.test_freq),
                  '--game-per-worker', str(game_per_worker),
                  '--num-workers', str(workers),
                  '--epochs-per-step', str(args.epochs_per_step),
                  '--engine-depth', str(args.engine_depth),
                  '--batch-games', str(batch_games),
                  '--save-dir', str(save_dir)]
    if args.eval_workers is not None:
        train_args += ['--eval-workers', str(args.eval_workers)]
    if args.seed is not None:
        train_args += ['--seed', str(args.seed)]
    if args.resume:
        train_args += ['--load-checkpoint']
    if args.ref_path:
        train_args += ['--ref-path', args.ref_path]
    train_args += [a for a in args.extra if a != '--']

    if args.resume and not (save_dir / 'log.npy').exists():
        print(f'WARNING: no log.npy in {save_dir}; train.py will start from step 1.')

    rows = [
        ('device', (f"{kind}: {dev.get('name')} ({dev.get('vram_gb')} GB, "
                    f"driver {dev.get('driver')})" if kind == 'cuda'
                    else f"cpu ({dev.get('cores')} logical cores)")),
        ('steps', args.steps),
        ('games/step', args.games_per_step),
        ('self-play', (f'vectorized, batch={batch_games}'
                       if kind == 'cuda' else f'classic, {workers} workers')),
        ('tests', f'every {args.test_freq} steps, {args.test_num} games'),
        ('seed', args.seed),
        ('save-dir', save_dir),
    ]
    if probe:
        rows.append(('throughput', f"{probe['games_per_s']} games/s"))
    if eta_s is not None:
        rows.append(('ETA', f'{eta_s / 3600:.1f} h ({eta_s / 86400:.1f} days)'))
    print(tabulate(rows, tablefmt='plain'))
    print('\ntrain.py command:\n  ' + ' '.join(train_args[1:]))

    (save_dir / 'config.json').write_text(json.dumps({
        'launched_at': datetime.now().isoformat(),
        'save_dir': str(save_dir),
        'device': dev,
        'launcher_args': vars(args),
        'train_argv': train_args[2:],
        'probe': probe,
        'eta_hours': round(eta_s / 3600, 2) if eta_s else None,
    }, indent=2))

    stop = threading.Event()
    monitor = threading.Thread(target=monitor_gpu,
                               args=(save_dir / 'gpu_log.csv', args.monitor_interval, stop),
                               daemon=True)
    monitor.start()

    print('')
    with PreventSleep(enabled=not args.no_prevent_sleep) as ps:
        print(ps.note)
        print('=' * 70)
        t0 = time.perf_counter()
        try:
            rc = subprocess.run(train_args, cwd=REPO).returncode
        except KeyboardInterrupt:
            rc = 130
            print('\nInterrupted.')
        dt = time.perf_counter() - t0
        stop.set()
        monitor.join(timeout=10)

    print(f'\nTraining exited with code {rc} after {dt / 3600:.2f} h.')
    if rc != 0:
        err = save_dir / 'errorlog.npy'
        if err.exists():
            try:
                import numpy as np
                print('train.py error log:', np.load(err, allow_pickle=True).item())
            except Exception:
                pass
        print(f'Resume with: python scripts/launch_training.py --resume --save-dir {save_dir}')
        sys.exit(rc)
    if args.resume:
        print(f'Resume again with: python scripts/launch_training.py --resume --save-dir {save_dir}')


if __name__ == '__main__':
    main()
