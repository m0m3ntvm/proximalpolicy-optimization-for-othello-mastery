#!/usr/bin/env bash
# SHA-ZA PPO full-scale training on an RTX 5090 (Blackwell, 32 GB).
#
#   ./scripts/launch.sh                                  # full run with the parameters below
#   STEPS=1000 GAMES_PER_STEP=8192 ./scripts/launch.sh   # override anything via env
#   RESUME=1 SAVE_DIR=runs/run_20261124_213000 ./scripts/launch.sh
#   EXTRA="--epsilon-max 0.1" ./scripts/launch.sh        # any train.py flag
#
# Bootstraps uv, verifies torch can actually *execute* on the GPU (Blackwell
# sm_120 cannot run the lockfile's cu121 build — the script upgrades to a
# CUDA 12.8 torch if needed), then hands off to scripts/launch_training.py,
# which probes throughput for an ETA, streams GPU telemetry, and protects
# the save dir. Everything is logged to <save-dir>/launch.log.
set -euo pipefail
cd "$(dirname "$0")/.."

# ------------------------- training scale -------------------------
STEPS="${STEPS:-6000}"                    # published schedule; the LR decay is tied to it
GAMES_PER_STEP="${GAMES_PER_STEP:-16384}" # 2x the notebook's 8192: fresher on-policy data per update
BATCH_GAMES="${BATCH_GAMES:-256}"         # games per batched self-play forward (GPU-bound part)
BATCH_SIZE="${BATCH_SIZE:-512}"           # PPO mini-batch (notebook: 256)
TEST_NUM="${TEST_NUM:-250}"
TEST_FREQ="${TEST_FREQ:-25}"
ENGINE_DEPTH="${ENGINE_DEPTH:-9}"   # depth-9 engine: more accurate eval rung (parallelized)
EVAL_WORKERS="${EVAL_WORKERS:-8}"   # parallel eval game processes
SEED="${SEED:-42}"
# Self-play is Python-bound, so a bigger net is nearly free on a 5090 and
# raises the ceiling; for published-checkpoint parity use:
#   ARCH="--internal-channels 64 --policy-net-depth 8"
ARCH="${ARCH:---internal-channels 128 --policy-net-depth 16}"
SAVE_DIR="${SAVE_DIR:-runs/run_$(date +%Y%m%d_%H%M%S)}"
RESUME="${RESUME:-0}"
EXTRA="${EXTRA:-}"
INIT_PATH="${INIT_PATH:-}"    # distill warm-start checkpoint for train.py --init-path

# ------------------------- environment -------------------------
if ! command -v uv >/dev/null 2>&1; then
    echo "uv not found — installing it"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# Fresh environment: sync from the lockfile (skip on later launches so a
# cu128 torch upgrade is never reverted by re-syncing). --no-sync is
# essential everywhere below: `uv run` otherwise re-syncs the venv back to
# the lockfile's cu121 torch and silently undoes the Blackwell upgrade.
if ! uv run --no-sync python -c "import torch" 2>/dev/null; then
    uv sync --frozen || uv sync
fi

# Blackwell (sm_120) has no cu121 kernels; probe with a real kernel launch
# and install a CUDA 12.8 build if it fails.
if ! uv run --no-sync python -c "
import torch
assert torch.cuda.is_available()
torch.zeros(4, device='cuda')
print(f'torch {torch.__version__} (cuda {torch.version.cuda}), capability {torch.cuda.get_device_capability()}')"; then
    echo "torch cannot execute on this GPU — installing torch 2.7.1+cu128 (Blackwell)"
    uv pip install --upgrade "torch==2.7.1+cu128" --index-url https://download.pytorch.org/whl/cu128
    uv run --no-sync python -c "import torch; assert torch.cuda.is_available(); torch.zeros(4, device='cuda'); print('torch', torch.__version__, 'ready on this GPU')"
fi

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$(nproc)}"

mkdir -p "$SAVE_DIR"

ARGS=(--steps "$STEPS" --games-per-step "$GAMES_PER_STEP" --batch-games "$BATCH_GAMES"
      --batch-size "$BATCH_SIZE" --test-num "$TEST_NUM" --test-freq "$TEST_FREQ"
      --engine-depth "$ENGINE_DEPTH" --eval-workers "$EVAL_WORKERS"
      --seed "$SEED" --save-dir "$SAVE_DIR")
if [ "$RESUME" = "1" ]; then
    ARGS+=(--resume)
fi
# ARCH, INIT_PATH and EXTRA are train.py flags: pass them through the launcher's bare --
ARGS+=(--)
read -r -a ARCH_ARGS <<< "$ARCH"
ARGS+=("${ARCH_ARGS[@]}")
if [ -n "$INIT_PATH" ]; then
    ARGS+=(--init-path "$INIT_PATH")
fi
if [ -n "$EXTRA" ]; then
    read -r -a EXTRA_ARGS <<< "$EXTRA"
    ARGS+=("${EXTRA_ARGS[@]}")
fi

# Keep the machine awake where systemd is available. The binary exists even
# in containers with no systemd bus, so probe that an inhibit actually works
# before wrapping (a failed wrapper aborts the run under `set -e`).
CMD=(uv run --no-sync python scripts/launch_training.py "${ARGS[@]}")
if command -v systemd-inhibit >/dev/null 2>&1 \
    && systemd-inhibit --what=sleep:shutdown --why="SHA-ZA PPO training" true 2>/dev/null; then
    CMD=(systemd-inhibit --what=sleep:shutdown --why="SHA-ZA PPO training" "${CMD[@]}")
fi

echo "launch.sh: ${CMD[*]}"
"${CMD[@]}" 2>&1 | tee -a "$SAVE_DIR/launch.log"
