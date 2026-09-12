#!/usr/bin/env bash
# IMS2026 HW0 Task 2 - the live TA demo.
#
# The assignment asks for exactly this during the session: LIBERO-Long only,
# 10 tasks, 1 episode each, with the live-view window and the terminal
# evaluation output both visible.
#
#   ./demo.sh              # the graded demo run
#   ./demo.sh --headless   # same numbers, no window (for a dry run over SSH)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source /home/yehzr/anaconda3/etc/profile.d/conda.sh
conda activate hw0-lerobot
export HF_HUB_DISABLE_PROGRESS_BARS=1 MUJOCO_GL=egl TOKENIZERS_PARALLELISM=false

CKPT="${HERE}/outputs/libero_smolvla/checkpoints/100000/pretrained_model"
[[ -d "$CKPT" ]] || { echo "checkpoint not found: $CKPT" >&2; exit 1; }

ARGS=(--policy-path "$CKPT"
      --suites libero_10          # LIBERO-Long, all 10 tasks
      --n-episodes 1              # 1 episode per task
      --obs-size 256
      --n-action-steps 10
      --output-dir "${HERE}/eval_logs/demo")
[[ "${1:-}" == "--headless" ]] && ARGS+=(--no-display)

echo "=== IMS2026 HW0 Task 2 demo — SmolVLA on LIBERO-Long ==="
echo "checkpoint : $CKPT"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | sed 's/^/gpu        : /'
echo
exec python "${HERE}/scripts/eval_liveview.py" "${ARGS[@]}"
