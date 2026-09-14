#!/usr/bin/env bash
# IMS2026 HW0 Task 2 - the live TA demo.
#
# Defaults to exactly what the assignment asks for during the session:
# LIBERO-Long only, 10 tasks, 1 episode each, with the live-view window and the
# terminal evaluation output both visible.
#
#   ./demo.sh                           the graded demo run (LIBERO-Long)
#   ./demo.sh --suite libero_goal       a different suite, if asked
#   ./demo.sh --suite all               all four suites
#   ./demo.sh --episodes 3              more episodes per task
#   ./demo.sh --headless                same numbers, no window (dry run over SSH)
#
# Suite names are LIBERO's own. Note that LIBERO-Long is `libero_10`:
#   libero_spatial  libero_object  libero_goal  libero_10
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

SUITE="libero_10"
EPISODES=1
DISPLAY_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --suite)    SUITE="${2:?--suite needs a value}"; shift 2 ;;
    --episodes) EPISODES="${2:?--episodes needs a value}"; shift 2 ;;
    --headless) DISPLAY_ARGS+=(--no-display); shift ;;
    -h|--help)  sed -n '2,/^[^#]/p' "${BASH_SOURCE[0]}" | grep '^#' | sed 's/^# \?//'; exit 0 ;;
    *)          echo "unknown option: $1" >&2; exit 1 ;;
  esac
done

# Accept the friendlier spellings people reach for, and "all".
case "$SUITE" in
  all)                    SUITE="libero_spatial,libero_object,libero_goal,libero_10" ;;
  libero_long|long|LONG)  SUITE="libero_10" ;;
  spatial)                SUITE="libero_spatial" ;;
  object)                 SUITE="libero_object" ;;
  goal)                   SUITE="libero_goal" ;;
esac

source /home/yehzr/anaconda3/etc/profile.d/conda.sh
conda activate hw0-lerobot
export HF_HUB_DISABLE_PROGRESS_BARS=1 MUJOCO_GL=egl TOKENIZERS_PARALLELISM=false

CKPT="${HERE}/outputs/libero_smolvla/checkpoints/100000/pretrained_model"
[[ -d "$CKPT" ]] || { echo "checkpoint not found: $CKPT" >&2; exit 1; }

echo "=== IMS2026 HW0 Task 2 demo — SmolVLA on LIBERO ==="
echo "checkpoint : $CKPT"
echo "suite(s)   : $SUITE"
echo "episodes   : $EPISODES per task"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | sed 's/^/gpu        : /'
echo

exec python "${HERE}/scripts/eval_liveview.py" \
  --policy-path "$CKPT" \
  --suites "$SUITE" \
  --n-episodes "$EPISODES" \
  --obs-size 256 \
  --n-action-steps 10 \
  --output-dir "${HERE}/eval_logs/demo" \
  "${DISPLAY_ARGS[@]}"
