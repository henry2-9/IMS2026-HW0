#!/usr/bin/env bash
# IMS2026 HW0 Task 2 - sweep n_action_steps.
#
# The SmolVLA paper evaluates in simulation by "sampling new observations and
# predicting a new action after each executed action" -- i.e. re-planning every
# step. Our checkpoints carry the training default n_action_steps=50, which
# executes a whole 50-action chunk open-loop before looking again. This sweep
# measures what that costs.
#
#   ./sweep_action_steps.sh <checkpoint-dir> [episodes-per-task]
set -euo pipefail

CKPT="${1:?usage: $0 <checkpoint-dir> [episodes-per-task]}"
EPS="${2:-3}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

source /home/yehzr/anaconda3/etc/profile.d/conda.sh
conda activate hw0-lerobot
export HF_HUB_DISABLE_PROGRESS_BARS=1 MUJOCO_GL=egl TOKENIZERS_PARALLELISM=false

# smolvla_base-derived checkpoints keep the SO-101 camera names.
RENAME=''
if python3 -c "
import json,sys
c=json.load(open('${CKPT}/config.json'))
sys.exit(0 if 'observation.images.camera1' in c.get('input_features',{}) else 1)
"; then
  RENAME='{"observation.images.image": "observation.images.camera1", "observation.images.image2": "observation.images.camera2"}'
  echo "==> checkpoint uses camera1/2 naming; applying rename map"
fi

for N in 1 5 10 50; do
  OUT="${HERE}/eval_logs/sweep_n${N}"
  echo "=== n_action_steps=${N} ==="
  rm -rf "$OUT"
  ARGS=(--policy-path "$CKPT" --n-episodes "$EPS" --obs-size 256
        --n-action-steps "$N" --output-dir "$OUT" --no-display --no-record)
  [[ -n "$RENAME" ]] && ARGS+=(--rename-map "$RENAME")
  python "${HERE}/scripts/eval_liveview.py" "${ARGS[@]}" 2>&1 | tail -9
done

echo
echo "================ sweep summary ================"
python3 - "$HERE" <<'PY'
import json, sys, pathlib
here = pathlib.Path(sys.argv[1])
order = ["libero_spatial","libero_object","libero_goal","libero_10"]
rows = []
for n in (1,5,10,50):
    f = here/f"eval_logs/sweep_n{n}/eval_info.json"
    if not f.exists():
        continue
    d = json.loads(f.read_text())
    rows.append((n, d["per_suite"], d["aggregate"]["mean_of_suite_success_rates"]))
if rows:
    print(f"{'n_action_steps':>15}" + "".join(f"{s.replace('libero_','')[:9]:>11}" for s in order) + f"{'average':>10}")
    for n, ps, avg in rows:
        print(f"{n:>15}" + "".join(f"{ps[s]['success_rate']:>10.1f}%" for s in order) + f"{avg:>9.1f}%")
    print(f"{'paper':>15}" + "".join(f"{v:>10.1f}%" for v in (90.0,96.0,92.0,71.0)) + f"{87.3:>9.1f}%")
PY
