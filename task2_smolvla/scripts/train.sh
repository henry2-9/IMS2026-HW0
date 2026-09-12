#!/usr/bin/env bash
# IMS2026 HW0 Task 2 — Train SmolVLA (0.45B) on LIBERO with LeRobot
#
#   ./train.sh smoke   -> 500 steps, sanity check the whole pipeline
#   ./train.sh full    -> 100k steps, the real reproduction run
#
# Hardware note: measured on the RTX 4090 (24 GB), batch_size=64 with no
# gradient accumulation peaks at ~12.2 GB and ~122 samples/s, so it matches the
# reference recipe directly. (batch 16 x accum 4 gives the same throughput but
# four times as many logged steps per optimizer update.)
set -euo pipefail

MODE="${1:-smoke}"
EXTRA_ARGS=()
HW0_ROOT="/home/yehzr/IMS2026_HW0"
source /home/yehzr/anaconda3/etc/profile.d/conda.sh
conda activate hw0-lerobot

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export HF_HUB_DISABLE_PROGRESS_BARS=1
export TOKENIZERS_PARALLELISM=false

# Pin the dataset revision so the numbers stay comparable (assignment requires
# reproducibility; Hub datasets can be re-uploaded).
DATASET_REPO="lerobot/libero"
DATASET_REV="$(python - <<'PY'
from huggingface_hub import HfApi
print(HfApi().dataset_info("lerobot/libero").sha)
PY
)"
echo "==> dataset revision pinned to ${DATASET_REV}"

case "$MODE" in
  smoke)
    STEPS=500;    SAVE_FREQ=500;    BATCH=64; ACCUM=1
    OUT="${HW0_ROOT}/task2_smolvla/outputs/smoke"
    JOB="smolvla_libero_smoke"
    ;;
  full)
    # SmolVLA defaults scheduler_decay_steps to 30k. The scheduler shortens that
    # when --steps is smaller, but never extends it: training 100k steps with the
    # default leaves the LR pinned at decay_lr=2.5e-6 (1/38 of peak) from step 30k
    # onward, so 70% of the run is wasted. Keep the decay aligned with --steps.
    STEPS=100000; SAVE_FREQ=10000;  BATCH=64; ACCUM=1
    EXTRA_ARGS=(--policy.scheduler_decay_steps=100000)
    OUT="${HW0_ROOT}/task2_smolvla/outputs/libero_smolvla"
    JOB="smolvla_libero_full"
    ;;
  finetune)
    # Reproducing the paper means fine-tuning the PRETRAINED SmolVLA, not
    # training the action expert from scratch. `--policy.type=smolvla` only
    # initialises the VLM backbone from SmolVLM2 and leaves the 100M-parameter
    # action expert random; the paper's LIBERO numbers come from
    # lerobot/smolvla_base, which is pretrained on community robot data.
    # The paper fine-tunes LIBERO for 100k steps at batch 64. STEPS is
    # overridable so the run can be extended with --resume rather than restarted:
    #   FT_STEPS=100000 ./train.sh finetune
    STEPS="${FT_STEPS:-30000}"; SAVE_FREQ=5000; BATCH=64; ACCUM=1
    OUT="${HW0_ROOT}/task2_smolvla/outputs/libero_smolvla_ft"
    JOB="smolvla_libero_finetune"
    if [[ "${FT_RESUME:-false}" == "true" ]]; then
      # lerobot-train raises "A config_path is expected when resuming a run"
      # unless --config_path names the checkpoint's train_config.json (or the
      # pretrained_model/ dir holding it). --resume alone is not enough.
      RESUME_CFG="${OUT}/checkpoints/last/pretrained_model/train_config.json"
      [[ -f "$RESUME_CFG" ]] || { echo "找不到續訓設定: $RESUME_CFG" >&2; exit 1; }
      EXTRA_ARGS=(--resume=true --config_path="$RESUME_CFG")
    fi
    ;;
  *) echo "usage: $0 {smoke|full|finetune}"; exit 1 ;;
esac

# lerobot-train refuses to write into an existing dir, so keep our own run
# metadata in a sibling folder.
META="${OUT}_meta"
mkdir -p "$META"
echo "==> mode=$MODE steps=$STEPS batch=$BATCH x accum=$ACCUM (effective $((BATCH*ACCUM)))"
echo "==> output: $OUT"

# Compute log for the report (assignment requires GPU / VRAM / wall-clock).
nvidia-smi --query-gpu=name,driver_version,memory.total \
  --format=csv > "${META}/gpu_info.csv"
date -Is > "${META}/train_started_at.txt"

# `finetune` starts from the pretrained checkpoint; the other modes build the
# policy from scratch via --policy.type.
if [[ "$MODE" == "finetune" && "${FT_RESUME:-false}" == "true" ]]; then
  # Resuming: TrainPipelineConfig.validate() applies exactly one source, in the
  # order reward-model path -> policy path -> resume. Passing --policy.path here
  # would win and silently skip resume resolution, leaving checkpoint_path None
  # and crashing in resume_before_prepare. The checkpoint already carries the
  # policy config, so pass none of the policy.path args.
  POLICY_ARGS=(--policy.empty_cameras=1)
elif [[ "$MODE" == "finetune" ]]; then
  # smolvla_base was pretrained on SO-101 data: state[6] + camera1/2/3.
  # LIBERO gives state[8] + image/image2. The state difference is a non-issue
  # (SmolVLA pads every state to max_state_dim=32), but the cameras must be
  # reconciled: rename LIBERO's two views onto camera1/camera2, and let
  # empty_cameras=1 substitute a zero-padded, masked-out frame for camera3.
  POLICY_ARGS=(
    --policy.path=lerobot/smolvla_base
    --policy.empty_cameras=1
    --rename_map='{"observation.images.image": "observation.images.camera1", "observation.images.image2": "observation.images.camera2"}'
  )
else
  POLICY_ARGS=(--policy.type=smolvla --policy.load_vlm_weights=true)
fi

lerobot-train \
  "${POLICY_ARGS[@]}" \
  --policy.push_to_hub=false \
  --policy.device=cuda \
  --policy.use_amp=true \
  --dataset.repo_id="${DATASET_REPO}" \
  --dataset.revision="${DATASET_REV}" \
  --dataset.video_backend=torchcodec \
  --batch_size="${BATCH}" \
  --accelerator.gradient_accumulation.steps="${ACCUM}" \
  --steps="${STEPS}" \
  --save_freq="${SAVE_FREQ}" \
  --env_eval_freq=0 \
  --num_workers=8 \
  --output_dir="${OUT}" \
  --job_name="${JOB}" \
  --wandb.enable=false \
  "${EXTRA_ARGS[@]}" \
  2>&1 | tee "${META}/train.log"

date -Is > "${META}/train_finished_at.txt"
echo "==> training done: ${OUT}"
