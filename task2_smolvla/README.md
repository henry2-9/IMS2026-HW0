# IMS2026 HW0 — Task 2: SmolVLA (0.45B) × LIBERO Reproduction

Reproduction of the LIBERO results reported in
[SmolVLA (arXiv:2506.01844)](https://arxiv.org/abs/2506.01844), Table 2, using the
official [LeRobot](https://github.com/huggingface/lerobot) implementation.

## Results

Submitted model: **`--policy.type=smolvla`, 100k steps, batch 64,
`scheduler_decay_steps=100000`**, evaluated over the full protocol (4 suites x
10 tasks x 10 episodes = 400 episodes per seed) with `n_action_steps=10` and
256x256 observations. LeRobot's LIBERO guide recommends averaging over three
seeds, so all three are reported.

Checkpoint: [`task2_smolvla_libero/pretrained_model`](https://huggingface.co/datasets/iug8oyo8/IMS2026-HW0-media/tree/main/task2_smolvla_libero)

| Suite | seed 42 | seed 43 | seed 44 | mean | sd | Paper | Delta | within ±3 pp |
|---|---|---|---|---|---|---|---|---|
| LIBERO-Spatial | 82 % | 80 % | 78 % | 80.0 % | 1.6 | 90 % | −10.0 | no |
| LIBERO-Object | 98 % | 94 % | 93 % | **95.0 %** | 2.2 | 96 % | −1.0 | **yes** |
| LIBERO-Goal | 93 % | 90 % | 94 % | **92.3 %** | 1.7 | 92 % | +0.3 | **yes** |
| LIBERO-Long | 71 % | 75 % | 79 % | 75.0 % | 3.3 | 71 % | **+4.0** | no |
| **Average** | 86.0 % | 84.8 % | 86.0 % | **85.6 %** | | 87.3 % | −1.7 | |

The average sits 1.7 pp below the paper. Two suites are inside the ±3 pp band —
and LIBERO-Long misses it by **overshooting**: at 75.0 % it beats the paper's
71.0 % by 4 points, which the band counts as a miss in either direction.

**LIBERO-Spatial is the one genuine shortfall.** It reads 78–82 % across eight
measurements spanning two models whose final training loss differed by 2.8x
(0.084 vs 0.234), while every other suite moved with training quality. Whatever
limits it is not training budget. It is not the training setup either: the paper
also trains a single multi-task model over all 40 tasks ("SmolVLA is always
trained in a multi-task setting", 1,693 episodes), which is exactly what
`lerobot/libero` and the command below do.

### What the learning-rate schedule was worth

The same recipe evaluated before and after aligning `scheduler_decay_steps` with
`--steps` (3-seed means):

| Suite | decay=30k | decay=100k | Paper |
|---|---|---|---|
| LIBERO-Spatial | 80.7 % | 80.0 % | 90 % |
| LIBERO-Object | 96.3 % | 95.0 % | 96 % |
| LIBERO-Goal | 89.3 % | **92.3 %** | 92 % |
| LIBERO-Long | 65.3 % | **75.0 %** | 71 % |
| **Average** | 82.9 % | **85.6 %** | 87.3 % |
| final train loss | 0.234 | **0.084** | — |

LIBERO-Long gains 9.7 pp and LIBERO-Goal 3.0 pp; Spatial does not move at all.

### Two initialisations, measured head to head

Fine-tuning the pretrained `lerobot/smolvla_base` instead of training the action
expert from scratch, at identical settings (100k steps, 400 episodes, seed 42,
`n_action_steps=10`, both with the 30k decay schedule):

| Suite | from scratch | fine-tuned from `smolvla_base` | Paper |
|---|---|---|---|
| LIBERO-Spatial | 80 % | 81 % | 90 % |
| LIBERO-Object | 98 % | 97 % | 96 % |
| LIBERO-Goal | 94 % | 87 % | 92 % |
| LIBERO-Long | 61 % | 59 % | 71 % |
| **Average** | **83.2 %** | 81.0 % | 87.3 % |

From-scratch is ahead by 2.2 pp on that seed, which is **within the noise** — the
seed-to-seed spread of its own average is 81.5–84.0 %. So neither initialisation
is shown to be better; fine-tuning `smolvla_base` simply brings no benefit worth
its cost, since adapting an SO-101-pretrained checkpoint to LIBERO needs a padded
third camera, raising peak VRAM from 12.2 GB to 21.5 GB and dropping throughput
from 127 to 77 samples/s.

The trap worth recording: the from-scratch model first measured at 67.0 %, which
prompted the switch to `smolvla_base`. That measurement used the checkpoint's
default `n_action_steps=50`. Comparing two models under different inference
settings produced a conclusion a controlled rerun did not support — and the
22 GPU-hours spent fine-tuning bought nothing.

## 1. Environment

Reference machine: RTX 4090 (24 GB), driver 570.133.07, Ubuntu 20.04, 32 GB RAM.

> **Driver note.** LeRobot's default `pip install` pulls PyTorch built for CUDA 13,
> which needs a ≥ 580 driver. On a 570.x driver that fails with
> *"NVIDIA driver on your system is too old"*. Install the **cu128** wheels
> instead — LeRobot's own `uv` config targets cu128 (driver floor 570.86).

### Docker (recommended — one-command reproduction)

```bash
cd docker
./run.sh build
./run.sh shell
```

### Conda (what was actually used for the run below)

```bash
conda create -y -n hw0-lerobot python=3.12
conda activate hw0-lerobot

git clone https://github.com/huggingface/lerobot.git
cd lerobot && git checkout 71a11efe77f55e61f3ab2ce45b40da8cf626afa9
pip install -e ".[smolvla,libero]"

# force the CUDA 12.8 build (see driver note above)
pip install --force-reinstall torch==2.11.0 torchvision==0.26.0 \
    --index-url https://download.pytorch.org/whl/cu128
pip install "numpy>=2.0.0,<2.3.0"

# torchcodec needs FFmpeg 7 shared libs on the loader path
conda install -y -c conda-forge "ffmpeg=7.*"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}"

# headless MuJoCo rendering
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl

# LIBERO prompts for a dataset path on first import — answer N once
echo N | python -c "import libero.libero"
```

## 2. Data

[`lerobot/libero`](https://huggingface.co/datasets/lerobot/libero) — 1,693 episodes,
273,465 frames, 40 tasks, 2 × 256×256 cameras, MP4-encoded (1.9 GB).
It is downloaded automatically on first use and the **revision is pinned** by
`scripts/train.sh` so the numbers stay comparable.

The dataset is **not** committed to this repository.

## 3. Matching the paper's recipe

Three details in the SmolVLA paper (arXiv:2506.01844) are easy to miss and all
of them change the result. They are recorded here because each one cost a
training run to discover.

**Training from scratch is at least as good as fine-tuning — see Results.**
`--policy.type=smolvla --policy.load_vlm_weights=true` initialises the VLM
backbone from SmolVLM2 and leaves the 100M-parameter action expert random.
Fine-tuning `lerobot/smolvla_base` instead sounds closer to the paper, but
measured head to head it scores 4.5 pp lower and costs nearly twice the VRAM.
The `finetune` mode below is kept so the comparison can be reproduced.

`smolvla_base` was pretrained on SO-101 data (`state[6]`, `camera1/2/3`) while
LIBERO gives `state[8]` and two views. The state difference is harmless — SmolVLA
pads every state to `max_state_dim=32` — but the cameras need
`--rename_map` onto `camera1`/`camera2` plus `--policy.empty_cameras=1` to
substitute a zero-padded, masked-out frame for the third view. That third view
costs real memory: peak VRAM rises from 12.2 GB to 21.5 GB and throughput drops
from 127 to 77 samples/s.

**100k steps, batch 64.** The paper fine-tunes LIBERO for 100,000 steps.

**Re-plan every step at evaluation.** The paper evaluates in simulation by
"sampling new observations and predicting a new action after each executed
action" — `n_action_steps=1`. Checkpoints carry the training default of
`n_action_steps=50`, which executes a whole chunk open-loop before looking
again. (LeRobot's own Pi0.5 reproduction likewise overrides this, to 10.)
`scripts/sweep_action_steps.sh` measures the effect.

The action statistics of `lerobot/libero` (xyz ±0.94, rotation ±0.38, gripper
±1.0, mean ≈ 0) confirm delta actions, so the default
`--env.control_mode=relative` is correct and needs no change.

## 4. Training

```bash
./scripts/train.sh smoke   # 500 steps — validates the pipeline
./scripts/train.sh full    # 100k steps — the reproduction run
```

`batch_size=64` with no gradient accumulation, matching the reference recipe
directly. Measured on the RTX 4090: **12.2 GB peak VRAM, ~122 samples/s**, so
100k steps takes **≈ 14 hours**. (batch 16 × accum 4 reaches the same effective
batch and the same throughput, but logs four steps per optimizer update.)

Checkpoint verified at **450,046,662 parameters (0.450 B)**, satisfying the
"approximately 0.45B" requirement.

## 5. Evaluation

`scripts/eval_liveview.py` wraps LeRobot's own env/policy/processor stack — the same
construction `lerobot-eval` uses — but drives the rollout loop directly so it can
render the live-view window the assignment requires:

* front (agentview) and wrist (eye-in-hand) camera panes
* task suite and task instruction
* current episode index and step count / step limit
* per-task and running total success statistics

Every episode is written to an mp4, and results are aggregated into `eval_info.json`.

```bash
# full protocol: 4 suites × 10 tasks × 10 episodes = 400 episodes
python scripts/eval_liveview.py \
    --policy-path outputs/libero_smolvla/checkpoints/last/pretrained_model \
    --output-dir eval_logs/full

# TA live demo: LIBERO-Long only, 1 episode per task
python scripts/eval_liveview.py \
    --policy-path outputs/libero_smolvla/checkpoints/last/pretrained_model \
    --suites libero_10 --n-episodes 1 \
    --output-dir eval_logs/demo
```

Evaluation uses `init_states=true` and `hard_reset=true`, matching the published
protocol (soft resets are faster but not bit-identical).

### Render resolution must match the training data

LeRobot's `LiberoEnv` defaults to **360×360**, but the `lerobot/libero` dataset —
and therefore the trained checkpoint's `input_features` — is **256×256**. Leaving
them mismatched is a train/eval distribution shift, so `eval_liveview.py` defaults
`--obs-size` to 256.

Measured on the 40k-step checkpoint, 30 episodes per suite (± = standard error):

| suite | 360×360 | 256×256 | diff |
|---|---|---|---|
| LIBERO-Spatial | 60.0 % ±8.9 | 73.3 % ±8.1 | +13.3 |
| LIBERO-Object | 66.7 % ±8.6 | 60.0 % ±8.9 | −6.7 |
| LIBERO-Goal | 56.7 % ±9.0 | 86.7 % ±6.2 | **+30.0** |
| LIBERO-Long | 40.0 % ±8.9 | 36.7 % ±8.8 | −3.3 |
| average | 55.8 % | 64.2 % | **+8.3** |

Only the LIBERO-Goal gap exceeds the noise floor (≈2.7σ); the two negative deltas
are well inside it. The effect is **not** specific to spatial reasoning — matching
the resolution helps broadly.

Cross-check against the official harness on a subset:

```bash
lerobot-eval --policy.path=<ckpt> --env.type=libero \
    --env.task=libero_10 --eval.batch_size=1 --eval.n_episodes=2 \
    --env.max_parallel_tasks=1
```

## 6. Layout

```
task2_smolvla/
├── docker/
│   ├── Dockerfile          # cu128 base, FFmpeg 7, EGL, LIBERO pre-seeded
│   └── run.sh              # build | shell | eval
├── scripts/
│   ├── train.sh            # smoke | full
│   └── eval_liveview.py    # live-view evaluation + eval_info.json
├── outputs/                # checkpoints (gitignored)
└── eval_logs/              # eval_info.json + videos (gitignored)
```

Per assignment rules, datasets, checkpoints and evaluation videos are **not**
committed; they are shared via Hugging Face Hub / cloud-drive links.
