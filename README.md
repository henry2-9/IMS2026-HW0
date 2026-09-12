# IMS2026 HW0 — Skills & Hardware Verification

Fall 2026 Intelligent Manufacturing Systems (GIMT, NTUST).
Due **9/18**, live online demo with the TA. Worth 20 % of the course grade.

## Machine

| | |
|---|---|
| GPU | RTX 4090, 24 GB — the assignment's stated **bare minimum** |
| Driver | 570.133.07 (CUDA 12.8) |
| RAM | 31 GB — Isaac Sim's stated minimum is 32 GB |
| OS | Ubuntu 20.04 — **Isaac Sim 5.0 requires 22.04/24.04** |

The OS mismatch is the reason Task 1 runs in NVIDIA's Isaac Sim container rather
than natively.

## Layout

```
IMS2026_HW0/
├── docs/
│   ├── 00_host_prereqs.sh        # Docker + NVIDIA Container Toolkit (sudo)
│   └── 01_isaacsim_container.sh  # Isaac Sim 5.0 pull + smoke test
├── task1_ticvla/                 # TIC-VLA × DynaNav × Isaac Sim
│   ├── docker/{Dockerfile,run.sh}
│   ├── checkpoints/              # TIC-VLA ckpt + InternVL3-1B (gitignored)
│   └── README.md
└── task2_smolvla/                # SmolVLA × LIBERO
    ├── docker/{Dockerfile,run.sh}
    ├── scripts/{train.sh,eval_liveview.py}
    └── README.md
```

## Environment gotchas worth remembering

Four things cost real time to diagnose; all are documented in the task READMEs.

1. **PyTorch CUDA build.** LeRobot's plain `pip install` pulls wheels built for
   CUDA 13, which need a ≥ 580 driver. On this 570.x driver they fail with
   *"NVIDIA driver on your system is too old"*. Install the **cu128** wheels —
   LeRobot's own `uv` config targets cu128 (driver floor 570.86).
2. **torchcodec** will not load without FFmpeg 7 shared libs on
   `LD_LIBRARY_PATH`. Handled by a conda `activate.d` hook.
3. **LIBERO blocks on an interactive prompt** on first import. Its config must be
   pre-seeded, or any headless/container run hangs forever.
4. **Isaac Sim needs numpy < 2** — it ships binary modules built against the
   NumPy 1.x ABI.

## Results

**Task 1 — TIC-VLA on DynaNav** (8 episodes: 4 scenes x 2 robot platforms)

| metric | value |
|---|---|
| Success Rate | 50.0 % (4/8) |
| Collision Rate | **0.0 %** |
| Avg Navigation Error | 9.45 m |
| Avg SPL | 0.440 |

Nova Carter 3/4, Spot 1/4, every episode collision-free. Details in
[`task1_ticvla/README.md`](task1_ticvla/README.md).

**Task 2 — SmolVLA on LIBERO** (400 episodes per seed, 3 seeds, `n_action_steps=10`)

| Suite | Ours (3-seed mean) | Paper | within ±3 pp |
|---|---|---|---|
| LIBERO-Spatial | 80.0 % | 90 % | no |
| LIBERO-Object | **95.0 %** | 96 % | **yes** |
| LIBERO-Goal | **92.3 %** | 92 % | **yes** |
| LIBERO-Long | 75.0 % | 71 % | no (overshoots by 4.0) |
| **Average** | **85.6 %** | 87.3 % | |

The average lands 1.7 pp under the paper. LIBERO-Long misses the band by beating
it. LIBERO-Spatial is the one real shortfall: it reads 78–82 % across eight
measurements spanning models whose training loss differed by 2.8x, so it is not a
training-budget problem. Details, the learning-rate schedule that was worth 2.7 pp
of average, and the head-to-head against fine-tuning `smolvla_base`, in
[`task2_smolvla/README.md`](task2_smolvla/README.md).

See [`DEMO.md`](DEMO.md) for the live-demo runbook.

## Status

Both tasks are complete: benchmarks run, metrics collected, videos rendered,
and every environment fix documented in the per-task READMEs.

Artifacts **not** in this repository, per the assignment: the LIBERO dataset,
model checkpoints, and rendered videos. Those are on the Hugging Face Hub:

**[iug8oyo8/IMS2026-HW0-media](https://huggingface.co/datasets/iug8oyo8/IMS2026-HW0-media)** — the Task 1 DynaNav videos (8 episodes,
four scenes x two robot platforms, RGB beside third-person) and the Task 2 LIBERO
live-view recordings.
