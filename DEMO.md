# IMS2026 HW0 — Live Demo Runbook

What the TA must see, and the exact commands that show it. Rehearse this end to
end before the session; both demos start a GPU process that takes minutes to
warm up.

---

## Task 2 — SmolVLA on LIBERO-Long

**Required on screen:** the live-view window, and the terminal evaluation output.
**Scope (per the assignment):** LIBERO-Long only, 10 tasks, 1 episode each.

```bash
cd task2_smolvla
./scripts/demo.sh                      # the graded run: LIBERO-Long, 10 tasks x 1 episode
```

If the TA asks for a different suite:

```bash
./scripts/demo.sh --suite libero_goal  # or libero_spatial / libero_object / libero_10
./scripts/demo.sh --suite all          # all four
./scripts/demo.sh --episodes 3         # more episodes per task
```

LIBERO's own name for LIBERO-Long is **`libero_10`**, not `libero_long` — the
script accepts both, plus the bare words `spatial`, `object`, `goal`, `long`.

Runtime ≈ 1–2 minutes. The window shows, for every step:

* front (agentview) and wrist (eye-in-hand) camera panes
* the task suite and the natural-language instruction
* current episode and step count / step limit
* per-task and running total success statistics

The terminal prints one line per episode and finishes with the four-suite
summary table plus `eval_info.json`.

> Press **q** in the window to abort early; partial results are still written.

### If the window cannot open

Over SSH or without an X display, `./scripts/demo.sh --headless` produces
identical numbers and records the same frames to mp4.

---

## Task 1 — TIC-VLA on DynaNav (Isaac Sim 5.0.0)

**Required on screen:** real-time inference logs (VLM output and velocity
commands) and the final aggregated metrics over the 8 episodes.

```bash
cd task1_ticvla
./docker/run.sh bench                      # all 8 episodes
./docker/run.sh episode episode_1          # a single episode
```

Runtime ≈ 1.5–2 hours for all 8 — **run this before the session** and present
the saved results, or demo a single episode live (≈ 10 minutes, of which ~3 are
Isaac Sim startup).

The terminal streams lines like:

```
[TIC-VLA] Delay: current_step=313, time_delay=1.800s
[TIC-VLA] robot_state (input): tensor([...], device='cuda:0')
v_cmd=1.500, w_cmd=-0.011
p_cur=[3.437, -0.443, 0.0]
```

A **single** episode runs in child mode, which writes its result to JSON rather
than printing the aggregate block. The TA needs the 8-episode metrics on screen,
so show the saved summary alongside the live run:

```bash
cat outputs/benchmark_results/*/benchmark_hw0_results_latest.txt
```

The full `./docker/run.sh bench` does end with that aggregate block itself
(Success Rate, Collision Rate, SPL, Navigation Error).

Per-episode videos, RGB on the left and third-person on the right:

```bash
./scripts/build_deliverables.sh     # writes outputs/videos/episode_<n>_<robot>.mp4
```

---

## Rehearsal timings (measured)

| step | time |
|---|---|
| Task 2 demo — 10 LIBERO-Long tasks, 1 episode each | **2.4 min** |
| Task 1 — one episode, of which ~2.8 min is Isaac Sim startup | **~4 min** |
| Task 1 — all 8 episodes | 1.5–2 h · run beforehand |

## Pre-flight checklist

| check | command | expected |
|---|---|---|
| GPU free | `nvidia-smi` | < 1 GB in use; the demo needs headroom |
| conda env | `conda activate hw0-lerobot && python -c "import torch;print(torch.cuda.is_available())"` | `True` |
| checkpoint present | `ls task2_smolvla/outputs/libero_smolvla/checkpoints/100000/pretrained_model` | config.json, model.safetensors |
| docker without sudo | `docker info` | succeeds (else `sg docker -c '<cmd>'`) |
| Isaac Sim image | `docker images ims2026-hw0-task1` | present |
| TIC-VLA weights | `ls task1_ticvla/checkpoints` | `TIC-VLA-model.ckpt`, `InternVL3-1B/` |

Nothing here downloads at demo time **if** the caches are warm — but the first
Isaac Sim launch on a cold `~/docker/isaac-sim` cache pulls scene assets and can
take much longer. Warm it the day before.
