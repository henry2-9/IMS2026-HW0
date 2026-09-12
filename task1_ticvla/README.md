# IMS2026 HW0 — Task 1: TIC-VLA Navigation on DynaNav (Isaac Sim 5.0.0)

Vision-Language-Action navigation with [TIC-VLA](https://github.com/ucla-mobility/TIC-VLA),
evaluated on the DynaNav benchmark inside Isaac Sim 5.0.0, across **four distinct
scenes** and **two robot platforms**.

## The blocker, and why this runs in a container

Isaac Sim 5.0.0 supports **Ubuntu 22.04/24.04 only**. This machine runs Ubuntu
20.04 (glibc 2.31), so a native install is not possible. The Isaac Sim container
ships its own 22.04 userspace, so the host distro stops mattering — only the
NVIDIA driver and `nvidia-container-toolkit` do. A Dockerfile is a required
deliverable anyway, so this is the same work either way.

## Setup

```bash
# 1. Host prerequisites (Docker + NVIDIA Container Toolkit) — needs sudo
bash ../docs/00_host_prereqs.sh

# 2. NGC login, required to pull nvcr.io/nvidia/isaac-sim:5.0.0
docker login nvcr.io -u '$oauthtoken' -p <NGC_API_KEY>

# 3. Build
./docker/run.sh build
```

### Model weights

Downloaded to `checkpoints/` (not committed):

| File | Source | Size |
|---|---|---|
| `TIC-VLA-model.ckpt` | `handsomeYun/TIC-VLA` (dataset repo) | 1.94 GB |
| `InternVL3-1B/` | `OpenGVLab/InternVL3-1B` | 1.89 GB |

```bash
python - <<'PY'
from huggingface_hub import hf_hub_download, snapshot_download
hf_hub_download("handsomeYun/TIC-VLA", "TIC-VLA-model.ckpt",
                repo_type="dataset", local_dir="checkpoints")
snapshot_download("OpenGVLab/InternVL3-1B", local_dir="checkpoints/InternVL3-1B")
PY
```

> The `handsomeYun/TIC-VLA` repo also hosts ~538 GB of training data
> (DynaNav / GND / SCAND zips). **None of it is needed** — this task is
> inference and evaluation only.

## Running the benchmark

```bash
./docker/run.sh bench                                   # HW0 8-episode set
./docker/run.sh bench DynaNav/configs/benchmark_full.yaml   # full 85 episodes
```

### `benchmark_hw0.yaml` — why it exists

The assignment requires four scenes evaluated on **two** robot platforms
(8 episodes). The shipped `benchmark_full.yaml` covers all four scenes but its
85 episodes are **all `nova_carter`**. The DynaNav runner itself supports both
platforms — see `DynaNav/benchmark.py::_spawn_robot`, which resolves `Spot` and
`Nova_Carter` USD assets — so `benchmark_hw0.yaml` pairs one representative
episode per scene with each platform:

| Episode | Platform | Scene | Timeout |
|---|---|---|---|
| 1 | nova_carter (wheeled) | hospital | 40 |
| 2 | nova_carter | warehouse | 70 |
| 3 | nova_carter | office | 50 |
| 4 | nova_carter | outdoor | 100 |
| 5 | spot (quadruped) | hospital | 60 |
| 6 | spot | warehouse | 105 |
| 7 | spot | office | 75 |
| 8 | spot | outdoor | 150 |

Spot timeouts are 1.5× the Carter values because the quadruped covers the same
route more slowly.

### Verified container state

Isaac Sim's `python.sh` sources `setup_python_env.sh` to put its bundled
packages on `PYTHONPATH`; running the bare `python3` misses them and reports
false negatives. With the env sourced, the image resolves:

| package | version | note |
|---|---|---|
| numpy | 1.26.0 | the `<2` pin held — required by Isaac's NumPy 1.x ABI |
| torch | 2.7.0+cu128 | Isaac's bundled build, matches the 570 driver |
| transformers | 4.57.6 | |
| timm | 1.0.29 | |
| cv2 | 4.11.0 | |
| ticvla | **fails** | `No module named 'pytorch_lightning'` — harmless, see below |

`pytorch_lightning` appears **only** in training code (`ticvla/data/vlm_data.py`,
`ticvla/data/policy_data.py`, `ticvla/training/train.py`). The benchmark path is
`benchmark.py` → `DynaNav/ticvla.py` → `DynaNav/ticvla_vlm.py`, which needs only
`torch` and `transformers`; no DynaNav module imports the `ticvla` package. So
the failing import is never on the evaluation path.

Do **not** "fix" it by installing `requirements-train.txt` — it pins
`torch==2.8.0+cu128` and would replace Isaac's own 2.7.0 stack.

Asset availability checked: Spot and Nova Carter USDs both return HTTP 200, and
the local `office.usd` / `outdoor_small.usd` scenes are present.

### The base image ignores your command

`nvcr.io/nvidia/isaac-sim:5.0.0` sets

```
ENTRYPOINT ["/bin/sh","-c","/isaac-sim/runheadless.sh"]
```

Because that is the *shell* form, any `CMD` — or any command appended to
`docker run` — is silently discarded, and the container always launches the
stock streaming app instead. Everything looks healthy while this happens: the
container starts, Isaac Sim loads, the GPU is busy, the log ends with
"Isaac Sim Full Streaming App is loaded". It simply never runs your benchmark.

The Dockerfile here resets `ENTRYPOINT []`, and `docker/run.sh` also passes
`--entrypoint /bin/bash` so an older image still behaves.

### Output paths that need mounting

Two directories hold deliverables and both live inside the image by default:

| path (in container) | contents |
|---|---|
| `DynaNav/benchmark_results/<run_id>/` | results JSON + text summary |
| `DynaNav/logs/<run_id>/{carter,spot}_ticvla_data/` | `head_frame_*.jpg` (robot RGB) and `tp_frame_*.jpg` (third person) |

`docker/run.sh` mounts both out to `outputs/`. Without that, a `--rm` container
discards every frame and every number it produced.

### Every episode overwrites the previous one's frames

All episodes share `logs/<run_id>/<robot>_ticvla_data/` and restart their frame
numbering at zero, so episode N+1 silently overwrites episode N. Left alone, an
8-episode run ends with only the last Carter episode and the last Spot episode
on disk — six of the eight visual records gone.

`scripts/archive_episode_frames.sh` watches the benchmark log and moves each
episode's frames into `outputs/episode_frames/<episode>/` as the next one
starts. Run it alongside the benchmark.

Separately, the robot-camera stream is the model's own input buffer: the
behaviour scripts keep only `_max_history_frames` (190) frames and delete the
rest. The copies under `DynaNav/behavior/` are patched to move those frames into
`rgb_keep/` instead of deleting them, which leaves the model's buffer behaviour
untouched but preserves the RGB deliverable. `docker/run.sh` mounts that
directory over the image's copy, so the patch applies without a rebuild.

### Episodes run as subprocesses

`benchmark.py` launches each episode as its own Isaac Sim subprocess, so the
~3-minute startup is paid once per episode. Budget roughly 1.5–2 hours for the
8-episode HW0 set, not 20 minutes.

### Two gotchas

* `DynaNav/run_benchmark.sh` defaults to `CUDA_VISIBLE_DEVICES=1`. This machine
  has a single GPU, so `docker/run.sh` exports `CUDA_VISIBLE_DEVICES=0`.
* `requirements-test.txt` pins `numpy<2` — Isaac Sim 5.0 ships binary modules
  built against the NumPy 1.x ABI. Do **not** install `requirements-train.txt`
  into Isaac's Python; it replaces Isaac's CUDA/PyTorch stack.

## Results

`benchmark_hw0.yaml`, 8 episodes, TIC-VLA pretrained checkpoint, Isaac Sim 5.0.0
in the container, RTX 4090 / driver 570.133.07.

| # | platform | scene | outcome | duration | nav. error |
|---|---|---|---|---|---|
| 1 | Nova Carter | hospital | **success** | 18.5 s | — |
| 2 | Nova Carter | warehouse | timeout | 70.0 s | — |
| 3 | Nova Carter | office | **success** | 21.2 s | — |
| 4 | Nova Carter | outdoor (200 people) | **success** | 23.4 s | — |
| 5 | Spot | hospital | fail | 60.0 s | 22.95 m |
| 6 | Spot | warehouse | **success** | 58.6 s | — |
| 7 | Spot | office | fail | 75.0 s | 6.96 m |
| 8 | Spot | outdoor (200 people) | timeout | 150.0 s | — |

| metric | value |
|---|---|
| Success Rate | **50.0 %** (4/8) |
| Collision Rate (human or physical) | **0.0 %** |
| Average Navigation Error | 9.45 m |
| Average SPL | 0.440 |
| Average Path Length | 22.02 m |

Nova Carter reaches 3/4, Spot 1/4. Every episode is collision-free, including
the two outdoor scenes populated with 200 pedestrians.

The two platforms fail on *different* scenes — Carter misses warehouse, Spot
misses hospital — so Spot's lower score is not a systematic "the quadruped
cannot walk" failure. Spot walks correctly under
`SpotFlatTerrainPolicy`; it navigates less accurately. Note that the upstream
benchmark ships 85 episodes and every one of them is `nova_carter`, so the
pretrained checkpoint has seen far more wheeled-base experience.

**Videos — [watch them in the browser](https://huggingface.co/spaces/iug8oyo8/IMS2026-HW0-videos)**, one clip per episode with the
RGB view on the left and a third-person view on the right. The mp4s themselves are
in the [media dataset](https://huggingface.co/datasets/iug8oyo8/IMS2026-HW0-media/tree/main/task1_ticvla_dynanav); the Space just plays
them inline, since a dataset repo only offers files for download. Videos are hosted
rather than committed, as the assignment requires. Rebuild locally with
`./scripts/build_deliverables.sh`.

## Live demo checklist

The terminal must show, during the session:

* real-time inference logs — VLM outputs and velocity commands
* final aggregated metrics (Success Rate, Collision Rate, …) over the 8 episodes

Visual deliverables: a GIF/video/image sequence covering both platforms across
all four scenes, showing **RGB view** and **third-person view**.

## Hardware

Isaac Sim VLA evaluation requires ≥ 24 GB VRAM. This machine has an RTX 4090
(24 GB) — the stated minimum, with no headroom.
