#!/usr/bin/env bash
# Build / run the Task 1 (TIC-VLA + Isaac Sim) container.
#   ./run.sh build              -> build the image
#   ./run.sh shell              -> interactive shell
#   ./run.sh bench [config]     -> run the DynaNav benchmark (default: HW0 8-episode set)
set -euo pipefail

# Until the docker group membership takes effect (needs a fresh login), fall
# back to sudo. Override explicitly with DOCKER="sudo docker" if needed.
DOCKER="${DOCKER:-docker}"
# -it needs a terminal; without one (background runs, CI) docker refuses to start.
TTY_ARGS=()
[[ -t 0 ]] && TTY_ARGS+=(-i -t)
if ! docker info >/dev/null 2>&1; then DOCKER="sudo docker"; fi
IMAGE="ims2026-hw0-task1:latest"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CACHE="${HOME}/docker/isaac-sim"

mkdir -p "${CACHE}"/{cache/kit,cache/ov,cache/pip,cache/glcache,cache/computecache,logs,data,documents} \
         "${HERE}/outputs/benchmark_results" "${HERE}/outputs/frames" "${HERE}/outputs/logs"

# `episode` mode writes into its own logs dir; LOGS_SRC must be set before
# common_args is built, since docker rejects two mounts on the same target.
if [[ "${1:-}" == "episode" && -n "${2:-}" ]]; then
  LOGS_SRC="${HERE}/outputs/logs_${2}"
  mkdir -p "$LOGS_SRC"
fi

common_args=(
  "${TTY_ARGS[@]}"
  --rm --gpus all --network host --ipc=host
  -e "ACCEPT_EULA=Y" -e "PRIVACY_CONSENT=Y"
  -e "DISPLAY=${DISPLAY:-}"
  # The runner defaults to CUDA_VISIBLE_DEVICES=1; this box has a single GPU.
  -e "CUDA_VISIBLE_DEVICES=0"
  -e "TICVLA_BASE_MODEL_PATH=/workspace/checkpoints/InternVL3-1B"
  -e "TICVLA_CHECKPOINT_PATH=/workspace/checkpoints/TIC-VLA-model.ckpt"
  -e "TICVLA_OUTPUT_DIR=/workspace/outputs"
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw
  -v "${CACHE}/cache/kit:/isaac-sim/kit/cache:rw"
  -v "${CACHE}/cache/ov:/root/.cache/ov:rw"
  -v "${CACHE}/cache/pip:/root/.cache/pip:rw"
  -v "${CACHE}/cache/glcache:/root/.cache/nvidia/GLCache:rw"
  -v "${CACHE}/cache/computecache:/root/.nv/ComputeCache:rw"
  -v "${CACHE}/logs:/root/.nvidia-omniverse/logs:rw"
  -v "${CACHE}/data:/root/.local/share/ov/data:rw"
  -v "${CACHE}/documents:/root/Documents:rw"
  -v "${HERE}/checkpoints:/workspace/checkpoints:rw"
  -v "${HERE}/outputs:/workspace/outputs:rw"
  # benchmark.py writes JSON/text results to ./benchmark_results relative to
  # TICVLA_DYNANAV_ROOT. That path is inside the image, so without this mount
  # every result is discarded when the --rm container exits.
  -v "${HERE}/outputs/benchmark_results:/workspace/TIC-VLA/DynaNav/benchmark_results:rw"
  # The replicator writes its RGB frames to <config dir>/tmp/<run_id>/benchmark_output,
  # which is also inside the image. Those frames are the assignment's visual
  # deliverable, so mount that tree out as well.
  -v "${HERE}/outputs/frames:/workspace/TIC-VLA/DynaNav/configs/tmp:rw"
  # The TIC-VLA behavior scripts write the two required camera views --
  # head_frame_*.jpg (robot RGB) and tp_frame_*.jpg (third person) -- under
  # DynaNav/logs/<run_id>/. That is the assignment's visual deliverable.
  -v "${LOGS_SRC:-${HERE}/outputs/logs}:/workspace/TIC-VLA/DynaNav/logs:rw"
  # Mount the behavior scripts over the baked-in copy so they can be edited
  # without rebuilding the ~25 GB image.
  -v "${HERE}/TIC-VLA/DynaNav/behavior:/workspace/TIC-VLA/DynaNav/behavior:ro"
)

case "${1:-shell}" in
  build)
    ${DOCKER} build -f "${HERE}/docker/Dockerfile" -t "${IMAGE}" "${HERE}"
    ;;
  shell)
    xhost +local:docker >/dev/null 2>&1 || true
    ${DOCKER} run "${common_args[@]}" --entrypoint /bin/bash "${IMAGE}"
    ;;
  bench)
    CONFIG="${2:-DynaNav/configs/benchmark_hw0.yaml}"
    xhost +local:docker >/dev/null 2>&1 || true
    ${DOCKER} run "${common_args[@]}" --entrypoint /bin/bash "${IMAGE}" \
      -lc "DynaNav/run_benchmark.sh ${CONFIG}"
    ;;
  episode)
    # Re-run a single episode. --episode_name only takes effect in child mode:
    # without --child (and a --result_json path) the parent ignores it and
    # re-runs the whole config, which is how a "rerun episode_1" silently
    # became a failed 8-episode run.
    EP="${2:?usage: $0 episode <episode_name>}"
    mkdir -p "${HERE}/outputs/logs_${EP}"
    ${DOCKER} run "${common_args[@]}" \
      --entrypoint /bin/bash "${IMAGE}" -lc \
      "cd DynaNav && \${ISAAC_SIM_PYTHON} benchmark.py \
         -c configs/benchmark_hw0.yaml --navigation_method ticvla \
         --child --episode_name ${EP} --result_json /workspace/outputs/${EP}_result.json"
    ;;
  *) echo "usage: $0 {build|shell|bench [config]|episode <name>}"; exit 1 ;;
esac
