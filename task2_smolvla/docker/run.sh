#!/usr/bin/env bash
# Build and enter the Task 2 container.
#   ./run.sh build          -> build the image
#   ./run.sh shell          -> interactive shell (GPU + caches mounted)
#   ./run.sh eval <ckpt>    -> run the live-view eval inside the container
set -euo pipefail

# Until the docker group membership takes effect (needs a fresh login), fall
# back to sudo. Override explicitly with DOCKER="sudo docker" if needed.
DOCKER="${DOCKER:-docker}"
# -it needs a terminal; without one (background runs, CI) docker refuses to start.
TTY_ARGS=()
[[ -t 0 ]] && TTY_ARGS+=(-i -t)
if ! docker info >/dev/null 2>&1; then DOCKER="sudo docker"; fi
IMAGE="ims2026-hw0-task2:latest"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

case "${1:-shell}" in
  build)
    ${DOCKER} build -f "${HERE}/docker/Dockerfile" -t "${IMAGE}" "${HERE}"
    ;;
  shell|eval)
    # xhost is only needed for the on-screen live view during the TA demo.
    xhost +local:docker >/dev/null 2>&1 || true
    ARGS=("${TTY_ARGS[@]}" --rm --gpus all --ipc=host
          -e DISPLAY="${DISPLAY:-}" -e MUJOCO_GL=egl
          -v /tmp/.X11-unix:/tmp/.X11-unix:rw
          -v "${HOME}/.cache/huggingface:/root/.cache/huggingface:rw"
          -v "${HOME}/.cache/libero:/root/.cache/libero:rw"
          -v "${HERE}/outputs:/workspace/outputs:rw"
          -v "${HERE}/eval_logs:/workspace/eval_logs:rw")
    if [[ "$1" == "eval" ]]; then
      CKPT="${2:?usage: ./run.sh eval <checkpoint-dir>}"
      ${DOCKER} run "${ARGS[@]}" "${IMAGE}" \
        python /workspace/scripts/eval_liveview.py \
          --policy-path "${CKPT}" \
          --output-dir /workspace/eval_logs/docker_run
    else
      ${DOCKER} run "${ARGS[@]}" "${IMAGE}" /bin/bash
    fi
    ;;
  *) echo "usage: $0 {build|shell|eval <ckpt>}"; exit 1 ;;
esac
