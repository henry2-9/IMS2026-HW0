#!/usr/bin/env bash
# IMS2026 HW0 Task 1 - Isaac Sim 5.0.0 via NVIDIA's container
# Host is Ubuntu 20.04, which Isaac Sim 5.0 does NOT support natively
# (it requires 22.04/24.04). The container ships its own 22.04 userspace,
# so the host distro stops mattering; only the driver + toolkit do.
#
# Prereq: docs/00_host_prereqs.sh must have been run.
set -euo pipefail

IMAGE="nvcr.io/nvidia/isaac-sim:5.0.0"
CACHE="${HOME}/docker/isaac-sim"

echo "==> creating persistent cache dirs (so re-runs don't re-download assets)"
mkdir -p "${CACHE}"/{cache/kit,cache/ov,cache/pip,cache/glcache,cache/computecache,logs,data,documents}

echo "==> pulling ${IMAGE} (~20 GB, needs an NGC login)"
echo "    If this 401s: get an NGC API key at https://ngc.nvidia.com and run"
echo "      docker login nvcr.io -u '\$oauthtoken' -p <API_KEY>"
docker pull "${IMAGE}"

echo "==> smoke test: headless Isaac Sim, should print a version banner and exit"
docker run --name isaac-sim-smoke --rm --gpus all \
  -e "ACCEPT_EULA=Y" -e "PRIVACY_CONSENT=Y" \
  -v "${CACHE}/cache/kit:/isaac-sim/kit/cache:rw" \
  -v "${CACHE}/cache/ov:/root/.cache/ov:rw" \
  -v "${CACHE}/cache/pip:/root/.cache/pip:rw" \
  -v "${CACHE}/cache/glcache:/root/.cache/nvidia/GLCache:rw" \
  -v "${CACHE}/cache/computecache:/root/.nv/ComputeCache:rw" \
  -v "${CACHE}/logs:/root/.nvidia-omniverse/logs:rw" \
  -v "${CACHE}/data:/root/.local/share/ov/data:rw" \
  -v "${CACHE}/documents:/root/Documents:rw" \
  "${IMAGE}" \
  ./python.sh -c "import isaacsim; print('Isaac Sim import OK')"

echo "==> DONE"
