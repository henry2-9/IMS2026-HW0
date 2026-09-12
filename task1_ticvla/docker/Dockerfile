# IMS2026 HW0 - Task 1: TIC-VLA on the DynaNav benchmark in Isaac Sim 5.0.0
#
# Derived from NVIDIA's Isaac Sim container. This is what makes the assignment
# runnable at all on this machine: the host is Ubuntu 20.04, which Isaac Sim 5.0
# does not support (it requires 22.04/24.04). The container brings its own
# 22.04 userspace, so only the host driver and nvidia-container-toolkit matter.
#
# Requires an NGC login for the base image:
#   docker login nvcr.io -u '$oauthtoken' -p <NGC_API_KEY>
FROM nvcr.io/nvidia/isaac-sim:5.0.0

ENV DEBIAN_FRONTEND=noninteractive \
    ACCEPT_EULA=Y \
    PRIVACY_CONSENT=Y \
    ISAAC_SIM_ROOT=/isaac-sim \
    ISAAC_SIM_PYTHON=/isaac-sim/python.sh

RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY TIC-VLA /workspace/TIC-VLA

# DynaNav dependencies go into Isaac Sim's own Python, NOT a separate venv.
# requirements-test.txt deliberately pins numpy<2: Isaac Sim 5.0 ships binary
# modules built against the NumPy 1.x ABI, and requirements-train.txt would
# replace Isaac's CUDA/PyTorch stack.
RUN cd /workspace/TIC-VLA \
    && ${ISAAC_SIM_PYTHON} -m pip install --no-cache-dir -e . \
    && ${ISAAC_SIM_PYTHON} -m pip install --no-cache-dir -r requirements-test.txt

# Model weights and outputs are mounted at run time, never baked into the image.
VOLUME ["/workspace/checkpoints", "/workspace/outputs", "/root/.cache/ov"]

WORKDIR /workspace/TIC-VLA

# The base image pins ENTRYPOINT to ["/bin/sh","-c","/isaac-sim/runheadless.sh"].
# Because that is the shell form, any CMD is silently ignored and the container
# always launches the stock streaming app instead of the command you asked for.
# Reset it so `docker run <image> <cmd>` actually runs <cmd>.
ENTRYPOINT []
CMD ["/bin/bash"]
