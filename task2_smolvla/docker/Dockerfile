# IMS2026 HW0 - Task 2: SmolVLA x LIBERO reproduction
#
# One-command reproduction environment. CUDA 12.8 base is deliberate: the
# grading machine may run a 570.x driver, which cannot load CUDA 13 binaries.
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    # LIBERO/robosuite render through EGL, with no X server in the container
    MUJOCO_GL=egl \
    PYOPENGL_PLATFORM=egl \
    TOKENIZERS_PARALLELISM=false

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv python3.12-dev python3-pip \
        git curl ca-certificates \
        # FFmpeg 7 shared libs for torchcodec's video decoding
        ffmpeg libavcodec-dev libavformat-dev libavutil-dev \
        libswscale-dev libswresample-dev libavfilter-dev libavdevice-dev \
        # EGL / OpenGL for headless MuJoCo rendering
        libegl1 libgl1 libglew-dev libosmesa6-dev libglfw3 \
        libx11-6 libxrandr2 libxinerama1 libxcursor1 libxi6 \
    && rm -rf /var/lib/apt/lists/*

RUN python3.12 -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

WORKDIR /workspace

# LeRobot, pinned to the commit this reproduction was run against.
ARG LEROBOT_SHA=71a11efe77f55e61f3ab2ce45b40da8cf626afa9
RUN git clone https://github.com/huggingface/lerobot.git \
    && cd lerobot && git checkout ${LEROBOT_SHA}

# Install torch from the cu128 index BEFORE lerobot, so the extras resolve
# against it instead of pulling the default cu130 wheels.
RUN pip install --no-cache-dir torch==2.11.0 torchvision==0.26.0 \
        --index-url https://download.pytorch.org/whl/cu128 \
    && pip install --no-cache-dir -e "./lerobot[smolvla,libero]" \
    && pip install --no-cache-dir "numpy>=2.0.0,<2.3.0" \
    # MuJoCo 3.4.0's box-box collision fix stops LIBERO's stored initial state
    # for libero_spatial task 5 from settling the bowl onto the ramekin, which
    # costs ~6 pp on that suite. Nothing upstream caps the version.
    && pip install --no-cache-dir "mujoco<3.4.0" \
    && pip install --no-cache-dir opencv-python-headless imageio imageio-ffmpeg

# LIBERO prompts for a dataset path on first import; pre-seed its config so
# the container is non-interactive.
RUN mkdir -p /root/.libero && printf '%s\n' \
    'assets: /opt/venv/lib/python3.12/site-packages/libero/libero/./assets' \
    'bddl_files: /opt/venv/lib/python3.12/site-packages/libero/libero/./bddl_files' \
    'benchmark_root: /opt/venv/lib/python3.12/site-packages/libero/libero' \
    'datasets: /opt/venv/lib/python3.12/site-packages/libero/libero/../datasets' \
    'init_states: /opt/venv/lib/python3.12/site-packages/libero/libero/./init_files' \
    > /root/.libero/config.yaml

COPY scripts/ /workspace/scripts/

# Datasets, checkpoints and videos are mounted at run time, never baked in.
VOLUME ["/workspace/outputs", "/workspace/eval_logs", "/root/.cache/huggingface"]

CMD ["/bin/bash"]
