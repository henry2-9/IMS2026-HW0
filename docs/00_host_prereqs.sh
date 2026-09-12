#!/usr/bin/env bash
# IMS2026 HW0 - host prerequisites (Docker + NVIDIA Container Toolkit)
# RUN THIS YOURSELF WITH SUDO. In Claude Code you can prefix with `!`.
# Host: Ubuntu 20.04 (focal), driver 570.133.07, RTX 4090 24GB
set -euo pipefail

echo "==> [1/4] Docker Engine (official repo, focal)"
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
  | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
https://download.docker.com/linux/ubuntu focal stable" \
  | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io \
  docker-buildx-plugin docker-compose-plugin

echo "==> [2/4] NVIDIA Container Toolkit"
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list > /dev/null
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

echo "==> [3/4] Run docker without sudo"
# When this script is invoked as `sudo bash ...`, $USER is root -- which would
# add root to the docker group and leave the real user without access.
# SUDO_USER holds the invoking account in that case.
TARGET_USER="${SUDO_USER:-$USER}"
sudo groupadd -f docker
sudo usermod -aG docker "$TARGET_USER"
echo "    added '$TARGET_USER' to the docker group"
echo "    NOTE: log out/in (or run 'newgrp docker') for this to take effect."

echo "==> [4/4] Verify GPU passthrough"
sudo docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi

echo "==> DONE. Next: docs/01_isaacsim_container.sh"
