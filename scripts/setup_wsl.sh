#!/usr/bin/env bash
# One-time system setup for Ubuntu 22.04 (WSL2). Needs sudo.
set -euo pipefail

sudo apt-get update
sudo apt-get install -y \
    build-essential g++ clang cmake ninja-build gdb \
    python3-venv python3-pip \
    pkg-config curl wget unzip

echo "System packages installed. Next: scripts/fetch_onnxruntime.sh and scripts/setup_python.sh"
