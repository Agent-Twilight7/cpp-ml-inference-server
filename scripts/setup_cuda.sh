#!/usr/bin/env bash
# Installs the CUDA 12 runtime libraries ONNX Runtime's CUDA provider needs
# (cuDNN 9, cuBLAS, cuFFT, cuRAND, cudart) as pip wheels into the project venv.
# No system CUDA toolkit or sudo required; the Windows NVIDIA driver is shared
# into WSL2 automatically.
set -euo pipefail

VENV="${VENV:-$HOME/.venvs/cpp_ml_inference_server}"
if [[ ! -x "$VENV/bin/pip" ]]; then
    echo "venv not found at $VENV; run scripts/setup_python.sh first" >&2
    exit 1
fi

"$VENV/bin/pip" install \
    "nvidia-cudnn-cu12>=9,<10" \
    nvidia-cublas-cu12 \
    nvidia-cuda-runtime-cu12 \
    nvidia-cufft-cu12 \
    nvidia-curand-cu12

echo "CUDA libraries installed. Before running GPU binaries: source scripts/env_cuda.sh"
