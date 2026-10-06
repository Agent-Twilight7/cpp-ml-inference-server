#!/usr/bin/env bash
# Creates .venv with the CPU build of PyTorch + ONNX tooling (used only for model export / reference checks).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${VENV:-$HOME/.venvs/cpp_ml_inference_server}"

python3 -m venv "$VENV"
"$VENV/bin/pip" install --upgrade pip
"$VENV/bin/pip" install --index-url https://download.pytorch.org/whl/cpu torch torchvision
"$VENV/bin/pip" install -r "$ROOT/models/requirements.txt"

echo "Python env ready: source $VENV/bin/activate"
