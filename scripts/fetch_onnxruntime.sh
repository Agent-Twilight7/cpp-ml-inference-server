#!/usr/bin/env bash
# Downloads prebuilt ONNX Runtime (Linux x64) into third_party/.
#   ORT_FLAVOR=cpu  (default) -> third_party/onnxruntime
#   ORT_FLAVOR=gpu            -> third_party/onnxruntime-gpu  (CUDA 12 build; also runs on CPU)
set -euo pipefail

ORT_VERSION="${ORT_VERSION:-1.30.0}"
ORT_FLAVOR="${ORT_FLAVOR:-cpu}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

case "$ORT_FLAVOR" in
    cpu) NAME="onnxruntime-linux-x64-${ORT_VERSION}";             DEST="$ROOT/third_party/onnxruntime" ;;
    gpu) NAME="onnxruntime-linux-x64-gpu_cuda12-${ORT_VERSION}";  DEST="$ROOT/third_party/onnxruntime-gpu" ;;
    *)   echo "ORT_FLAVOR must be cpu or gpu" >&2; exit 1 ;;
esac

if [[ -f "$DEST/include/onnxruntime_cxx_api.h" ]]; then
    echo "ONNX Runtime ($ORT_FLAVOR) already present at $DEST"
    exit 0
fi

mkdir -p "$ROOT/third_party"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

curl -fsSL --retry 3 -o "$TMP/ort.tgz" \
    "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/${NAME}.tgz"
tar -xzf "$TMP/ort.tgz" -C "$TMP"
rm -rf "$DEST"
mv "$TMP/$NAME" "$DEST"
echo "ONNX Runtime ${ORT_VERSION} ($ORT_FLAVOR) installed to $DEST"
