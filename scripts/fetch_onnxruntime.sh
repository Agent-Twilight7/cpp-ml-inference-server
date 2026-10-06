#!/usr/bin/env bash
# Downloads the prebuilt ONNX Runtime (CPU, Linux x64) into third_party/onnxruntime.
set -euo pipefail

ORT_VERSION="${ORT_VERSION:-1.30.0}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$ROOT/third_party/onnxruntime"
NAME="onnxruntime-linux-x64-${ORT_VERSION}"

if [[ -f "$DEST/include/onnxruntime_cxx_api.h" ]]; then
    echo "ONNX Runtime already present at $DEST"
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
echo "ONNX Runtime ${ORT_VERSION} installed to $DEST"
