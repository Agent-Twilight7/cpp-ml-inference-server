# Source this (". scripts/env_cuda.sh") before running binaries with --device cuda.
# ORT dlopen()s its CUDA provider at runtime, and that library's own
# dependencies (cuDNN, cuBLAS, ...) are found through LD_LIBRARY_PATH.
VENV="${VENV:-$HOME/.venvs/cpp_ml_inference_server}"
_nv_libs="$(ls -d "$VENV"/lib/python3*/site-packages/nvidia/*/lib 2>/dev/null | tr '\n' ':')"
if [[ -z "$_nv_libs" ]]; then
    echo "env_cuda.sh: no CUDA wheels found in $VENV (run scripts/setup_cuda.sh)" >&2
else
    export LD_LIBRARY_PATH="${_nv_libs}${LD_LIBRARY_PATH:-}"
fi
unset _nv_libs
