# C++ ML Inference Server

A high-throughput model-serving server in C++20: a custom epoll HTTP/1.1 stack,
**dynamic batching** (the technique behind Triton and vLLM), preallocated memory
pools, ONNX Runtime execution, and a metrics endpoint with p50/p99 latency,
throughput, queue depth and batch-size distribution.

> Status: **Phase 1 done — C++ inference matches PyTorch.** See [Roadmap](#roadmap).

## Architecture

```
 clients ──HTTP──▶ epoll reactors ──▶ bounded request queue ──▶ dynamic batcher
                        ▲                                            │
                        │ eventfd                                    ▼
                  completion queue ◀── scatter outputs ◀── executor pool (ONNX Runtime)
```

## Build (Ubuntu 22.04 / WSL2)

```bash
bash scripts/setup_wsl.sh          # one-time: compilers, cmake, python venv support (sudo)
bash scripts/fetch_onnxruntime.sh  # prebuilt ONNX Runtime into third_party/
bash scripts/setup_python.sh       # venv with PyTorch (CPU) for model export
source ~/.venvs/cpp_ml_inference_server/bin/activate
python models/export_resnet.py     # resnet18.onnx + reference batch

cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
./build/ort_smoke models
```

Sanitizer builds: add `-DINFER_ENABLE_ASAN=ON` or `-DINFER_ENABLE_TSAN=ON`.

## Roadmap

- [x] Phase 0 — toolchain, ONNX Runtime, ResNet-18 export
- [x] Phase 1 — C++ inference matches PyTorch
- [ ] Phase 2 — epoll HTTP server, unbatched baseline
- [ ] Phase 3 — request queue + dynamic batcher + eventfd completion path
- [ ] Phase 4 — memory pools, multiple executor workers
- [ ] Phase 5 — `/metrics` (Prometheus) and `/stats` (JSON)
- [ ] Phase 6 — load generator and benchmarks
- [ ] Phase 7 — Docker, CI, graceful shutdown
