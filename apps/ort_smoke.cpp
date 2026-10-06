// Phase 1 smoke test: load the ONNX model through OnnxBackend, run the
// reference batch exported by models/export_resnet.py, compare against
// PyTorch's output, then time a few batch sizes to show why batching matters.
//
// Usage: ort_smoke [models_dir] [intra_op_threads]

#include "infer/runtime/onnx_backend.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <string>
#include <vector>

namespace {

std::vector<float> read_floats(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) {
        std::fprintf(stderr, "cannot open %s (run models/export_resnet.py first)\n", path.c_str());
        std::exit(1);
    }
    const auto bytes = static_cast<std::size_t>(f.tellg());
    std::vector<float> data(bytes / sizeof(float));
    f.seekg(0);
    f.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(data.size() * sizeof(float)));
    return data;
}

std::size_t argmax(const float* v, std::size_t n) {
    return static_cast<std::size_t>(std::max_element(v, v + n) - v);
}

}  // namespace

int run_smoke(int argc, char** argv) {
    const std::string dir = argc > 1 ? argv[1] : "models";
    const int threads = argc > 2 ? std::atoi(argv[2]) : 1;

    infer::OnnxBackend backend({.model_path = dir + "/resnet18.onnx", .intra_op_threads = threads});
    const std::size_t in_n = backend.input_elems_per_sample();
    const std::size_t out_n = backend.output_elems_per_sample();
    std::printf("model loaded: %zu input floats/sample, %zu output floats/sample\n", in_n, out_n);

    // ---- Correctness against PyTorch ----
    const std::vector<float> ref_in = read_floats(dir + "/ref_input.bin");
    const std::vector<float> ref_out = read_floats(dir + "/ref_output.bin");
    const std::size_t batch = ref_in.size() / in_n;
    if (batch == 0 || ref_out.size() != batch * out_n) {
        std::fprintf(stderr, "reference files have unexpected sizes\n");
        return 1;
    }

    std::vector<float> out(batch * out_n);
    backend.run(ref_in, out, batch);

    float max_diff = 0.f;
    std::size_t top1_match = 0;
    for (std::size_t i = 0; i < out.size(); ++i) max_diff = std::max(max_diff, std::fabs(out[i] - ref_out[i]));
    for (std::size_t b = 0; b < batch; ++b) {
        top1_match += argmax(&out[b * out_n], out_n) == argmax(&ref_out[b * out_n], out_n);
    }
    std::printf("batch=%zu  max abs diff vs PyTorch = %.3e  top-1 agreement = %zu/%zu\n", batch, max_diff, top1_match,
                batch);
    const bool ok = max_diff < 1e-3f && top1_match == batch;
    std::printf("correctness: %s\n", ok ? "PASS" : "FAIL");

    // ---- Throughput by batch size ----
    std::printf("\n%8s %12s %14s %16s\n", "batch", "ms/batch", "ms/sample", "samples/sec");
    for (std::size_t b : {1u, 2u, 4u, 8u, 16u, 32u}) {
        std::vector<float> in(b * in_n, 0.5f);
        std::vector<float> o(b * out_n);
        backend.run(in, o, b);  // warm-up

        constexpr int kIters = 10;
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < kIters; ++i) backend.run(in, o, b);
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / kIters;
        std::printf("%8zu %12.2f %14.2f %16.1f\n", b, ms, ms / static_cast<double>(b), 1000.0 * static_cast<double>(b) / ms);
    }
    return ok ? 0 : 1;
}

int main(int argc, char** argv) {
    try {
        return run_smoke(argc, argv);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
