// Inference server entry point: parses flags and wires the pieces together.
//
//   HttpServer (epoll reactors) -> InferenceService -> DynamicBatcher -> OnnxBackend
//
// Usage:
//   inference_server --model models/resnet18.onnx [--device cpu|cuda]
//                    [--port 8080] [--reactors 2] [--workers 2] [--ort-threads 1]
//                    [--max-batch 16] [--max-delay-us 2000]
//                    [--queue-size 1024] [--queue-timeout-ms 0]
//                    [--buckets auto] [--pinned 1]
//
// --max-batch 1 turns batching off (same pipeline, one request per run).
// Endpoints: /health, /v1/infer, /stats, /metrics (see InferenceService).

#include "infer/batching/dynamic_batcher.hpp"
#include "infer/memory/block_pool.hpp"
#include "infer/net/http_server.hpp"
#include "infer/runtime/onnx_backend.hpp"
#include "infer/server/inference_service.hpp"

#include <algorithm>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace {

struct Config {
    std::string model = "models/resnet18.onnx";
    infer::Device device = infer::Device::kCpu;
    std::uint16_t port = 8080;
    int reactors = 2;
    int workers = 2;
    int ort_threads = 1;
    std::size_t max_batch = 16;
    long max_delay_us = 2000;
    std::size_t queue_size = 1024;
    long queue_timeout_ms = 0;
    std::string buckets = "auto";  // "auto", "none", or e.g. "1,2,4,8,16"
    bool pinned = true;            // CUDA pinned batch buffers
};

[[noreturn]] void usage(const char* argv0) {
    std::fprintf(stderr,
                 "usage: %s [--model PATH] [--device cpu|cuda] [--port N] [--reactors N] [--workers N]\n"
                 "          [--ort-threads N] [--max-batch N] [--max-delay-us N] [--queue-size N]\n"
                 "          [--queue-timeout-ms N] [--buckets auto|none|N,N,...] [--pinned 0|1]\n",
                 argv0);
    std::exit(2);
}

Config parse_args(int argc, char** argv) {
    Config cfg;
    for (int i = 1; i < argc; ++i) {
        const std::string flag = argv[i];
        if (flag == "-h" || flag == "--help") usage(argv[0]);
        if (i + 1 >= argc) usage(argv[0]);
        const std::string value = argv[++i];
        if (flag == "--model") {
            cfg.model = value;
        } else if (flag == "--device") {
            cfg.device = infer::parse_device(value);
        } else if (flag == "--port") {
            cfg.port = static_cast<std::uint16_t>(std::stoi(value));
        } else if (flag == "--reactors") {
            cfg.reactors = std::stoi(value);
        } else if (flag == "--workers") {
            cfg.workers = std::stoi(value);
        } else if (flag == "--ort-threads") {
            cfg.ort_threads = std::stoi(value);
        } else if (flag == "--max-batch") {
            cfg.max_batch = std::stoul(value);
        } else if (flag == "--max-delay-us") {
            cfg.max_delay_us = std::stol(value);
        } else if (flag == "--queue-size") {
            cfg.queue_size = std::stoul(value);
        } else if (flag == "--queue-timeout-ms") {
            cfg.queue_timeout_ms = std::stol(value);
        } else if (flag == "--buckets") {
            cfg.buckets = value;
        } else if (flag == "--pinned") {
            cfg.pinned = value != "0";
        } else {
            usage(argv[0]);
        }
    }
    return cfg;
}

// Resolves --buckets. "auto" means power-of-two buckets on CUDA (where shape
// changes are expensive) and none on CPU (where padding costs real compute).
std::vector<std::size_t> resolve_buckets(const Config& cfg) {
    if (cfg.buckets == "none") return {};
    if (cfg.buckets == "auto") {
        return cfg.device == infer::Device::kCuda ? infer::power_of_two_buckets(cfg.max_batch)
                                                  : std::vector<std::size_t>{};
    }
    std::vector<std::size_t> out;
    std::size_t pos = 0;
    while (pos < cfg.buckets.size()) {
        const std::size_t comma = cfg.buckets.find(',', pos);
        out.push_back(std::stoul(cfg.buckets.substr(pos, comma - pos)));
        if (comma == std::string::npos) break;
        pos = comma + 1;
    }
    if (out.empty() || *std::max_element(out.begin(), out.end()) < cfg.max_batch) {
        throw std::invalid_argument("--buckets must include a size >= --max-batch");
    }
    return out;
}

std::string join(const std::vector<std::size_t>& v) {
    std::string s;
    for (std::size_t x : v) s += (s.empty() ? "" : ",") + std::to_string(x);
    return s.empty() ? "none" : s;
}

void install_terminate_handler() {
    // If anything ever escapes a thread, say what it was before aborting.
    std::set_terminate([] {
        if (auto e = std::current_exception()) {
            try {
                std::rethrow_exception(e);
            } catch (const std::exception& ex) {
                std::fprintf(stderr, "terminate: uncaught exception: %s\n", ex.what());
            } catch (...) {
                std::fprintf(stderr, "terminate: uncaught non-standard exception\n");
            }
        } else {
            std::fprintf(stderr, "terminate called without an active exception\n");
        }
        std::abort();
    });
}

}  // namespace

int main(int argc, char** argv) {
    install_terminate_handler();
    const Config cfg = parse_args(argc, argv);

    // Block termination signals in every thread (threads inherit the mask);
    // the main thread then waits for them synchronously with sigwait().
    sigset_t signals;
    sigemptyset(&signals);
    sigaddset(&signals, SIGINT);
    sigaddset(&signals, SIGTERM);
    pthread_sigmask(SIG_BLOCK, &signals, nullptr);

    try {
        const std::vector<std::size_t> buckets = resolve_buckets(cfg);
        std::fprintf(stderr, "loading %s on %s (%d workers, batch buckets: %s)...\n", cfg.model.c_str(),
                     infer::to_string(cfg.device), cfg.workers, join(buckets).c_str());
        const infer::OnnxBackendOptions backend_opts{.model_path = cfg.model,
                                                     .device = cfg.device,
                                                     .intra_op_threads = cfg.ort_threads,
                                                     .batch_buckets = buckets,
                                                     .pinned_host_memory = cfg.pinned};

        // Declaration order is destruction order in reverse: metrics and the
        // request pool must outlive the batcher, whose queued requests hold
        // pool blocks and whose workers record metrics.
        infer::BatcherMetrics batcher_metrics(cfg.max_batch);
        std::optional<infer::BlockPool> request_pool;  // sized once the input shape is known
        infer::DynamicBatcher batcher(
            infer::BatcherOptions{
                .max_batch_size = cfg.max_batch,
                .max_queue_delay = std::chrono::microseconds(cfg.max_delay_us),
                .max_queue_size = cfg.queue_size,
                .queue_timeout = std::chrono::milliseconds(cfg.queue_timeout_ms),
                .num_workers = cfg.workers,
                .metrics = &batcher_metrics,
            },
            [&] { return std::make_unique<infer::OnnxBackend>(backend_opts); });
        batcher.start();  // includes warm-up

        // Enough blocks for a full queue plus the batches being gathered. An
        // empty pool means overload (503), so this also caps request memory.
        // Untouched blocks cost no physical memory.
        request_pool.emplace(batcher.input_elems(),
                             cfg.queue_size + static_cast<std::size_t>(cfg.workers) * cfg.max_batch);

        infer::InferenceService service(batcher, *request_pool, &batcher_metrics);
        infer::net::HttpServer server({.port = cfg.port, .num_reactors = cfg.reactors});
        service.install(server);
        service.start();
        server.start();
        std::fprintf(stderr, "listening on :%u | reactors=%d workers=%d max_batch=%zu max_delay=%ldus queue=%zu\n",
                     server.port(), cfg.reactors, cfg.workers, cfg.max_batch, cfg.max_delay_us, cfg.queue_size);

        int sig = 0;
        sigwait(&signals, &sig);
        std::fprintf(stderr, "received %s, shutting down\n", sig == SIGINT ? "SIGINT" : "SIGTERM");
        // Batcher first: its callbacks send through the reactors, which must
        // still be alive. New requests arriving meanwhile get 503.
        batcher.stop();
        server.stop();
        service.stop();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "fatal: %s\n", e.what());
        return 1;
    }
    return 0;
}
