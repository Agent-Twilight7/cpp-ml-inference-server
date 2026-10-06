// Inference server entry point.
//
// Phase 2 (unbatched baseline): each request runs the model by itself, inline
// on the reactor thread that parsed it. Phase 3 replaces this with the dynamic
// batcher.
//
// Usage:
//   inference_server --model models/resnet18.onnx [--device cpu|cuda]
//                    [--port 8080] [--reactors 4] [--ort-threads 1]
//
// API:
//   GET  /health     -> "ok"
//   POST /v1/infer   body: one sample as raw little-endian float32
//                    (3x224x224 for ResNet-18) -> JSON top-5 classes

#include "infer/net/http_server.hpp"
#include "infer/runtime/onnx_backend.hpp"

#include <algorithm>
#include <cmath>
#include <condition_variable>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <numeric>
#include <string>
#include <vector>

namespace {

struct Config {
    std::string model = "models/resnet18.onnx";
    infer::Device device = infer::Device::kCpu;
    std::uint16_t port = 8080;
    int reactors = 4;
    int ort_threads = 1;
};

[[noreturn]] void usage(const char* argv0) {
    std::fprintf(stderr,
                 "usage: %s [--model PATH] [--device cpu|cuda] [--port N] [--reactors N] [--ort-threads N]\n", argv0);
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
        } else if (flag == "--ort-threads") {
            cfg.ort_threads = std::stoi(value);
        } else {
            usage(argv[0]);
        }
    }
    return cfg;
}

// A fixed set of model sessions handed out one at a time. ORT sessions are not
// meant to be shared across concurrent Run() calls with our caller-owned
// output buffers, so each in-flight request checks one out.
class BackendPool {
public:
    BackendPool(const infer::OnnxBackendOptions& opts, int count) {
        for (int i = 0; i < count; ++i) {
            backends_.push_back(std::make_unique<infer::OnnxBackend>(opts));
            free_.push_back(backends_.back().get());
        }
    }

    class Lease {
    public:
        Lease(BackendPool& pool, infer::OnnxBackend* b) : pool_(pool), backend_(b) {}
        ~Lease() { pool_.release(backend_); }
        Lease(const Lease&) = delete;
        Lease& operator=(const Lease&) = delete;
        infer::OnnxBackend* operator->() const { return backend_; }

    private:
        BackendPool& pool_;
        infer::OnnxBackend* backend_;
    };

    Lease acquire() {
        std::unique_lock lock(mu_);
        cv_.wait(lock, [&] { return !free_.empty(); });
        infer::OnnxBackend* b = free_.back();
        free_.pop_back();
        return Lease(*this, b);
    }

    infer::OnnxBackend& any() { return *backends_.front(); }

    // Only safe while no requests are in flight (startup).
    template <typename Fn>
    void for_each(Fn&& fn) {
        for (auto& b : backends_) fn(*b);
    }

private:
    void release(infer::OnnxBackend* b) {
        {
            std::lock_guard lock(mu_);
            free_.push_back(b);
        }
        cv_.notify_one();
    }

    std::vector<std::unique_ptr<infer::OnnxBackend>> backends_;
    std::vector<infer::OnnxBackend*> free_;
    std::mutex mu_;
    std::condition_variable cv_;
};

// Formats softmax top-k of one logits row as JSON.
std::string topk_json(const float* logits, std::size_t n, std::size_t k) {
    std::vector<std::size_t> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    k = std::min(k, n);
    std::partial_sort(idx.begin(), idx.begin() + static_cast<std::ptrdiff_t>(k), idx.end(),
                      [&](std::size_t a, std::size_t b) { return logits[a] > logits[b]; });

    const float max_logit = logits[idx[0]];
    double denom = 0.0;
    for (std::size_t i = 0; i < n; ++i) denom += std::exp(static_cast<double>(logits[i] - max_logit));

    std::string out = "{\"top_k\":[";
    char buf[96];
    for (std::size_t j = 0; j < k; ++j) {
        const double p = std::exp(static_cast<double>(logits[idx[j]] - max_logit)) / denom;
        std::snprintf(buf, sizeof(buf), "%s{\"class\":%zu,\"prob\":%.6f}", j ? "," : "", idx[j], p);
        out += buf;
    }
    out += "]}\n";
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    const Config cfg = parse_args(argc, argv);

    // Block termination signals in every thread (threads inherit the mask);
    // the main thread then waits for them synchronously with sigwait().
    sigset_t signals;
    sigemptyset(&signals);
    sigaddset(&signals, SIGINT);
    sigaddset(&signals, SIGTERM);
    pthread_sigmask(SIG_BLOCK, &signals, nullptr);

    try {
        std::fprintf(stderr, "loading %s on %s (%d sessions)...\n", cfg.model.c_str(), infer::to_string(cfg.device),
                     cfg.reactors);
        BackendPool pool({.model_path = cfg.model, .device = cfg.device, .intra_op_threads = cfg.ort_threads},
                         cfg.reactors);
        const std::size_t in_n = pool.any().input_elems_per_sample();
        const std::size_t out_n = pool.any().output_elems_per_sample();

        // Warm up every session so the first real request doesn't pay for
        // lazy initialization (cuDNN kernel selection, memory arenas).
        {
            std::vector<float> in(in_n, 0.f), out(out_n);
            pool.for_each([&](infer::OnnxBackend& b) { b.run(in, out, 1); });
        }

        infer::net::HttpServer server({.port = cfg.port, .num_reactors = cfg.reactors});

        server.route("GET", "/health", [](infer::net::HttpRequest&&, infer::net::Responder r) {
            r.send(infer::net::HttpResponse::text(200, "ok\n"));
        });

        server.route("POST", "/v1/infer", [&pool, in_n, out_n](infer::net::HttpRequest&& req, infer::net::Responder r) {
            const std::size_t want = in_n * sizeof(float);
            if (req.body.size() != want) {
                r.send(infer::net::HttpResponse::text(
                    400, "expected " + std::to_string(want) + " bytes (" + std::to_string(in_n) +
                             " float32), got " + std::to_string(req.body.size()) + "\n"));
                return;
            }
            // Copy out of the string so the floats are properly aligned.
            std::vector<float> input(in_n);
            std::memcpy(input.data(), req.body.data(), want);
            std::vector<float> logits(out_n);
            {
                auto backend = pool.acquire();
                backend->run(input, logits, 1);
            }
            r.send(infer::net::HttpResponse::json(200, topk_json(logits.data(), out_n, 5)));
        });

        server.start();
        std::fprintf(stderr, "listening on :%u with %d reactors (unbatched baseline)\n", server.port(), cfg.reactors);

        int sig = 0;
        sigwait(&signals, &sig);
        std::fprintf(stderr, "received %s, shutting down\n", sig == SIGINT ? "SIGINT" : "SIGTERM");
        server.stop();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "fatal: %s\n", e.what());
        return 1;
    }
    return 0;
}
