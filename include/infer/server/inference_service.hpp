#pragma once

#include "infer/batching/dynamic_batcher.hpp"
#include "infer/memory/block_pool.hpp"
#include "infer/metrics/histogram.hpp"
#include "infer/net/http_server.hpp"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <mutex>
#include <string>
#include <thread>

namespace infer {

// Request-level metrics owned by the service (the batcher records its own
// into BatcherMetrics).
struct ServiceMetrics {
    metrics::LatencyHistogram request_latency_us;  // successful /v1/infer: parsed -> response handed to reactor
    metrics::Counter ok;           // 200
    metrics::Counter bad_request;  // 400
    metrics::Counter rejected;     // 503: queue or request pool full
    metrics::Counter timeouts;     // 504
    metrics::Counter errors;       // 500
};

struct InferenceServiceOptions {
    std::size_t top_k = 5;
    std::chrono::seconds default_stats_window{10};
    std::chrono::seconds max_stats_window{60};
    std::chrono::milliseconds tick{1000};  // how often the sliding window advances
};

// The HTTP face of the server. Registers:
//   GET  /health    liveness
//   POST /v1/infer  one float32 sample -> top-k JSON (via the batcher)
//   GET  /stats     JSON: windowed p50/p90/p99/p99.9, throughput, batching, gauges
//                   (?window=N seconds, default 10, max 60)
//   GET  /metrics   Prometheus text exposition
//
// Windowed statistics: a ticker thread snapshots every histogram and counter
// once per tick into a ring covering max_stats_window. A window's numbers are
// (live snapshot) - (snapshot from N seconds ago), so request threads only ever
// do relaxed atomic increments; nothing on the hot path takes a lock.
class InferenceService {
public:
    // All references must outlive the service. `batcher_metrics` is the
    // object passed in the batcher's options (may be null).
    InferenceService(DynamicBatcher& batcher, BlockPool& request_pool, BatcherMetrics* batcher_metrics,
                     InferenceServiceOptions opts = {});
    ~InferenceService();

    InferenceService(const InferenceService&) = delete;
    InferenceService& operator=(const InferenceService&) = delete;

    // Registers the routes and remembers `server` for connection gauges.
    void install(net::HttpServer& server);
    // Starts / stops the window ticker thread.
    void start();
    void stop();
    // While draining, /health answers 503 so load balancers stop routing here.
    void set_draining(bool draining) { draining_.store(draining, std::memory_order_relaxed); }

    const ServiceMetrics& metrics() const { return metrics_; }
    std::string render_stats_json(std::chrono::seconds window) const;
    std::string render_prometheus() const;

private:
    struct Sample {
        Clock::time_point at;
        metrics::LatencyHistogram::Snapshot latency, queue_wait, compute;
        metrics::SmallHistogram::Snapshot batch_size;
        std::uint64_t ok = 0, bad_request = 0, rejected = 0, timeouts = 0, errors = 0;
        std::uint64_t batches = 0, padded_slots = 0;
    };

    void handle_infer(net::HttpRequest&& req, net::Responder r);
    Sample take_sample() const;
    // The newest ring sample at least `window` old (or the oldest one).
    Sample baseline(std::chrono::seconds window) const;
    void ticker_loop();

    DynamicBatcher& batcher_;
    BlockPool& pool_;
    BatcherMetrics* batcher_metrics_;
    InferenceServiceOptions opts_;
    net::HttpServer* server_ = nullptr;
    const Clock::time_point started_ = Clock::now();

    ServiceMetrics metrics_;
    std::atomic<std::int64_t> inflight_{0};
    std::atomic<bool> draining_{false};

    mutable std::mutex ring_mu_;
    std::deque<Sample> ring_;  // oldest first

    std::mutex tick_mu_;
    std::condition_variable tick_cv_;
    bool stopping_ = false;
    std::thread ticker_;
};

}  // namespace infer
