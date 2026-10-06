#include "infer/server/inference_service.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <vector>

namespace infer {
namespace {

using std::chrono::duration;
using std::chrono::duration_cast;
using std::chrono::microseconds;

std::uint64_t micros_since(Clock::time_point t0) {
    return static_cast<std::uint64_t>(duration_cast<microseconds>(Clock::now() - t0).count());
}

// Softmax top-k of one logits row, plus batching info, as JSON.
std::string result_json(std::span<const float> logits, std::size_t k, const InferResult& res) {
    const std::size_t n = logits.size();
    std::vector<std::size_t> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    k = std::min(k, n);
    std::partial_sort(idx.begin(), idx.begin() + static_cast<std::ptrdiff_t>(k), idx.end(),
                      [&](std::size_t a, std::size_t b) { return logits[a] > logits[b]; });

    const float max_logit = logits[idx[0]];
    double denom = 0.0;
    for (float v : logits) denom += std::exp(static_cast<double>(v - max_logit));

    std::string out = "{\"top_k\":[";
    char buf[128];
    for (std::size_t j = 0; j < k; ++j) {
        const double p = std::exp(static_cast<double>(logits[idx[j]] - max_logit)) / denom;
        std::snprintf(buf, sizeof(buf), "%s{\"class\":%zu,\"prob\":%.6f}", j ? "," : "", idx[j], p);
        out += buf;
    }
    std::snprintf(buf, sizeof(buf), "],\"batch_size\":%zu,\"queue_us\":%lld,\"compute_us\":%lld}\n", res.batch_size,
                  static_cast<long long>(res.queue_time.count()), static_cast<long long>(res.compute_time.count()));
    out += buf;
    return out;
}

// Small helpers for hand-written JSON / Prometheus text.
void appendf(std::string& out, const char* fmt, auto... args) {
    char buf[256];
    const int n = std::snprintf(buf, sizeof(buf), fmt, args...);
    if (n > 0) out.append(buf, std::min(static_cast<std::size_t>(n), sizeof(buf) - 1));
}

void append_percentiles_ms(std::string& out, const char* name, const metrics::LatencyHistogram::Snapshot& s) {
    appendf(out, "\"%s\":{\"count\":%llu,\"mean\":%.3f,\"p50\":%.3f,\"p90\":%.3f,\"p99\":%.3f,\"p999\":%.3f,\"max\":%.3f}",
            name, static_cast<unsigned long long>(s.count), s.mean() / 1000.0, s.percentile(0.50) / 1000.0,
            s.percentile(0.90) / 1000.0, s.percentile(0.99) / 1000.0, s.percentile(0.999) / 1000.0,
            s.max() / 1000.0);
}

void append_summary(std::string& out, const char* name, const char* help,
                    const metrics::LatencyHistogram::Snapshot& window,
                    const metrics::LatencyHistogram::Snapshot& total) {
    appendf(out, "# HELP %s %s\n# TYPE %s summary\n", name, help, name);
    for (double q : {0.5, 0.9, 0.99, 0.999}) {
        appendf(out, "%s{quantile=\"%g\"} %.6f\n", name, q, window.percentile(q) / 1e6);
    }
    appendf(out, "%s_sum %.6f\n%s_count %llu\n", name, static_cast<double>(total.sum) / 1e6, name,
            static_cast<unsigned long long>(total.count));
}

void append_metric(std::string& out, const char* name, const char* type, const char* help, double value) {
    appendf(out, "# HELP %s %s\n# TYPE %s %s\n%s %.17g\n", name, help, name, type, name, value);
}

}  // namespace

InferenceService::InferenceService(DynamicBatcher& batcher, BlockPool& request_pool, BatcherMetrics* batcher_metrics,
                                   InferenceServiceOptions opts)
    : batcher_(batcher), pool_(request_pool), batcher_metrics_(batcher_metrics), opts_(opts) {
    ring_.push_back(take_sample());  // baseline for windows longer than the uptime
}

InferenceService::~InferenceService() { stop(); }

void InferenceService::install(net::HttpServer& server) {
    server_ = &server;
    server.route("GET", "/health", [](net::HttpRequest&&, net::Responder r) {
        r.send(net::HttpResponse::text(200, "ok\n"));
    });
    server.route("POST", "/v1/infer", [this](net::HttpRequest&& req, net::Responder r) {
        handle_infer(std::move(req), r);
    });
    server.route("GET", "/stats", [this](net::HttpRequest&& req, net::Responder r) {
        long window = opts_.default_stats_window.count();
        const std::string_view target = req.target;
        if (auto pos = target.find("window="); pos != std::string_view::npos) {
            window = std::atol(std::string(target.substr(pos + 7)).c_str());
        }
        window = std::clamp(window, 1L, static_cast<long>(opts_.max_stats_window.count()));
        r.send(net::HttpResponse::json(200, render_stats_json(std::chrono::seconds(window))));
    });
    server.route("GET", "/metrics", [this](net::HttpRequest&&, net::Responder r) {
        net::HttpResponse resp{200, "text/plain; version=0.0.4; charset=utf-8", render_prometheus(), {}};
        r.send(std::move(resp));
    });
}

void InferenceService::start() {
    if (ticker_.joinable()) return;
    ticker_ = std::thread([this] { ticker_loop(); });
}

void InferenceService::stop() {
    {
        std::lock_guard lock(tick_mu_);
        stopping_ = true;
    }
    tick_cv_.notify_all();
    if (ticker_.joinable()) ticker_.join();
}

void InferenceService::handle_infer(net::HttpRequest&& req, net::Responder r) {
    const Clock::time_point t0 = Clock::now();
    const std::size_t in_n = batcher_.input_elems();
    const std::size_t want = in_n * sizeof(float);
    if (req.body.size() != want) {
        metrics_.bad_request.inc();
        r.send(net::HttpResponse::text(400, "expected " + std::to_string(want) + " bytes (" + std::to_string(in_n) +
                                                " float32), got " + std::to_string(req.body.size()) + "\n"));
        return;
    }

    // The one copy of the input: from the connection's receive buffer into a
    // preallocated, aligned pool block. No allocation.
    BlockPool::Block input = pool_.try_acquire();
    if (!input) {
        metrics_.rejected.inc();
        r.send(net::HttpResponse::text(503, "server overloaded, retry later\n"));
        return;
    }
    std::memcpy(input.data(), req.body.data(), want);

    inflight_.fetch_add(1, std::memory_order_relaxed);
    const bool accepted = batcher_.submit(std::move(input), [this, r, t0](const InferResult& res) {
        switch (res.status) {
            case InferStatus::kOk:
                r.send(net::HttpResponse::json(200, result_json(res.output, opts_.top_k, res)));
                metrics_.ok.inc();
                metrics_.request_latency_us.record(micros_since(t0));
                break;
            case InferStatus::kTimeout:
                metrics_.timeouts.inc();
                r.send(net::HttpResponse::text(504, "timed out in queue\n"));
                break;
            case InferStatus::kError:
                metrics_.errors.inc();
                r.send(net::HttpResponse::text(500, "inference failed: " + std::string(res.error) + "\n"));
                break;
        }
        inflight_.fetch_sub(1, std::memory_order_relaxed);
    });
    if (!accepted) {
        inflight_.fetch_sub(1, std::memory_order_relaxed);
        metrics_.rejected.inc();
        r.send(net::HttpResponse::text(503, "server overloaded, retry later\n"));
    }
}

InferenceService::Sample InferenceService::take_sample() const {
    Sample s;
    s.at = Clock::now();
    s.latency = metrics_.request_latency_us.snapshot();
    s.ok = metrics_.ok.value();
    s.bad_request = metrics_.bad_request.value();
    s.rejected = metrics_.rejected.value();
    s.timeouts = metrics_.timeouts.value();
    s.errors = metrics_.errors.value();
    if (batcher_metrics_) {
        s.queue_wait = batcher_metrics_->queue_wait_us.snapshot();
        s.compute = batcher_metrics_->batch_compute_us.snapshot();
        s.batch_size = batcher_metrics_->batch_size.snapshot();
        s.batches = batcher_metrics_->batches.value();
        s.padded_slots = batcher_metrics_->padded_slots.value();
    }
    return s;
}

InferenceService::Sample InferenceService::baseline(std::chrono::seconds window) const {
    const Clock::time_point target = Clock::now() - window;
    std::lock_guard lock(ring_mu_);
    for (auto it = ring_.rbegin(); it != ring_.rend(); ++it) {
        if (it->at <= target) return *it;
    }
    return ring_.front();  // uptime shorter than the window: since start
}

void InferenceService::ticker_loop() {
    std::unique_lock lock(tick_mu_);
    while (!tick_cv_.wait_for(lock, opts_.tick, [this] { return stopping_; })) {
        Sample s = take_sample();
        std::lock_guard ring_lock(ring_mu_);
        ring_.push_back(std::move(s));
        // Keep one sample older than the longest window as its baseline.
        const Clock::time_point oldest_needed = ring_.back().at - opts_.max_stats_window - opts_.tick;
        while (ring_.size() > 2 && ring_[1].at <= oldest_needed) ring_.pop_front();
    }
}

std::string InferenceService::render_stats_json(std::chrono::seconds window) const {
    const Sample now = take_sample();
    const Sample base = baseline(window);
    const double secs = std::max(duration<double>(now.at - base.at).count(), 1e-6);

    const auto latency = now.latency - base.latency;
    const auto queue_wait = now.queue_wait - base.queue_wait;
    const auto compute = now.compute - base.compute;
    const auto batch = now.batch_size - base.batch_size;
    const std::uint64_t ok = now.ok - base.ok;

    std::string out = "{";
    appendf(out, "\"window_s\":%.1f,\"uptime_s\":%.1f,", secs, duration<double>(now.at - started_).count());
    appendf(out, "\"throughput_rps\":%.1f,", static_cast<double>(ok) / secs);
    appendf(out,
            "\"requests\":{\"ok\":%llu,\"bad_request\":%llu,\"rejected\":%llu,\"timeouts\":%llu,\"errors\":%llu},",
            static_cast<unsigned long long>(ok), static_cast<unsigned long long>(now.bad_request - base.bad_request),
            static_cast<unsigned long long>(now.rejected - base.rejected),
            static_cast<unsigned long long>(now.timeouts - base.timeouts),
            static_cast<unsigned long long>(now.errors - base.errors));
    append_percentiles_ms(out, "latency_ms", latency);
    out += ',';
    append_percentiles_ms(out, "queue_wait_ms", queue_wait);
    out += ',';
    append_percentiles_ms(out, "batch_compute_ms", compute);

    const std::uint64_t batches = now.batches - base.batches;
    const std::uint64_t padded = now.padded_slots - base.padded_slots;
    appendf(out, ",\"batching\":{\"batches\":%llu,\"avg_batch_size\":%.2f,\"padding_waste\":%.3f,\"batch_sizes\":{",
            static_cast<unsigned long long>(batches), batch.mean(),
            (batch.sum + padded) > 0 ? static_cast<double>(padded) / static_cast<double>(batch.sum + padded) : 0.0);
    bool first = true;
    for (std::size_t i = 0; i < batch.counts.size(); ++i) {
        if (batch.counts[i] == 0) continue;
        appendf(out, "%s\"%zu\":%llu", first ? "" : ",", i, static_cast<unsigned long long>(batch.counts[i]));
        first = false;
    }
    out += "}},";

    appendf(out, "\"queue_depth\":%zu,\"inflight\":%lld,\"connections\":%zu,", batcher_.queue_depth(),
            static_cast<long long>(inflight_.load(std::memory_order_relaxed)),
            server_ ? server_->active_connections() : std::size_t{0});
    appendf(out, "\"request_pool\":{\"in_use\":%zu,\"capacity\":%zu,\"high_watermark\":%zu}}\n", pool_.in_use(),
            pool_.capacity(), pool_.high_watermark());
    return out;
}

std::string InferenceService::render_prometheus() const {
    const Sample now = take_sample();
    const Sample base = baseline(opts_.max_stats_window);
    std::string out;
    out.reserve(8192);

    out += "# HELP infer_requests_total Requests to /v1/infer by HTTP status.\n# TYPE infer_requests_total counter\n";
    appendf(out, "infer_requests_total{code=\"200\"} %llu\n", static_cast<unsigned long long>(now.ok));
    appendf(out, "infer_requests_total{code=\"400\"} %llu\n", static_cast<unsigned long long>(now.bad_request));
    appendf(out, "infer_requests_total{code=\"500\"} %llu\n", static_cast<unsigned long long>(now.errors));
    appendf(out, "infer_requests_total{code=\"503\"} %llu\n", static_cast<unsigned long long>(now.rejected));
    appendf(out, "infer_requests_total{code=\"504\"} %llu\n", static_cast<unsigned long long>(now.timeouts));

    append_summary(out, "infer_request_latency_seconds",
                   "Server-side latency of successful inferences; quantiles over the last 60 s.",
                   now.latency - base.latency, now.latency);
    append_summary(out, "infer_queue_wait_seconds", "Time requests waited for their batch; quantiles over 60 s.",
                   now.queue_wait - base.queue_wait, now.queue_wait);
    append_summary(out, "infer_batch_compute_seconds", "Model run time per batch; quantiles over 60 s.",
                   now.compute - base.compute, now.compute);

    out += "# HELP infer_batch_size Requests per executed batch.\n# TYPE infer_batch_size histogram\n";
    std::uint64_t cumulative = 0;
    for (std::size_t i = 1; i < now.batch_size.counts.size(); ++i) {
        cumulative += now.batch_size.counts[i];
        appendf(out, "infer_batch_size_bucket{le=\"%zu\"} %llu\n", i, static_cast<unsigned long long>(cumulative));
    }
    appendf(out, "infer_batch_size_bucket{le=\"+Inf\"} %llu\ninfer_batch_size_sum %llu\ninfer_batch_size_count %llu\n",
            static_cast<unsigned long long>(now.batch_size.count), static_cast<unsigned long long>(now.batch_size.sum),
            static_cast<unsigned long long>(now.batch_size.count));

    append_metric(out, "infer_batches_total", "counter", "Batches executed.", static_cast<double>(now.batches));
    append_metric(out, "infer_padded_slots_total", "counter", "Batch rows run only to fill a batch-size bucket.",
                  static_cast<double>(now.padded_slots));
    append_metric(out, "infer_queue_depth", "gauge", "Requests waiting for a batch.",
                  static_cast<double>(batcher_.queue_depth()));
    append_metric(out, "infer_inflight_requests", "gauge", "Accepted inference requests not yet answered.",
                  static_cast<double>(inflight_.load(std::memory_order_relaxed)));
    append_metric(out, "infer_request_pool_blocks_in_use", "gauge", "Request input blocks in use.",
                  static_cast<double>(pool_.in_use()));
    append_metric(out, "infer_request_pool_blocks_capacity", "gauge", "Request input blocks in the pool.",
                  static_cast<double>(pool_.capacity()));
    append_metric(out, "infer_request_pool_blocks_high_watermark", "gauge", "Most request blocks ever in use at once.",
                  static_cast<double>(pool_.high_watermark()));
    if (server_) {
        append_metric(out, "infer_connections_active", "gauge", "Open client connections.",
                      static_cast<double>(server_->active_connections()));
        append_metric(out, "infer_connections_accepted_total", "counter", "Client connections accepted.",
                      static_cast<double>(server_->accepted_connections()));
    }
    append_metric(out, "infer_uptime_seconds", "gauge", "Seconds since the service started.",
                  duration<double>(now.at - started_).count());
    return out;
}

}  // namespace infer
