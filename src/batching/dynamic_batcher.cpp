#include "infer/batching/dynamic_batcher.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <exception>
#include <stdexcept>

namespace infer {
namespace {

std::chrono::microseconds micros_between(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration_cast<std::chrono::microseconds>(b - a);
}

void invoke_callback(const InferCallback& done, const InferResult& result) {
    // A throwing callback must not take down the worker or skip the others.
    try {
        done(result);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "batcher: callback threw: %s\n", e.what());
    } catch (...) {
        std::fprintf(stderr, "batcher: callback threw\n");
    }
}

}  // namespace

DynamicBatcher::DynamicBatcher(BatcherOptions opts, const BackendFactory& make_backend) : opts_(opts) {
    if (opts_.max_batch_size == 0) throw std::invalid_argument("max_batch_size must be >= 1");
    if (opts_.num_workers < 1) throw std::invalid_argument("num_workers must be >= 1");
    if (opts_.max_queue_size == 0) throw std::invalid_argument("max_queue_size must be >= 1");

    for (int i = 0; i < opts_.num_workers; ++i) {
        auto w = std::make_unique<Worker>();
        w->backend = make_backend();
        if (i == 0) {
            input_elems_ = w->backend->input_elems_per_sample();
            output_elems_ = w->backend->output_elems_per_sample();
        } else if (w->backend->input_elems_per_sample() != input_elems_ ||
                   w->backend->output_elems_per_sample() != output_elems_) {
            throw std::invalid_argument("all backends must have the same shapes");
        }
        // Room for the largest batch after the backend pads it.
        const std::size_t capacity = w->backend->padded_batch_size(opts_.max_batch_size);
        // Allocated by the backend so a GPU backend can make them pinned memory.
        w->batch_in = w->backend->allocate_host_buffer(capacity * input_elems_);
        w->batch_out = w->backend->allocate_host_buffer(capacity * output_elems_);
        std::fill_n(w->batch_in.get(), capacity * input_elems_, 0.0f);
        w->batch.reserve(opts_.max_batch_size);
        workers_.push_back(std::move(w));
    }
}

DynamicBatcher::~DynamicBatcher() { stop(); }

void DynamicBatcher::start() {
    if (started_) throw std::logic_error("DynamicBatcher already started");
    started_ = true;

    // Run each batch size once up front so lazy initialization (cuDNN kernel
    // choice per shape, memory arenas) doesn't land on real requests.
    for (auto& w : workers_) {
        std::size_t last = 0;
        for (std::size_t b = 1;; b *= 2) {
            const std::size_t n = std::min(b, opts_.max_batch_size);
            const std::size_t p = w->backend->padded_batch_size(n);
            if (p != last) {
                w->backend->run(w->inputs(p, input_elems_), w->outputs(p, output_elems_), p);
                last = p;
            }
            if (n == opts_.max_batch_size) break;
        }
    }
    for (auto& w : workers_) {
        Worker* wp = w.get();
        w->thread = std::thread([this, wp] { worker_loop(*wp); });
    }
}

bool DynamicBatcher::submit(BlockPool::Block&& input, InferCallback done) {
    if (input.size() != input_elems_) throw std::invalid_argument("submit: wrong input size");
    std::size_t depth;
    {
        std::lock_guard lock(mu_);
        if (stopping_ || queue_.size() >= opts_.max_queue_size) return false;
        queue_.push_back(Request{std::move(input), std::move(done), Clock::now()});
        depth = queue_.size();
    }
    // The leader only cares about two transitions: the queue becoming
    // non-empty (start the delay clock) and the batch filling up (run now).
    if (depth == 1 || depth == opts_.max_batch_size) cv_.notify_one();
    return true;
}

void DynamicBatcher::stop() {
    {
        std::lock_guard lock(mu_);
        if (stopping_) return;
        stopping_ = true;
    }
    cv_.notify_all();
    for (auto& w : workers_) {
        if (w->thread.joinable()) w->thread.join();
    }
    // If start() was never called, nobody will run what's queued: fail it.
    std::deque<Request> leftover;
    {
        std::lock_guard lock(mu_);
        leftover.swap(queue_);
    }
    for (auto& r : leftover) {
        invoke_callback(r.done, InferResult{.status = InferStatus::kError, .error = "batcher stopped"});
    }
}

std::size_t DynamicBatcher::queue_depth() const {
    std::lock_guard lock(mu_);
    return queue_.size();
}

bool DynamicBatcher::form_batch(std::vector<Request>& out) {
    // Only the leader waits on the queue; other idle workers wait here. That
    // keeps a single thread on the condition variable, so notify_one() in
    // submit() always reaches the thread that is collecting.
    std::lock_guard leader(leader_mu_);
    std::unique_lock lock(mu_);

    cv_.wait(lock, [&] { return stopping_ || !queue_.empty(); });
    if (queue_.empty()) return false;  // stopping and drained

    if (!stopping_) {
        // The delay is measured from when the *oldest* request arrived, so a
        // request that already waited behind the previous batch goes right away.
        const Clock::time_point deadline = queue_.front().enqueued + opts_.max_queue_delay;
        cv_.wait_until(lock, deadline, [&] { return stopping_ || queue_.size() >= opts_.max_batch_size; });
    }

    const std::size_t n = std::min(queue_.size(), opts_.max_batch_size);
    for (std::size_t i = 0; i < n; ++i) {
        out.push_back(std::move(queue_.front()));
        queue_.pop_front();
    }
    return true;
}

void DynamicBatcher::worker_loop(Worker& w) {
    while (form_batch(w.batch)) {
        run_batch(w, Clock::now());
        w.batch.clear();
    }
}

void DynamicBatcher::run_batch(Worker& w, Clock::time_point formed) {
    // Drop requests that already waited too long; the client has likely given up.
    if (opts_.queue_timeout.count() > 0) {
        auto expired = std::stable_partition(w.batch.begin(), w.batch.end(), [&](const Request& r) {
            return formed - r.enqueued <= opts_.queue_timeout;
        });
        for (auto it = expired; it != w.batch.end(); ++it) {
            invoke_callback(it->done, InferResult{.status = InferStatus::kTimeout,
                                         .queue_time = micros_between(it->enqueued, formed)});
        }
        w.batch.erase(expired, w.batch.end());
        if (w.batch.empty()) return;
    }

    // Gather into the batch buffer, returning each request's block to the pool
    // right away so it can take a new request while this batch computes.
    const std::size_t n = w.batch.size();
    for (std::size_t i = 0; i < n; ++i) {
        std::memcpy(w.batch_in.get() + i * input_elems_, w.batch[i].input.data(), input_elems_ * sizeof(float));
        w.batch[i].input.reset();
    }

    const Clock::time_point t0 = Clock::now();
    try {
        // Rows n..padded-1 hold stale data from earlier batches; their outputs
        // are ignored, so there's no need to clear them.
        const std::size_t padded = w.backend->padded_batch_size(n);
        w.backend->run(w.inputs(padded, input_elems_), w.outputs(padded, output_elems_), padded);
    } catch (const std::exception& e) {
        for (auto& r : w.batch) {
            invoke_callback(r.done, InferResult{.status = InferStatus::kError, .batch_size = n, .error = e.what()});
        }
        return;
    }
    const auto compute = micros_between(t0, Clock::now());

    for (std::size_t i = 0; i < n; ++i) {
        const Request& r = w.batch[i];
        invoke_callback(r.done, InferResult{
                           .status = InferStatus::kOk,
                           .output = std::span<const float>(w.batch_out.get() + i * output_elems_, output_elems_),
                           .batch_size = n,
                           .queue_time = micros_between(r.enqueued, formed),
                           .compute_time = compute,
                       });
    }
}

}  // namespace infer
