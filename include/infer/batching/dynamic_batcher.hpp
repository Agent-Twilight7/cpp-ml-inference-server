#pragma once

#include "infer/memory/block_pool.hpp"
#include "infer/runtime/model_backend.hpp"

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <span>
#include <string_view>
#include <thread>
#include <vector>

namespace infer {

using Clock = std::chrono::steady_clock;

enum class InferStatus { kOk, kTimeout, kError };

struct InferResult {
    InferStatus status = InferStatus::kOk;
    std::span<const float> output{};  // this request's slice; valid only during the callback
    std::size_t batch_size = 0;     // how many requests ran together
    std::chrono::microseconds queue_time{0};    // submit -> batch formed
    std::chrono::microseconds compute_time{0};  // backend run() for the whole batch
    std::string_view error{};       // set when status == kError
};

// Called exactly once per accepted request, on a worker thread.
using InferCallback = std::function<void(const InferResult&)>;

using BackendFactory = std::function<std::unique_ptr<ModelBackend>()>;

struct BatcherOptions {
    std::size_t max_batch_size = 16;
    // A batch runs when it is full, or when its oldest request has waited this long.
    std::chrono::microseconds max_queue_delay{2000};
    // Requests beyond this many waiting are rejected (backpressure).
    std::size_t max_queue_size = 1024;
    // Requests that waited longer than this when picked up fail with kTimeout.
    // Zero disables the check.
    std::chrono::microseconds queue_timeout{0};
    int num_workers = 1;
};

// Groups single-sample requests from many threads into batches and runs them
// on a set of worker threads, each owning one ModelBackend.
//
//   submit() ──▶ [ bounded FIFO ] ──▶ batch former ──▶ worker run() ──▶ callbacks
//
// Exactly one worker forms a batch at a time (the "leader"); once it has
// taken its batch it hands leadership to the next idle worker and goes off to
// compute. So while one batch runs on the device, the next one is already
// being collected.
class DynamicBatcher {
public:
    DynamicBatcher(BatcherOptions opts, const BackendFactory& make_backend);
    ~DynamicBatcher();

    DynamicBatcher(const DynamicBatcher&) = delete;
    DynamicBatcher& operator=(const DynamicBatcher&) = delete;

    // Warms each backend on every power-of-two batch size up to the max, then
    // launches the worker threads.
    void start();

    // Thread-safe. `input` is a pool block holding exactly input_elems()
    // floats; it goes back to its pool as soon as it has been copied into a
    // batch. Returns false, without calling `done`, if the queue is full or the
    // batcher is stopping (the block is then released by the caller's handle).
    bool submit(BlockPool::Block&& input, InferCallback done);

    // Stops accepting work, lets the workers finish everything already queued
    // (without waiting out max_queue_delay), and joins them. Idempotent.
    void stop();

    std::size_t input_elems() const { return input_elems_; }
    std::size_t output_elems() const { return output_elems_; }
    std::size_t queue_depth() const;
    const BatcherOptions& options() const { return opts_; }

private:
    struct Request {
        BlockPool::Block input;
        InferCallback done;
        Clock::time_point enqueued;
    };

    struct Worker {
        // Declared before the buffers: the backend may own their allocator,
        // so it must be destroyed after them.
        std::unique_ptr<ModelBackend> backend;
        HostBuffer batch_in;   // padded max batch * input_elems, allocated once
        HostBuffer batch_out;  // padded max batch * output_elems, allocated once
        std::vector<Request> batch;  // reused across batches
        std::thread thread;

        std::span<const float> inputs(std::size_t n, std::size_t elems) const { return {batch_in.get(), n * elems}; }
        std::span<float> outputs(std::size_t n, std::size_t elems) const { return {batch_out.get(), n * elems}; }
    };

    void worker_loop(Worker& w);
    // Blocks until a batch is ready and moves it into `out`. Returns false once
    // the batcher is stopping and the queue is empty.
    bool form_batch(std::vector<Request>& out);
    void run_batch(Worker& w, Clock::time_point formed);

    BatcherOptions opts_;
    std::size_t input_elems_ = 0;
    std::size_t output_elems_ = 0;
    std::vector<std::unique_ptr<Worker>> workers_;

    std::mutex leader_mu_;  // held by the worker currently forming a batch

    mutable std::mutex mu_;  // guards queue_ and stopping_
    std::condition_variable cv_;
    std::deque<Request> queue_;
    bool stopping_ = false;
    bool started_ = false;
};

}  // namespace infer
