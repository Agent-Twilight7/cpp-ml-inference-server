#include "infer/batching/dynamic_batcher.hpp"
#include "infer/memory/block_pool.hpp"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <future>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

using namespace std::chrono_literals;
using infer::BatcherOptions;
using infer::DynamicBatcher;
using infer::InferResult;
using infer::InferStatus;

namespace {

// Each sample is kIn floats; the output is kOut floats: [sum(input), 2*input[0]].
// Records every batch size it is asked to run (excluding warm-up when reset).
class FakeBackend final : public infer::ModelBackend {
public:
    static constexpr std::size_t kIn = 4;
    static constexpr std::size_t kOut = 2;

    struct Shared {
        std::mutex mu;
        std::vector<std::size_t> batch_sizes;
        std::chrono::microseconds compute_delay{0};
        std::atomic<bool> fail{false};
        std::size_t pad_to_multiple = 1;  // emulate a backend with fixed batch buckets
        void reset() {
            std::lock_guard lock(mu);
            batch_sizes.clear();
        }
        std::vector<std::size_t> sizes() {
            std::lock_guard lock(mu);
            return batch_sizes;
        }
    };

    explicit FakeBackend(Shared& shared) : shared_(shared) {}

    std::size_t input_elems_per_sample() const override { return kIn; }
    std::size_t output_elems_per_sample() const override { return kOut; }
    std::size_t padded_batch_size(std::size_t n) const override {
        const std::size_t m = shared_.pad_to_multiple;
        return (n + m - 1) / m * m;
    }

    void run(std::span<const float> input, std::span<float> output, std::size_t batch) override {
        {
            std::lock_guard lock(shared_.mu);
            shared_.batch_sizes.push_back(batch);
        }
        if (shared_.fail) throw std::runtime_error("boom");
        if (shared_.compute_delay.count() > 0) std::this_thread::sleep_for(shared_.compute_delay);
        for (std::size_t b = 0; b < batch; ++b) {
            float sum = 0;
            for (std::size_t i = 0; i < kIn; ++i) sum += input[b * kIn + i];
            output[b * kOut] = sum;
            output[b * kOut + 1] = 2 * input[b * kIn];
        }
    }

private:
    Shared& shared_;
};

// What a callback saw, copied out (the output span dies after the callback).
struct Outcome {
    InferStatus status;
    std::vector<float> output;
    std::size_t batch_size;
    std::chrono::microseconds queue_time;
};

class BatcherTest : public ::testing::Test {
protected:
    // A pool block holding one sample filled with `v`.
    infer::BlockPool::Block sample(float v) {
        infer::BlockPool::Block b = pool_.try_acquire();
        EXPECT_TRUE(b) << "test pool exhausted";
        for (float& x : b.span()) x = v;
        return b;
    }

    std::unique_ptr<DynamicBatcher> make(BatcherOptions opts) {
        auto b = std::make_unique<DynamicBatcher>(opts, [this] { return std::make_unique<FakeBackend>(shared_); });
        b->start();
        shared_.reset();  // forget warm-up runs
        return b;
    }

    // Submits and returns a future for the outcome.
    std::future<Outcome> submit(DynamicBatcher& b, float v) {
        auto promise = std::make_shared<std::promise<Outcome>>();
        auto fut = promise->get_future();
        const bool ok = b.submit(sample(v), [promise](const InferResult& r) {
            promise->set_value(Outcome{r.status, std::vector<float>(r.output.begin(), r.output.end()), r.batch_size,
                                       r.queue_time});
        });
        EXPECT_TRUE(ok);
        return fut;
    }

    FakeBackend::Shared shared_;
    infer::BlockPool pool_{FakeBackend::kIn, 4096};  // outlives every batcher in a test
};

}  // namespace

TEST_F(BatcherTest, WarmsUpPowerOfTwoBatchSizes) {
    auto b = std::make_unique<DynamicBatcher>(BatcherOptions{.max_batch_size = 12},
                                              [this] { return std::make_unique<FakeBackend>(shared_); });
    b->start();
    EXPECT_EQ(shared_.sizes(), (std::vector<std::size_t>{1, 2, 4, 8, 12}));
}

TEST_F(BatcherTest, LoneRequestRunsAfterMaxDelay) {
    auto b = make({.max_batch_size = 8, .max_queue_delay = 30ms});
    const auto t0 = infer::Clock::now();
    Outcome o = submit(*b, 1.5f).get();
    const auto elapsed = infer::Clock::now() - t0;

    EXPECT_EQ(o.status, InferStatus::kOk);
    EXPECT_EQ(o.batch_size, 1u);
    EXPECT_EQ(o.output, (std::vector<float>{6.0f, 3.0f}));
    EXPECT_GE(elapsed, 30ms);   // waited for company...
    EXPECT_LT(elapsed, 500ms);  // ...but not forever
}

TEST_F(BatcherTest, FullBatchRunsWithoutWaitingForDelay) {
    auto b = make({.max_batch_size = 4, .max_queue_delay = 10s});
    const auto t0 = infer::Clock::now();
    std::vector<std::future<Outcome>> futs;
    for (int i = 0; i < 4; ++i) futs.push_back(submit(*b, static_cast<float>(i)));
    for (auto& f : futs) EXPECT_EQ(f.get().batch_size, 4u);
    EXPECT_LT(infer::Clock::now() - t0, 2s);
    EXPECT_EQ(shared_.sizes(), (std::vector<std::size_t>{4}));
}

TEST_F(BatcherTest, EachCallerGetsItsOwnResult) {
    auto b = make({.max_batch_size = 16, .max_queue_delay = 20ms});
    std::vector<std::future<Outcome>> futs;
    for (int i = 0; i < 40; ++i) futs.push_back(submit(*b, static_cast<float>(i)));
    for (int i = 0; i < 40; ++i) {
        Outcome o = futs[i].get();
        ASSERT_EQ(o.status, InferStatus::kOk);
        EXPECT_EQ(o.output[0], 4.0f * static_cast<float>(i)) << "request " << i;
        EXPECT_EQ(o.output[1], 2.0f * static_cast<float>(i)) << "request " << i;
    }
    for (std::size_t s : shared_.sizes()) EXPECT_LE(s, 16u);
    EXPECT_EQ(pool_.in_use(), 0u) << "input blocks must return to the pool once batched";
}

TEST_F(BatcherTest, PadsToBackendBucketsAndIgnoresPaddingRows) {
    shared_.pad_to_multiple = 4;
    auto b = make({.max_batch_size = 6, .max_queue_delay = 20ms});
    std::vector<std::future<Outcome>> futs;
    for (int i = 1; i <= 5; ++i) futs.push_back(submit(*b, static_cast<float>(i)));
    for (int i = 1; i <= 5; ++i) {
        Outcome o = futs[i - 1].get();
        ASSERT_EQ(o.status, InferStatus::kOk);
        EXPECT_EQ(o.output[0], 4.0f * static_cast<float>(i));
        EXPECT_LE(o.batch_size, 5u);  // reports real requests, not padding
    }
    for (std::size_t s : shared_.sizes()) EXPECT_EQ(s % 4, 0u) << "backend got unpadded batch " << s;
}

TEST_F(BatcherTest, RejectsWhenQueueFull) {
    shared_.compute_delay = 200ms;  // keep the single worker busy
    auto b = make({.max_batch_size = 1, .max_queue_delay = 0us, .max_queue_size = 2, .num_workers = 1});
    auto first = submit(*b, 0);   // picked up by the worker
    std::this_thread::sleep_for(50ms);
    auto q1 = submit(*b, 1);      // queued
    auto q2 = submit(*b, 2);      // queued (queue now full)
    EXPECT_FALSE(b->submit(sample(3), [](const InferResult&) { FAIL() << "rejected request must not complete"; }));
    first.get();
    q1.get();
    q2.get();
}

TEST_F(BatcherTest, StopDrainsQueuedRequests) {
    shared_.compute_delay = 20ms;
    auto b = make({.max_batch_size = 2, .max_queue_delay = 10s, .num_workers = 1});
    std::vector<std::future<Outcome>> futs;
    for (int i = 0; i < 7; ++i) futs.push_back(submit(*b, static_cast<float>(i)));
    b->stop();  // must not wait 10 s for the odd one out
    for (auto& f : futs) {
        ASSERT_EQ(f.wait_for(0s), std::future_status::ready);
        EXPECT_EQ(f.get().status, InferStatus::kOk);
    }
    EXPECT_FALSE(b->submit(sample(0), [](const InferResult&) {}));
}

TEST_F(BatcherTest, BackendFailureFailsWholeBatch) {
    auto b = make({.max_batch_size = 3, .max_queue_delay = 10s});
    shared_.fail = true;
    std::vector<std::future<Outcome>> futs;
    for (int i = 0; i < 3; ++i) futs.push_back(submit(*b, 0));
    for (auto& f : futs) EXPECT_EQ(f.get().status, InferStatus::kError);
}

TEST_F(BatcherTest, ExpiredRequestsTimeOut) {
    shared_.compute_delay = 100ms;
    auto b = make({.max_batch_size = 1, .max_queue_delay = 0us, .queue_timeout = 30ms, .num_workers = 1});
    auto first = submit(*b, 0);  // occupies the worker for 100 ms
    std::this_thread::sleep_for(10ms);
    auto late = submit(*b, 1);   // waits ~90 ms > 30 ms timeout
    EXPECT_EQ(first.get().status, InferStatus::kOk);
    Outcome o = late.get();
    EXPECT_EQ(o.status, InferStatus::kTimeout);
    EXPECT_GE(o.queue_time, 30ms);
}

TEST_F(BatcherTest, ConcurrentProducersMultipleWorkers) {
    shared_.compute_delay = 2ms;
    auto b = make({.max_batch_size = 8, .max_queue_delay = 1ms, .max_queue_size = 10000, .num_workers = 3});
    constexpr int kThreads = 8;
    constexpr int kEach = 200;
    std::atomic<int> correct{0};
    std::atomic<int> done{0};
    std::vector<std::thread> producers;
    for (int t = 0; t < kThreads; ++t) {
        producers.emplace_back([&, t] {
            for (int i = 0; i < kEach; ++i) {
                const float v = static_cast<float>(t * kEach + i);
                ASSERT_TRUE(b->submit(sample(v), [&, v](const InferResult& r) {
                    if (r.status == InferStatus::kOk && r.output[0] == 4 * v) correct.fetch_add(1);
                    done.fetch_add(1);
                }));
            }
        });
    }
    for (auto& p : producers) p.join();
    b->stop();
    EXPECT_EQ(done.load(), kThreads * kEach);
    EXPECT_EQ(correct.load(), kThreads * kEach);

    std::size_t total = 0, batches = 0;
    for (std::size_t s : shared_.sizes()) {
        EXPECT_LE(s, 8u);
        total += s;
        ++batches;
    }
    EXPECT_EQ(total, static_cast<std::size_t>(kThreads * kEach));
    EXPECT_LT(batches, total);  // some batching actually happened
}
