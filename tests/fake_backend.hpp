#pragma once

#include "infer/runtime/model_backend.hpp"

#include <atomic>
#include <chrono>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

namespace testing_support {

// Each sample is kIn floats; the output is kOut floats: [sum(input), 2*input[0]].
// Records every batch size it is asked to run.
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

}  // namespace testing_support
