#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace infer::metrics {

// Monotonic counter. Relaxed atomics: totals only need to be eventually exact.
class Counter {
public:
    void inc(std::uint64_t n = 1) { v_.fetch_add(n, std::memory_order_relaxed); }
    std::uint64_t value() const { return v_.load(std::memory_order_relaxed); }

private:
    std::atomic<std::uint64_t> v_{0};
};

// Log-linear histogram of non-negative integers (we record microseconds), in
// the style of HdrHistogram: values below 16 get exact buckets, and every
// power-of-two range above that is split into 16 equal sub-buckets. So a
// bucket is never wider than 1/16 of its lower bound, which bounds percentile
// error at 6.25% across nine-plus orders of magnitude with only ~600 buckets.
//
// record() is wait-free: three relaxed fetch_adds, no locks. Readers take a
// Snapshot (a plain copy) and compute statistics from it; snapshots subtract,
// which is how windowed percentiles are computed.
class LatencyHistogram {
public:
    static constexpr unsigned kSubBits = 4;
    static constexpr std::uint64_t kSub = 1u << kSubBits;  // 16 sub-buckets per octave
    static constexpr unsigned kMaxMsb = 39;                // ~5.5e11 us; larger values clamp
    static constexpr std::size_t kBuckets = kSub + (kMaxMsb - kSubBits + 1) * kSub;

    struct Snapshot {
        std::array<std::uint64_t, kBuckets> counts{};
        std::uint64_t count = 0;
        std::uint64_t sum = 0;

        Snapshot operator-(const Snapshot& older) const;
        // q in [0, 1]. Linear interpolation within the bucket. 0 if empty.
        double percentile(double q) const;
        double mean() const { return count ? static_cast<double>(sum) / static_cast<double>(count) : 0.0; }
        // Upper bound of the highest non-empty bucket (0 if empty).
        double max() const;
    };

    void record(std::uint64_t value);
    Snapshot snapshot() const;

    static std::size_t bucket_index(std::uint64_t value);
    static std::uint64_t bucket_lower(std::size_t index);
    static std::uint64_t bucket_upper(std::size_t index);  // exclusive

private:
    std::array<std::atomic<std::uint64_t>, kBuckets> counts_{};
    std::atomic<std::uint64_t> count_{0};
    std::atomic<std::uint64_t> sum_{0};
};

// Exact counts for small integers 0..max (larger values land in the last
// bucket). Used for batch sizes.
class SmallHistogram {
public:
    explicit SmallHistogram(std::size_t max_value);

    struct Snapshot {
        std::vector<std::uint64_t> counts;  // counts[v] = how many times v was recorded
        std::uint64_t count = 0;
        std::uint64_t sum = 0;

        Snapshot operator-(const Snapshot& older) const;
        double mean() const { return count ? static_cast<double>(sum) / static_cast<double>(count) : 0.0; }
    };

    void record(std::size_t value);
    Snapshot snapshot() const;
    std::size_t max_value() const { return max_; }

private:
    std::size_t max_;
    std::unique_ptr<std::atomic<std::uint64_t>[]> counts_;
    std::atomic<std::uint64_t> count_{0};
    std::atomic<std::uint64_t> sum_{0};
};

}  // namespace infer::metrics
