#include "infer/metrics/histogram.hpp"

#include <algorithm>
#include <bit>

namespace infer::metrics {

// ---- LatencyHistogram ----

std::size_t LatencyHistogram::bucket_index(std::uint64_t v) {
    if (v < kSub) return static_cast<std::size_t>(v);
    unsigned msb = static_cast<unsigned>(std::bit_width(v)) - 1;  // >= kSubBits
    if (msb > kMaxMsb) {
        msb = kMaxMsb;
        v = (std::uint64_t{1} << (kMaxMsb + 1)) - 1;  // clamp to the top bucket
    }
    const unsigned shift = msb - kSubBits;
    const std::uint64_t sub = (v >> shift) - kSub;  // 0..kSub-1
    return static_cast<std::size_t>(kSub + (msb - kSubBits) * kSub + sub);
}

std::uint64_t LatencyHistogram::bucket_lower(std::size_t i) {
    if (i < kSub) return i;
    const std::size_t k = i - kSub;
    const unsigned octave = static_cast<unsigned>(k / kSub);  // msb - kSubBits
    const std::uint64_t sub = k % kSub;
    return (kSub + sub) << octave;
}

std::uint64_t LatencyHistogram::bucket_upper(std::size_t i) {
    if (i + 1 < kBuckets) return bucket_lower(i + 1);
    return std::uint64_t{1} << (kMaxMsb + 1);
}

void LatencyHistogram::record(std::uint64_t value) {
    counts_[bucket_index(value)].fetch_add(1, std::memory_order_relaxed);
    count_.fetch_add(1, std::memory_order_relaxed);
    sum_.fetch_add(value, std::memory_order_relaxed);
}

LatencyHistogram::Snapshot LatencyHistogram::snapshot() const {
    // Not an atomic cut across buckets: a record() racing with the copy may
    // show up in count but not yet in its bucket. Percentiles use the sum of
    // the copied buckets, so they stay self-consistent.
    Snapshot s;
    for (std::size_t i = 0; i < kBuckets; ++i) s.counts[i] = counts_[i].load(std::memory_order_relaxed);
    s.count = count_.load(std::memory_order_relaxed);
    s.sum = sum_.load(std::memory_order_relaxed);
    return s;
}

LatencyHistogram::Snapshot LatencyHistogram::Snapshot::operator-(const Snapshot& older) const {
    Snapshot d;
    for (std::size_t i = 0; i < kBuckets; ++i) d.counts[i] = counts[i] - older.counts[i];
    d.count = count - older.count;
    d.sum = sum - older.sum;
    return d;
}

double LatencyHistogram::Snapshot::percentile(double q) const {
    std::uint64_t total = 0;
    for (std::uint64_t c : counts) total += c;
    if (total == 0) return 0.0;
    q = std::clamp(q, 0.0, 1.0);

    const double rank = q * static_cast<double>(total);
    double cumulative = 0.0;
    for (std::size_t i = 0; i < kBuckets; ++i) {
        if (counts[i] == 0) continue;
        const double next = cumulative + static_cast<double>(counts[i]);
        if (next >= rank) {
            const double frac = (rank - cumulative) / static_cast<double>(counts[i]);
            const double lo = static_cast<double>(bucket_lower(i));
            const double hi = static_cast<double>(bucket_upper(i));
            return lo + frac * (hi - lo);
        }
        cumulative = next;
    }
    return max();
}

double LatencyHistogram::Snapshot::max() const {
    for (std::size_t i = kBuckets; i-- > 0;) {
        if (counts[i] != 0) return static_cast<double>(bucket_upper(i));
    }
    return 0.0;
}

// ---- SmallHistogram ----

SmallHistogram::SmallHistogram(std::size_t max_value)
    : max_(max_value), counts_(std::make_unique<std::atomic<std::uint64_t>[]>(max_value + 1)) {}

void SmallHistogram::record(std::size_t value) {
    counts_[std::min(value, max_)].fetch_add(1, std::memory_order_relaxed);
    count_.fetch_add(1, std::memory_order_relaxed);
    sum_.fetch_add(value, std::memory_order_relaxed);
}

SmallHistogram::Snapshot SmallHistogram::snapshot() const {
    Snapshot s;
    s.counts.resize(max_ + 1);
    for (std::size_t i = 0; i <= max_; ++i) s.counts[i] = counts_[i].load(std::memory_order_relaxed);
    s.count = count_.load(std::memory_order_relaxed);
    s.sum = sum_.load(std::memory_order_relaxed);
    return s;
}

SmallHistogram::Snapshot SmallHistogram::Snapshot::operator-(const Snapshot& older) const {
    Snapshot d;
    d.counts.resize(counts.size());
    for (std::size_t i = 0; i < counts.size(); ++i) {
        d.counts[i] = counts[i] - (i < older.counts.size() ? older.counts[i] : 0);
    }
    d.count = count - older.count;
    d.sum = sum - older.sum;
    return d;
}

}  // namespace infer::metrics
