#include "infer/metrics/histogram.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <random>
#include <thread>
#include <vector>

using infer::metrics::LatencyHistogram;
using infer::metrics::SmallHistogram;

TEST(LatencyHistogram, SmallValuesHaveExactBuckets) {
    for (std::uint64_t v = 0; v < LatencyHistogram::kSub; ++v) {
        const std::size_t i = LatencyHistogram::bucket_index(v);
        EXPECT_EQ(LatencyHistogram::bucket_lower(i), v);
        EXPECT_EQ(LatencyHistogram::bucket_upper(i), v + 1);
    }
}

TEST(LatencyHistogram, BucketsContainTheirValuesAndStayNarrow) {
    std::mt19937_64 rng(42);
    std::size_t prev_index = 0;
    for (std::uint64_t v = 0; v < 100000; ++v) {  // dense sweep: indices never decrease
        const std::size_t i = LatencyHistogram::bucket_index(v);
        ASSERT_GE(i, prev_index);
        prev_index = i;
    }
    for (int k = 0; k < 100000; ++k) {
        const std::uint64_t v = rng() >> (rng() % 40 + 24);  // spread over many magnitudes
        const std::size_t i = LatencyHistogram::bucket_index(v);
        ASSERT_LT(i, LatencyHistogram::kBuckets);
        const std::uint64_t lo = LatencyHistogram::bucket_lower(i);
        const std::uint64_t hi = LatencyHistogram::bucket_upper(i);
        ASSERT_LE(lo, v);
        ASSERT_LT(v, hi);
        if (lo >= LatencyHistogram::kSub) {
            ASSERT_LE(static_cast<double>(hi - lo) / static_cast<double>(lo), 1.0 / LatencyHistogram::kSub + 1e-12);
        }
    }
}

TEST(LatencyHistogram, HugeValuesClampToLastBucket) {
    EXPECT_EQ(LatencyHistogram::bucket_index(UINT64_MAX), LatencyHistogram::kBuckets - 1);
}

TEST(LatencyHistogram, PercentilesWithinBucketError) {
    LatencyHistogram h;
    std::vector<std::uint64_t> values;
    std::mt19937_64 rng(7);
    std::lognormal_distribution<double> dist(8.0, 1.0);  // latency-like, median ~3 ms in us
    for (int i = 0; i < 200000; ++i) {
        const auto v = static_cast<std::uint64_t>(dist(rng));
        values.push_back(v);
        h.record(v);
    }
    std::sort(values.begin(), values.end());
    const auto snap = h.snapshot();
    EXPECT_EQ(snap.count, values.size());
    for (double q : {0.5, 0.9, 0.99, 0.999}) {
        const double exact = static_cast<double>(values[static_cast<std::size_t>(q * static_cast<double>(values.size() - 1))]);
        const double approx = snap.percentile(q);
        EXPECT_NEAR(approx, exact, exact * 0.0625) << "q=" << q;
    }
}

TEST(LatencyHistogram, SnapshotDifferenceIsolatesAWindow) {
    LatencyHistogram h;
    for (int i = 0; i < 1000; ++i) h.record(100);  // "old" traffic
    const auto before = h.snapshot();
    for (int i = 0; i < 1000; ++i) h.record(10000);  // "recent" traffic
    const auto window = h.snapshot() - before;
    EXPECT_EQ(window.count, 1000u);
    EXPECT_NEAR(window.percentile(0.5), 10000, 10000 * 0.0625);
    EXPECT_NEAR(window.mean(), 10000, 1e-9);
}

TEST(LatencyHistogram, EmptySnapshotIsZero) {
    LatencyHistogram h;
    const auto s = h.snapshot();
    EXPECT_EQ(s.percentile(0.99), 0.0);
    EXPECT_EQ(s.max(), 0.0);
    EXPECT_EQ(s.mean(), 0.0);
}

TEST(LatencyHistogram, ConcurrentRecordsAreAllCounted) {
    LatencyHistogram h;
    std::vector<std::thread> ts;
    for (int t = 0; t < 8; ++t) {
        ts.emplace_back([&h, t] {
            for (int i = 0; i < 50000; ++i) h.record(static_cast<std::uint64_t>(t * 1000 + i % 1000));
        });
    }
    for (auto& t : ts) t.join();
    const auto s = h.snapshot();
    std::uint64_t bucket_total = 0;
    for (auto c : s.counts) bucket_total += c;
    EXPECT_EQ(s.count, 400000u);
    EXPECT_EQ(bucket_total, 400000u);
}

TEST(SmallHistogram, CountsExactlyAndClamps) {
    SmallHistogram h(4);
    for (std::size_t v : {1u, 2u, 2u, 4u, 9u}) h.record(v);
    const auto s = h.snapshot();
    EXPECT_EQ(s.counts[1], 1u);
    EXPECT_EQ(s.counts[2], 2u);
    EXPECT_EQ(s.counts[4], 2u);  // 4 and the clamped 9
    EXPECT_EQ(s.count, 5u);
    EXPECT_DOUBLE_EQ(s.mean(), 18.0 / 5.0);

    h.record(3);
    const auto d = h.snapshot() - s;
    EXPECT_EQ(d.count, 1u);
    EXPECT_EQ(d.counts[3], 1u);
}
