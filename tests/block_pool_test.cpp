#include "infer/memory/block_pool.hpp"

#include <gtest/gtest.h>

#include <atomic>
#include <cstdint>
#include <set>
#include <thread>
#include <vector>

using infer::BlockPool;

TEST(BlockPool, HandsOutDistinctAlignedBlocksUntilExhausted) {
    BlockPool pool(/*block_floats=*/100, /*num_blocks=*/4);
    std::vector<BlockPool::Block> held;
    std::set<float*> ptrs;
    for (int i = 0; i < 4; ++i) {
        BlockPool::Block b = pool.try_acquire();
        ASSERT_TRUE(b);
        EXPECT_EQ(b.size(), 100u);
        EXPECT_EQ(reinterpret_cast<std::uintptr_t>(b.data()) % 64, 0u) << "blocks must be cache-line aligned";
        ptrs.insert(b.data());
        held.push_back(std::move(b));
    }
    EXPECT_EQ(ptrs.size(), 4u);
    EXPECT_EQ(pool.in_use(), 4u);
    EXPECT_FALSE(pool.try_acquire()) << "pool should be exhausted";

    held.pop_back();  // RAII release
    EXPECT_EQ(pool.in_use(), 3u);
    EXPECT_TRUE(pool.try_acquire());
    EXPECT_EQ(pool.high_watermark(), 4u);
}

TEST(BlockPool, BlocksDoNotOverlap) {
    BlockPool pool(/*block_floats=*/33, /*num_blocks=*/3);  // stride rounds up to 48 floats
    auto a = pool.try_acquire();
    auto b = pool.try_acquire();
    auto c = pool.try_acquire();
    for (std::size_t i = 0; i < 33; ++i) {
        a.data()[i] = 1;
        b.data()[i] = 2;
        c.data()[i] = 3;
    }
    for (std::size_t i = 0; i < 33; ++i) {
        ASSERT_EQ(a.data()[i], 1);
        ASSERT_EQ(b.data()[i], 2);
        ASSERT_EQ(c.data()[i], 3);
    }
}

TEST(BlockPool, LifoReusesMostRecentlyReleasedBlock) {
    BlockPool pool(16, 8);
    auto a = pool.try_acquire();
    float* p = a.data();
    a.reset();
    EXPECT_EQ(pool.try_acquire().data(), p);
}

TEST(BlockPool, MoveTransfersOwnership) {
    BlockPool pool(16, 1);
    auto a = pool.try_acquire();
    BlockPool::Block b = std::move(a);
    EXPECT_FALSE(a);  // NOLINT(bugprone-use-after-move)
    EXPECT_TRUE(b);
    EXPECT_EQ(pool.in_use(), 1u);
    a = std::move(b);
    EXPECT_EQ(pool.in_use(), 1u);
    a = BlockPool::Block{};
    EXPECT_EQ(pool.in_use(), 0u);
}

TEST(BlockPool, CustomAllocatorIsUsedAndFreed) {
    static int allocs = 0, frees = 0;
    {
        BlockPool pool(
            16, 2,
            [](std::size_t bytes) {
                ++allocs;
                return ::operator new(bytes, std::align_val_t{64});
            },
            [](void* p) {
                ++frees;
                ::operator delete(p, std::align_val_t{64});
            });
        EXPECT_TRUE(pool.try_acquire());
    }
    EXPECT_EQ(allocs, 1);
    EXPECT_EQ(frees, 1);
}

// Many threads acquire, write a thread-unique pattern, verify nobody else
// touched the block, and release. A broken free list (ABA, lost update) shows
// up as two threads holding the same block, or as blocks leaking.
TEST(BlockPool, ConcurrentAcquireReleaseIsExclusive) {
    constexpr int kThreads = 8;
    constexpr int kIters = 20000;
    constexpr std::size_t kBlockFloats = 8;
    BlockPool pool(kBlockFloats, /*num_blocks=*/4);  // fewer blocks than threads: heavy contention

    std::atomic<int> collisions{0};
    std::atomic<int> acquired{0};
    std::vector<std::thread> threads;
    for (int t = 0; t < kThreads; ++t) {
        threads.emplace_back([&, t] {
            const float tag = static_cast<float>(t + 1);
            for (int i = 0; i < kIters; ++i) {
                BlockPool::Block b = pool.try_acquire();
                if (!b) continue;
                acquired.fetch_add(1, std::memory_order_relaxed);
                for (std::size_t k = 0; k < kBlockFloats; ++k) b.data()[k] = tag;
                for (std::size_t k = 0; k < kBlockFloats; ++k) {
                    if (b.data()[k] != tag) {
                        collisions.fetch_add(1, std::memory_order_relaxed);
                        break;
                    }
                }
            }
        });
    }
    for (auto& th : threads) th.join();

    EXPECT_EQ(collisions.load(), 0);
    EXPECT_GT(acquired.load(), 0);
    EXPECT_EQ(pool.in_use(), 0u);
    // Every block must be back on the free list.
    std::vector<BlockPool::Block> all;
    for (int i = 0; i < 4; ++i) {
        all.push_back(pool.try_acquire());
        EXPECT_TRUE(all.back());
    }
    EXPECT_FALSE(pool.try_acquire());
}
