#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>

namespace infer {

// A fixed number of equally sized float buffers carved out of one slab that is
// allocated up front. try_acquire() and release are lock-free: the free list is
// a Treiber stack whose head packs a 32-bit block index with a 32-bit version
// tag, so a block popped and pushed back between another thread's load and CAS
// (the ABA problem) changes the tag and makes that CAS fail.
//
// The free list is LIFO, so the most recently released (cache- and TLB-warm)
// block is handed out next, and untouched blocks never get physical pages.
class BlockPool {
public:
    using Allocator = std::function<void*(std::size_t bytes)>;
    using Deallocator = std::function<void(void*)>;

    // Move-only RAII handle; returns its block to the pool when destroyed.
    class Block {
    public:
        Block() = default;
        Block(Block&& other) noexcept { *this = std::move(other); }
        Block& operator=(Block&& other) noexcept;
        Block(const Block&) = delete;
        Block& operator=(const Block&) = delete;
        ~Block() { reset(); }

        explicit operator bool() const { return pool_ != nullptr; }
        float* data() const;
        std::size_t size() const;
        std::span<float> span() const { return {data(), size()}; }
        void reset();  // return the block early

    private:
        friend class BlockPool;
        Block(BlockPool* pool, std::uint32_t index) : pool_(pool), index_(index) {}

        BlockPool* pool_ = nullptr;
        std::uint32_t index_ = 0;
    };

    // Blocks in 64-byte-aligned heap memory.
    BlockPool(std::size_t block_floats, std::size_t num_blocks);
    // Blocks in memory from a custom allocator (e.g. pinned host memory).
    BlockPool(std::size_t block_floats, std::size_t num_blocks, const Allocator& alloc, Deallocator dealloc);
    ~BlockPool();

    BlockPool(const BlockPool&) = delete;
    BlockPool& operator=(const BlockPool&) = delete;

    // Returns an empty Block if every block is in use.
    Block try_acquire();

    std::size_t block_floats() const { return block_floats_; }
    std::size_t capacity() const { return capacity_; }
    std::size_t in_use() const { return in_use_.load(std::memory_order_relaxed); }
    std::size_t high_watermark() const { return high_water_.load(std::memory_order_relaxed); }

private:
    static constexpr std::uint32_t kNil = UINT32_MAX;

    void release(std::uint32_t index);
    float* block_ptr(std::uint32_t index) const { return slab_ + static_cast<std::size_t>(index) * stride_; }

    std::size_t block_floats_;
    std::size_t stride_;  // block_floats rounded up to a 64-byte multiple
    std::size_t capacity_;
    float* slab_ = nullptr;
    Deallocator dealloc_;

    std::atomic<std::uint64_t> head_;                    // (tag << 32) | index
    std::unique_ptr<std::atomic<std::uint32_t>[]> next_;  // free-list links
    std::atomic<std::size_t> in_use_{0};
    std::atomic<std::size_t> high_water_{0};
};

}  // namespace infer
