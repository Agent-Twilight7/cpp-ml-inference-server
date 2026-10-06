#include "infer/memory/block_pool.hpp"

#include <cstdio>
#include <new>
#include <stdexcept>

namespace infer {
namespace {

constexpr std::size_t kAlign = 64;  // cache line
constexpr std::size_t kFloatsPerLine = kAlign / sizeof(float);

constexpr std::uint64_t pack(std::uint32_t tag, std::uint32_t index) {
    return (static_cast<std::uint64_t>(tag) << 32) | index;
}
constexpr std::uint32_t index_of(std::uint64_t head) { return static_cast<std::uint32_t>(head); }
constexpr std::uint32_t tag_of(std::uint64_t head) { return static_cast<std::uint32_t>(head >> 32); }

}  // namespace

// ---- Block ----

BlockPool::Block& BlockPool::Block::operator=(Block&& other) noexcept {
    if (this != &other) {
        reset();
        pool_ = other.pool_;
        index_ = other.index_;
        other.pool_ = nullptr;
    }
    return *this;
}

float* BlockPool::Block::data() const { return pool_ ? pool_->block_ptr(index_) : nullptr; }

std::size_t BlockPool::Block::size() const { return pool_ ? pool_->block_floats_ : 0; }

void BlockPool::Block::reset() {
    if (pool_) {
        pool_->release(index_);
        pool_ = nullptr;
    }
}

// ---- BlockPool ----

BlockPool::BlockPool(std::size_t block_floats, std::size_t num_blocks)
    : BlockPool(
          block_floats, num_blocks,
          [](std::size_t bytes) { return ::operator new(bytes, std::align_val_t{kAlign}); },
          [](void* p) { ::operator delete(p, std::align_val_t{kAlign}); }) {}

BlockPool::BlockPool(std::size_t block_floats, std::size_t num_blocks, const Allocator& alloc, Deallocator dealloc)
    : block_floats_(block_floats),
      stride_((block_floats + kFloatsPerLine - 1) / kFloatsPerLine * kFloatsPerLine),
      capacity_(num_blocks),
      dealloc_(std::move(dealloc)) {
    if (block_floats == 0 || num_blocks == 0) throw std::invalid_argument("BlockPool: sizes must be > 0");
    if (num_blocks >= kNil) throw std::invalid_argument("BlockPool: too many blocks");

    slab_ = static_cast<float*>(alloc(stride_ * capacity_ * sizeof(float)));
    if (!slab_) throw std::bad_alloc();

    // Initial free list: 0 -> 1 -> ... -> n-1 -> nil.
    next_ = std::make_unique<std::atomic<std::uint32_t>[]>(capacity_);
    for (std::size_t i = 0; i < capacity_; ++i) {
        next_[i].store(i + 1 < capacity_ ? static_cast<std::uint32_t>(i + 1) : kNil, std::memory_order_relaxed);
    }
    head_.store(pack(0, 0), std::memory_order_release);
}

BlockPool::~BlockPool() {
    if (in_use() != 0) {
        // A Block outliving its pool would be a use-after-free; make it loud.
        std::fprintf(stderr, "BlockPool destroyed with %zu blocks still in use\n", in_use());
    }
    dealloc_(slab_);
}

BlockPool::Block BlockPool::try_acquire() {
    std::uint64_t old = head_.load(std::memory_order_acquire);
    for (;;) {
        const std::uint32_t index = index_of(old);
        if (index == kNil) return Block{};
        // If another thread pops `index` first, this read may be stale, but
        // then the head's tag has changed and the CAS below fails.
        const std::uint32_t next = next_[index].load(std::memory_order_relaxed);
        if (head_.compare_exchange_weak(old, pack(tag_of(old) + 1, next), std::memory_order_acquire,
                                        std::memory_order_acquire)) {
            const std::size_t now = in_use_.fetch_add(1, std::memory_order_relaxed) + 1;
            std::size_t hw = high_water_.load(std::memory_order_relaxed);
            while (now > hw && !high_water_.compare_exchange_weak(hw, now, std::memory_order_relaxed)) {
            }
            return Block(this, index);
        }
    }
}

void BlockPool::release(std::uint32_t index) {
    in_use_.fetch_sub(1, std::memory_order_relaxed);
    std::uint64_t old = head_.load(std::memory_order_relaxed);
    for (;;) {
        next_[index].store(index_of(old), std::memory_order_relaxed);
        // Release: the link above and the caller's writes to the block become
        // visible to whichever thread acquires this block next.
        if (head_.compare_exchange_weak(old, pack(tag_of(old) + 1, index), std::memory_order_release,
                                        std::memory_order_relaxed)) {
            return;
        }
    }
}

}  // namespace infer
