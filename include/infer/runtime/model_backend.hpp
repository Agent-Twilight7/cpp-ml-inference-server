#pragma once

#include <cstddef>
#include <span>

namespace infer {

// A model that consumes a batch of fixed-size float samples and produces a
// batch of fixed-size float outputs. Buffers are caller-owned so the server can
// hand in preallocated memory from its pools; backends must not allocate
// per-call storage for inputs or outputs.
class ModelBackend {
public:
    virtual ~ModelBackend() = default;

    // Number of floats in one input sample (e.g. 3*224*224).
    virtual std::size_t input_elems_per_sample() const = 0;
    // Number of floats in one output sample (e.g. 1000 logits).
    virtual std::size_t output_elems_per_sample() const = 0;

    // The batch size run() should be called with to process `n` real samples.
    // Backends that only support certain sizes round up; the caller fills the
    // extra rows with anything and ignores their outputs.
    virtual std::size_t padded_batch_size(std::size_t n) const { return n; }

    // Runs `batch` samples, where batch is a value padded_batch_size() can
    // return. `input` holds batch * input_elems_per_sample() floats, `output`
    // must have room for batch * output_elems_per_sample().
    // Not required to be thread-safe: use one backend instance per worker.
    virtual void run(std::span<const float> input, std::span<float> output, std::size_t batch) = 0;
};

}  // namespace infer
