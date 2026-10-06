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

    // Runs `batch` samples. `input` holds batch * input_elems_per_sample()
    // floats, `output` must have room for batch * output_elems_per_sample().
    // Not required to be thread-safe: use one backend instance per worker.
    virtual void run(std::span<const float> input, std::span<float> output, std::size_t batch) = 0;
};

}  // namespace infer
