#pragma once

#include "infer/runtime/model_backend.hpp"

#include <onnxruntime_cxx_api.h>

#include <cstdint>
#include <string>
#include <vector>

namespace infer {

enum class Device { kCpu, kCuda };

// Parses "cpu" / "cuda"; throws std::invalid_argument otherwise.
Device parse_device(const std::string& s);
const char* to_string(Device d);

struct OnnxBackendOptions {
    std::string model_path;
    Device device = Device::kCpu;
    int cuda_device_id = 0;
    int intra_op_threads = 1;  // CPU threads ORT may use inside a single Run()
    // Empty: one session that accepts any batch size. Otherwise one session
    // per listed size, and batches are padded up to the next size. This
    // matters on CUDA, where ORT redoes per-layer setup (~30 ms for ResNet-18)
    // every time a session sees a different input shape than its last run.
    std::vector<std::size_t> batch_buckets{};
    // On CUDA, give batch buffers page-locked (pinned) host memory.
    bool pinned_host_memory = true;
};

// Powers of two up to max_batch, plus max_batch itself: 1, 2, 4, ..., max.
std::vector<std::size_t> power_of_two_buckets(std::size_t max_batch);

// ModelBackend over an ONNX model with exactly one float input and one float
// output, where dimension 0 of both is the (dynamic) batch dimension.
class OnnxBackend final : public ModelBackend {
public:
    explicit OnnxBackend(const OnnxBackendOptions& opts);

    std::size_t input_elems_per_sample() const override { return input_elems_; }
    std::size_t output_elems_per_sample() const override { return output_elems_; }
    std::size_t padded_batch_size(std::size_t n) const override;
    // On CUDA: page-locked host memory from ORT's CudaPinned allocator, so
    // host<->device copies are direct DMA. Falls back to aligned heap memory.
    HostBuffer allocate_host_buffer(std::size_t floats) override;

    void run(std::span<const float> input, std::span<float> output, std::size_t batch) override;

    // Per-sample shapes, without the batch dimension.
    const std::vector<int64_t>& input_sample_shape() const { return input_shape_; }
    const std::vector<int64_t>& output_sample_shape() const { return output_shape_; }

private:
    Ort::Session& session_for(std::size_t batch);

    // sessions_[i] serves batches of exactly buckets_[i]; with no buckets
    // there is a single session for every size.
    Device device_ = Device::kCpu;
    int cuda_device_id_ = 0;
    bool pinned_ = true;
    std::vector<std::size_t> buckets_;
    std::vector<Ort::Session> sessions_;
    Ort::MemoryInfo mem_info_{nullptr};
    std::string input_name_;
    std::string output_name_;
    std::vector<int64_t> input_shape_;   // without batch dim
    std::vector<int64_t> output_shape_;  // without batch dim
    std::size_t input_elems_ = 0;
    std::size_t output_elems_ = 0;
};

}  // namespace infer
