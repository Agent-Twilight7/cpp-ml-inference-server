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
};

// ModelBackend over an ONNX model with exactly one float input and one float
// output, where dimension 0 of both is the (dynamic) batch dimension.
class OnnxBackend final : public ModelBackend {
public:
    explicit OnnxBackend(const OnnxBackendOptions& opts);

    std::size_t input_elems_per_sample() const override { return input_elems_; }
    std::size_t output_elems_per_sample() const override { return output_elems_; }

    void run(std::span<const float> input, std::span<float> output, std::size_t batch) override;

    // Per-sample shapes, without the batch dimension.
    const std::vector<int64_t>& input_sample_shape() const { return input_shape_; }
    const std::vector<int64_t>& output_sample_shape() const { return output_shape_; }

private:
    Ort::Session session_{nullptr};
    Ort::MemoryInfo mem_info_{nullptr};
    std::string input_name_;
    std::string output_name_;
    std::vector<int64_t> input_shape_;   // without batch dim
    std::vector<int64_t> output_shape_;  // without batch dim
    std::size_t input_elems_ = 0;
    std::size_t output_elems_ = 0;
};

}  // namespace infer
