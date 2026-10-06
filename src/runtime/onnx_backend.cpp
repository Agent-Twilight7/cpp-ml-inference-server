#include "infer/runtime/onnx_backend.hpp"

#include <stdexcept>

namespace infer {
namespace {

// ORT allows exactly one Env per process; every session shares it.
Ort::Env& ort_env() {
    static Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "infer"};
    return env;
}

// Splits a model shape into (batch dim, per-sample dims) and validates it.
std::vector<int64_t> sample_shape(const std::vector<int64_t>& full, const char* what) {
    if (full.size() < 2) {
        throw std::runtime_error(std::string(what) + " must have a batch dim plus at least one more dim");
    }
    std::vector<int64_t> sample(full.begin() + 1, full.end());
    for (int64_t d : sample) {
        if (d <= 0) {
            throw std::runtime_error(std::string(what) + " has a dynamic non-batch dim; only dim 0 may be dynamic");
        }
    }
    return sample;
}

std::size_t elem_count(const std::vector<int64_t>& shape) {
    std::size_t n = 1;
    for (int64_t d : shape) n *= static_cast<std::size_t>(d);
    return n;
}

}  // namespace

OnnxBackend::OnnxBackend(const OnnxBackendOptions& opts) {
    Ort::SessionOptions so;
    so.SetIntraOpNumThreads(opts.intra_op_threads);
    so.SetInterOpNumThreads(1);
    so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    session_ = Ort::Session(ort_env(), opts.model_path.c_str(), so);
    mem_info_ = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    if (session_.GetInputCount() != 1 || session_.GetOutputCount() != 1) {
        throw std::runtime_error("model must have exactly one input and one output");
    }

    Ort::AllocatorWithDefaultOptions alloc;
    input_name_ = session_.GetInputNameAllocated(0, alloc).get();
    output_name_ = session_.GetOutputNameAllocated(0, alloc).get();

    // The shape infos are non-owning views into the TypeInfo objects, so those
    // must stay alive while we read them.
    Ort::TypeInfo in_type = session_.GetInputTypeInfo(0);
    Ort::TypeInfo out_type = session_.GetOutputTypeInfo(0);
    auto in_info = in_type.GetTensorTypeAndShapeInfo();
    auto out_info = out_type.GetTensorTypeAndShapeInfo();
    if (in_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
        out_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        throw std::runtime_error("model input and output must be float32");
    }

    input_shape_ = sample_shape(in_info.GetShape(), "input");
    output_shape_ = sample_shape(out_info.GetShape(), "output");
    input_elems_ = elem_count(input_shape_);
    output_elems_ = elem_count(output_shape_);
}

void OnnxBackend::run(std::span<const float> input, std::span<float> output, std::size_t batch) {
    if (batch == 0) return;
    if (input.size() < batch * input_elems_ || output.size() < batch * output_elems_) {
        throw std::invalid_argument("OnnxBackend::run: buffer too small for batch");
    }

    std::vector<int64_t> in_shape{static_cast<int64_t>(batch)};
    in_shape.insert(in_shape.end(), input_shape_.begin(), input_shape_.end());
    std::vector<int64_t> out_shape{static_cast<int64_t>(batch)};
    out_shape.insert(out_shape.end(), output_shape_.begin(), output_shape_.end());

    // Wrap caller memory directly: no copies, no allocations for tensor data.
    // ORT's API takes a non-const pointer but does not write to inputs.
    Ort::Value in_tensor = Ort::Value::CreateTensor<float>(
        mem_info_, const_cast<float*>(input.data()), batch * input_elems_, in_shape.data(), in_shape.size());
    Ort::Value out_tensor = Ort::Value::CreateTensor<float>(
        mem_info_, output.data(), batch * output_elems_, out_shape.data(), out_shape.size());

    const char* in_names[] = {input_name_.c_str()};
    const char* out_names[] = {output_name_.c_str()};
    session_.Run(Ort::RunOptions{nullptr}, in_names, &in_tensor, 1, out_names, &out_tensor, 1);
}

}  // namespace infer
