#include "infer/runtime/onnx_backend.hpp"

#include <algorithm>
#include <stdexcept>

namespace infer {
namespace {

// ORT allows exactly one Env per process; every session shares it. It is
// deliberately never destroyed: tearing it down during static destruction can
// run after the CUDA runtime has already unloaded and abort the process at exit.
Ort::Env& ort_env() {
    static Ort::Env* env = new Ort::Env{ORT_LOGGING_LEVEL_WARNING, "infer"};
    return *env;
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

Device parse_device(const std::string& s) {
    if (s == "cpu") return Device::kCpu;
    if (s == "cuda") return Device::kCuda;
    throw std::invalid_argument("unknown device '" + s + "' (expected cpu or cuda)");
}

const char* to_string(Device d) { return d == Device::kCuda ? "cuda" : "cpu"; }

OnnxBackend::OnnxBackend(const OnnxBackendOptions& opts) {
    // The Env must exist before any SessionOptions call that touches ORT's
    // logger (e.g. appending the CUDA provider), so create it first.
    Ort::Env& env = ort_env();

    Ort::SessionOptions so;
    so.SetIntraOpNumThreads(opts.intra_op_threads);
    so.SetInterOpNumThreads(1);
    so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    if (opts.device == Device::kCuda) {
#ifdef INFER_HAS_CUDA
        // Inputs/outputs stay in host memory; ORT copies them to and from the
        // GPU around each Run(). Nodes CUDA can't run fall back to CPU.
        OrtCUDAProviderOptions cuda{};
        cuda.device_id = opts.cuda_device_id;
        cuda.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchHeuristic;
        so.AppendExecutionProvider_CUDA(cuda);
#else
        throw std::runtime_error("built without CUDA support; reconfigure with -DINFER_ENABLE_CUDA=ON");
#endif
    }

    buckets_ = opts.batch_buckets;
    std::sort(buckets_.begin(), buckets_.end());
    buckets_.erase(std::unique(buckets_.begin(), buckets_.end()), buckets_.end());
    if (!buckets_.empty() && buckets_.front() == 0) throw std::invalid_argument("batch bucket sizes must be >= 1");

    const std::size_t num_sessions = buckets_.empty() ? 1 : buckets_.size();
    for (std::size_t i = 0; i < num_sessions; ++i) {
        sessions_.emplace_back(env, opts.model_path.c_str(), so);
    }
    mem_info_ = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    Ort::Session& meta = sessions_.front();
    if (meta.GetInputCount() != 1 || meta.GetOutputCount() != 1) {
        throw std::runtime_error("model must have exactly one input and one output");
    }

    Ort::AllocatorWithDefaultOptions alloc;
    input_name_ = meta.GetInputNameAllocated(0, alloc).get();
    output_name_ = meta.GetOutputNameAllocated(0, alloc).get();

    // The shape infos are non-owning views into the TypeInfo objects, so those
    // must stay alive while we read them.
    Ort::TypeInfo in_type = meta.GetInputTypeInfo(0);
    Ort::TypeInfo out_type = meta.GetOutputTypeInfo(0);
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

std::size_t OnnxBackend::padded_batch_size(std::size_t n) const {
    if (buckets_.empty()) return n;
    auto it = std::lower_bound(buckets_.begin(), buckets_.end(), n);
    if (it == buckets_.end()) throw std::invalid_argument("batch larger than the biggest bucket");
    return *it;
}

Ort::Session& OnnxBackend::session_for(std::size_t batch) {
    if (buckets_.empty()) return sessions_.front();
    auto it = std::lower_bound(buckets_.begin(), buckets_.end(), batch);
    if (it == buckets_.end() || *it != batch) {
        throw std::invalid_argument("batch size " + std::to_string(batch) + " is not a bucket; use padded_batch_size()");
    }
    return sessions_[static_cast<std::size_t>(it - buckets_.begin())];
}

std::vector<std::size_t> power_of_two_buckets(std::size_t max_batch) {
    std::vector<std::size_t> out;
    for (std::size_t b = 1; b < max_batch; b *= 2) out.push_back(b);
    out.push_back(max_batch);
    return out;
}

void OnnxBackend::run(std::span<const float> input, std::span<float> output, std::size_t batch) {
    if (batch == 0) return;
    Ort::Session& session = session_for(batch);
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
    session.Run(Ort::RunOptions{nullptr}, in_names, &in_tensor, 1, out_names, &out_tensor, 1);
}

}  // namespace infer
