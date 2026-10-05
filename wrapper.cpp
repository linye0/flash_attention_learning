#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

#include "attention.h"

namespace {

void check_inputs(const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v) {
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(), "Q, K and V must be CUDA tensors");
    TORCH_CHECK(q.device() == k.device() && q.device() == v.device(), "Q, K and V must use the same device");
    TORCH_CHECK(q.scalar_type() == at::kHalf && k.scalar_type() == at::kHalf && v.scalar_type() == at::kHalf,
                "Q, K and V must have dtype torch.float16");
    TORCH_CHECK(q.dim() == 2 && k.dim() == 2 && v.dim() == 2, "Q, K and V must have shape [N, D]");
    TORCH_CHECK(q.sizes() == k.sizes() && q.sizes() == v.sizes(), "Q, K and V must have identical shapes");
    TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous(), "Q, K and V must be contiguous");
    TORCH_CHECK(q.size(1) == 64, "this kernel currently supports head dimension D=64 only");
    TORCH_CHECK(q.size(0) % 64 == 0, "sequence length N must be a multiple of 64");
}

torch::Tensor run_v4(torch::Tensor q, torch::Tensor k, torch::Tensor v) {
    check_inputs(q, k, v);
    c10::cuda::CUDAGuard device_guard(q.device());
    auto output = torch::empty_like(q);
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream(q.get_device());
    launch_v4_flash_wmma_stream(
        q.data_ptr<at::Half>(), k.data_ptr<at::Half>(), v.data_ptr<at::Half>(),
        output.data_ptr<at::Half>(), static_cast<int>(q.size(0)),
        static_cast<int>(q.size(1)), stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

torch::Tensor run_v5(torch::Tensor q, torch::Tensor k, torch::Tensor v) {
    check_inputs(q, k, v);
    c10::cuda::CUDAGuard device_guard(q.device());
    auto output = torch::empty_like(q);
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream(q.get_device());
    launch_v5_flash_fa2_wmma_stream(
        q.data_ptr<at::Half>(), k.data_ptr<at::Half>(), v.data_ptr<at::Half>(),
        output.data_ptr<at::Half>(), static_cast<int>(q.size(0)),
        static_cast<int>(q.size(1)), stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("run_v4", &run_v4, "FlashAttention V4 forward (CUDA, FP16)");
    module.def("run_v5", &run_v5, "FlashAttention V5 FA2-style forward (CUDA, FP16)");
}
