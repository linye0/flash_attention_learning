#include <torch/extension.h>
#include "attention.h"

torch::Tensor run_v4_flash_attention(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
    // 1. 获取维度信息
    int N = Q.size(0); // 假设 shape 是 [N, d]
    int d = Q.size(1);

    // 2. 让 PyTorch 自动分配输出显存 O (自动在 GPU 上，不用写 cudaMalloc)
    auto O = torch::empty_like(Q);

    // 3. 抠出裸指针并强转为 half*
    const void* q_ptr = Q.data_ptr<at::Half>();
    const void* k_ptr = K.data_ptr<at::Half>();
    const void* v_ptr = V.data_ptr<at::Half>();
    void* o_ptr = O.data_ptr<at::Half>();

    // 4. 直接调用你手写的内核！
    // 注：由于你的 V4 是纯 CUDA 算子不依赖 cuBLAS，handle 传 nullptr 即可
    launch_v4_flash_wmma(nullptr, q_ptr, k_ptr, v_ptr, o_ptr, nullptr, nullptr, N, d);

    return O;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run_v4", &run_v4_flash_attention, "V4 Flash Attention using WMMA");
}