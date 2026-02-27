#ifndef ATTENTION_H
#define ATTENTION_H
#include <cublas_v2.h>
#include <string>
#include <vector>

// 统一的 Attention 函数接口
// Q, K, V 形状: [N, d]
// O 形状: [N, d]
// S, P 是仅 V0 需要的中间矩阵，形状: [N, N]
typedef void (*AttnFunc)(cublasHandle_t, const float*, const float*, const float*, float*, float*, float*, int, int);
void cpu_attention_reference_partial(const float* Q, const float* K, const float* V, float* O, int N, int d, int num_rows);

struct KernelInfo {
    AttnFunc func;
    std::string name;
    bool is_multipass; // 新增：如果是 V0 则为 true，V1+ 为 false
};

std::vector<KernelInfo> get_kernels();
void launch_v0_cublas(cublasHandle_t handle, const float* Q, const float* K, const float* V, float* O, float* S, float* P, int N, int d);

#endif