#ifndef ATTENTION_H
#define ATTENTION_H
#include <cublas_v2.h>
#include <string>
#include <vector>

// 统一的 Attention 函数接口
// Q, K, V 形状: [N, d]
// O 形状: [N, d]
// S, P 是仅 V0 需要的中间矩阵，形状: [N, N]
typedef void (*AttnFunc)(cublasHandle_t, const void*, const void*, const void*, void*, float*, float*, int, int);
void cpu_attention_reference_partial(const float* Q, const float* K, const float* V, float* O, int N, int d, int num_rows);

struct KernelInfo {
    AttnFunc func;
    std::string name;
    bool is_multipass = false; // 是否需要全部传递S和P
    bool is_halfacc = false; // 是否是半精度算法
    bool is_decoding = false;
};

std::vector<KernelInfo> get_kernels();

void launch_v4_flash_wmma(cublasHandle_t handle, const void* Q_ptr, const void* K_ptr, const void* V_ptr, void* O_ptr, float* S, float* P, int N, int d);

#endif