#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <math.h>
#include <math_constants.h>
#include "attention.h"

// =========================================================
// 你的战场：手写 V0 版本的 Softmax
// =========================================================
__global__ void softmax_v0_kernel(float* S, float* P, int N) {
    // 提示：这是一个标准的全局内存 Softmax
    // 建议：让 1 个 Thread 负责处理 S 矩阵的 1 行数据
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    float maxval = -CUDART_INF_F;
    float sum = 0.0;
    
    if (row < N) {
        // TODO 1: 遍历该行，找到最大值 max_val
        for (int i = 0; i < N; ++i) {
            int idx = row * N + i; 
            maxval = max(maxval, S[idx]);
        }
        
        // TODO 2: 再次遍历该行，计算 sum = exp(S[row, i] - max_val) 的总和
        for (int i = 0; i < N; ++i) {
            int idx = row * N + i;
            sum += exp(S[idx] - maxval);
        }
        
        // TODO 3: 第三次遍历该行，计算 P[row, i] = exp(S[row, i] - max_val) / sum
        for (int i = 0; i < N; ++i) {
            int idx = row * N + i;
            P[idx] = exp(S[idx] - maxval) / sum;
        }
    }
}

// =========================================================
// Launcher: V0 Multi-pass
// =========================================================
void launch_v0_cublas(cublasHandle_t handle, const float* Q, const float* K, const float* V, float* O, float* S, float* P, int N, int d) {
    float alpha = 1.0f / sqrtf((float)d);
    float beta = 0.0f;

    // 1. S = (1/sqrt(d)) * Q * K^T
    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, N, N, d, &alpha, K, d, Q, d, &beta, S, N);

    // 2. P = Softmax(S)
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    softmax_v0_kernel<<<blocks, threads>>>(S, P, N);
    
    // 3. O = P * V
    float alpha_1 = 1.0f;
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, d, N, N, &alpha_1, V, d, P, N, &beta, O, d);
}

// 注册列表
std::vector<KernelInfo> get_kernels() {
    std::vector<KernelInfo> kernels;
    kernels.push_back({launch_v0_cublas, "00_V0_Multipass", true});
    // 未来在这里添加 V1, V2...
    // kernels.push_back({launch_v1_naive, "01_V1_Naive_Tiled"});
    return kernels;
}