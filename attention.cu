#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <math.h>
#include <math_constants.h>
#include "attention.h"

// =========================================================
// 你的战场：手写 V0 版本的 Softmax
// =========================================================
__global__ void softmax_v0_kernel(float* S, float* P, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N) return;

    // 必须使用 size_t，并且强制转换 row 为 size_t 防止计算中途溢出
    size_t row_offset = (size_t)row * N; 
    float maxval = -CUDART_INF_F;
    float sum = 0.0;

    for (int i = 0; i < N; ++i) {
        maxval = max(maxval, S[row_offset + i]);
    }
    for (int i = 0; i < N; ++i) {
        sum += exp(S[row_offset + i] - maxval);
    }
    for (int i = 0; i < N; ++i) {
        P[row_offset + i] = exp(S[row_offset + i] - maxval) / sum;
    }
}

__global__ void flash_attn_v1_kernel(
    const float* Q, const float* K, const float* V, float* O,
    int N, int d, int Tc, int Tr, int Bc, int Br, float scale
) {
    extern __shared__ float sram[];
    float* s_Q = sram;
    float* s_K = s_Q + Br * d;
    float* s_V = s_K + Bc * d;

    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int row_idx = bx * Br + tx;

    if (row_idx >= N) return;

    float m_i = -CUDART_INF_F;
    float l_i = 0.0f;

    float o_reg[64];
    #pragma unroll
    for (int k = 0; k < d; ++k) o_reg[k] = 0.0f;

    for (int i = tx; i < Br * d; i += blockDim.x) {
        s_Q[i] = Q[bx * Br * d + i];
    }

    __syncthreads();

    for (int j = 0; j < Tc; ++j) {
        for (int k = tx * 4; k < Bc * d; k += blockDim.x * 4) {
            // 这边的K是还没有进行转置的
            *(float4*)(&s_K[k]) = *(float4*)(&K[j * Bc * d + k]);
            *(float4*)(&s_V[k]) = *(float4*)(&V[j * Bc * d + k]);
        }
        __syncthreads();

        #pragma unroll
        for (int curr_bc = 0; curr_bc < Bc; ++curr_bc) {
            float sum = 0.0f;
            #pragma unroll
            for (int k = 0; k < d; ++k) {
                sum += s_Q[tx * d + k] * s_K[curr_bc * d + k];
            }
            sum *= scale;

            float m_prev = m_i;
            m_i = max(m_prev, sum);

            float alpha = expf(m_prev - m_i);
            float beta = expf(sum - m_i);

            l_i = l_i * alpha + beta;

            // 重放缩旧的 O 并加上新的 PV 贡献
            for (int k = 0; k < d; ++k) {
                o_reg[k] = o_reg[k] * alpha + beta * s_V[curr_bc * d + k];
            }
        }
        __syncthreads(); // 准备进入下一个 KV 块
    }

    // 6. 最终归一化并写回 HBM
    for (int k = 0; k < d; ++k) {
        O[row_idx * d + k] = o_reg[k] / l_i;
    }
}

__global__ void flash_attn_v2_kernel(
    const float* Q, const float* K, const float* V, float* O,
    int N, int d, int Tc, int Tr, int Bc, int Br, float scale
) {
    extern __shared__ float sram[];
    float* s_Q = sram;
    float* s_K = s_Q + Br * d;
    float* s_V = s_K + Bc * d;

    int tx = threadIdx.x;
    int bx = blockIdx.x;
    // 这个线程所负责的行
    int wid = tx / 32;
    int row_idx = bx * Br + tid / 4;
}

// =========================================================
// Launchers
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

void launch_v1_flash_tiling(cublasHandle_t handle, const float* Q, const float* K, const float* V, float* O, float* S, float* P, int N, int d) {
    // 从v1开始，我们采用online计算中间矩阵的方式，所以传入的handle，S，P都是无用的变量
    const int Br = 32;
    const int Bc = 32;
    
    const int Tr = (N + Br - 1) / Br;
    const int Tc = (N + Bc - 1) / Bc;
    const float scale = 1.0f / sqrtf(d);

    dim3 grid(Tr);
    dim3 block(Br);

    size_t shared_mem_size = (Br * d + 2 * Bc * d) * sizeof(float);

    // 检查SRAM限制是否超过最大值(RTX 3060为48KB)
    if (shared_mem_size > 48 * 1024) {
        fprintf(stderr, "Error: Shared memory request (%zu bytes) exceeds 48KB limit!\n", shared_mem_size);
        exit(1);
    }

    flash_attn_v1_kernel<<<grid, block, shared_mem_size>>>(Q, K, V, O, N, d, Tc, Tr, Bc, Br, scale);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("FlashAttn V1 Launch Error: %s\n", cudaGetErrorString(err));
    }
}

void launch_v2_flash_vectorized(cublasHandle_t handle, const float* Q, const float* K, const float* V, float* O, float* S, float* P, int N, int d) {
    const int Br = 32;
    const int Bc = 32;

    const int Tr = (N + Br - 1) / Br;
    const int Tc = (N + Bc - 1) / Bc;
    const float scale = 1.0f / sqrtf(d);

    dim3 grid(Tr);
    // 一个block内改为128个线程，一共4个warp，负责S矩阵内的32*32范围的数据
    dim3 block(128);

    size_t shared_mem_size = (Br * d + 2 * Bc * d) * sizeof(float);
    
    if (shared_mem_size > 48 * 1024) {
        fprintf(stderr, "Error: Shared memory request (%zu bytes) exceeds 48KB limit!\n", shared_mem_size);
        exit(1);
    }

    flash_attn_v2_kernel<<<grid, block, shared_mem_size>>>(Q, K, V, O, N, d, Tc, Tr, Bc, Br, scale);

    cudaError err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("FlashAttn V2 Launch Error: %s\n", cudaGetErrorString(err));
    }
}

// 注册列表
std::vector<KernelInfo> get_kernels() {
    std::vector<KernelInfo> kernels;
    kernels.push_back({launch_v0_cublas, "V0_Multipass", true});
    kernels.push_back({launch_v1_flash_tiling, "V1_flash_tiling", false});
    kernels.push_back({launch_v2_flash_vectorized, "V2_flash_vectorized", false});
    // 未来在这里添加 V1, V2...
    // kernels.push_back({launch_v1_naive, "01_V1_Naive_Tiled"});
    return kernels;
}