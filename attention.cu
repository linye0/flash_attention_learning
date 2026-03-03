#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <math.h>
#include <math_constants.h>
#include <cuda_pipeline.h>
#include <mma.h>
#include "attention.h"

#define BR 32
#define BC 32

using namespace nvcuda;

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

// =========================================================
// Kenels
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

__device__ void load_global_to_shared(const float* src, float* dst, int nrow, int ncol, int bx, int tx) {
    const float* src_ptr = src + nrow * ncol * bx;
    for (int i = tx * 4; i < nrow * ncol; i += blockDim.x * 4) *(float4*)(&dst[i]) = *(const float4*)(&src_ptr[i]);
}

__device__ void load_global_to_shared_2(const float* src, float* dst, int nrow, int ncol, int bx, int tx, const float* src2 = nullptr, float* dst2 = nullptr) {
    const float* src_ptr = src + nrow * ncol * bx;
    const float* src2_ptr = src2 + nrow * ncol * bx;
    for (int i = tx * 4; i < nrow * ncol; i += blockDim.x * 4) *(float4*)(&dst[i]) = *(const float4*)(&src_ptr[i]);
    for (int i = tx * 4; i < nrow * ncol; i += blockDim.x * 4) *(float4*)(&dst2[i]) = *(const float4*)(&src2_ptr[i]);
}

__global__ void flash_attn_v2_kernel(
    const float* Q, const float* K, const float* V, float* O,
    int N, int d, int Tc, int Tr, int Bc, int Br, float scale
) {
    int tx = threadIdx.x;
    int bx = blockIdx.x;

    // 这个线程所负责的行
    int row_id = tx / 4; // 范围: 0-31（对应Br的行索引）
    int lane_id = tx % 4; // 范围: 0-3（对应一行内的四个分段）
    int col_offset = lane_id * 16; // 范围: [0, 16, 32, 48]（每个分段的起始位置）

    extern __shared__ float sram[];
    float* s_Q = sram;
    float* s_K = s_Q + Br * d;
    float* s_V = s_K + Bc * d;

    float q_frag[16];
    float k_frag[16];
    float v_frag[16];
    float o_reg[16] = {0.0f};

    float m_i = -CUDART_INF_F;
    float l_i = 0.0f;

    // 阶段1：先从global_mem里面搬运Q到shared_mem的s_Q当中，然后读取到线程私有的寄存器当中
    load_global_to_shared(Q, s_Q, Br, d, bx, tx);
    __syncthreads();

    float4* s_Q_ptr = (float4*)(&s_Q[row_id * d + col_offset]);
    *(float4*)(&q_frag[0]) = s_Q_ptr[0];
    *(float4*)(&q_frag[4]) = s_Q_ptr[1];
    *(float4*)(&q_frag[8]) = s_Q_ptr[2];
    *(float4*)(&q_frag[12]) = s_Q_ptr[3];
    #pragma unroll
    for (int i = 0; i < 16; ++i) q_frag[i] *= scale;

    // 阶段2
    for (int j = 0; j < Tc; ++j) {
        load_global_to_shared_2(K, s_K, Bc, d, j, tx, V, s_V);
        __syncthreads();

        for (int t = 0; t < Bc; ++t) {
            float4* s_K_ptr = (float4*)(&s_K[t * d + col_offset]);
            *(float4*)(&k_frag[0]) = s_K_ptr[0];
            *(float4*)(&k_frag[4]) = s_K_ptr[1];
            *(float4*)(&k_frag[8]) = s_K_ptr[2];
            *(float4*)(&k_frag[12]) = s_K_ptr[3];

            float sum_partial = 0.0f;
            #pragma unroll
            for (int i = 0; i < 16; ++i) {
                sum_partial += q_frag[i] * k_frag[i];
            }

            float S_ij = sum_partial;
            S_ij += __shfl_xor_sync(0xffffffff, S_ij, 1);
            S_ij += __shfl_xor_sync(0xffffffff, S_ij, 2);

            float m_prev = m_i;
            m_i = max(m_prev, S_ij);
            
            float alpha = expf(m_prev - m_i);
            float exp_S = expf(S_ij - m_i);
            l_i = l_i * alpha + exp_S;

            // 5. 加载 V 片段并更新 O 累加器
            float4* s_V_ptr = (float4*)(&s_V[t * d + col_offset]);
            *(float4*)(&v_frag[0])  = s_V_ptr[0];
            *(float4*)(&v_frag[4])  = s_V_ptr[1];
            *(float4*)(&v_frag[8])  = s_V_ptr[2];
            *(float4*)(&v_frag[12]) = s_V_ptr[3];

            #pragma unroll
            for (int i = 0; i < 16; ++i) {
                o_reg[i] = o_reg[i] * alpha + exp_S * v_frag[i];
            }
        }

        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        o_reg[i] /= l_i;
    }

    // 2. 直接向量化写回 (HBM)
    // 对应 O 的 (bx*Br + row_id) 行，col_offset 列开始的 16 个元素
    float4* O_ptr = (float4*)(&O[(bx * Br + row_id) * d + col_offset]);
    O_ptr[0] = *(float4*)(&o_reg[0]);
    O_ptr[1] = *(float4*)(&o_reg[4]);
    O_ptr[2] = *(float4*)(&o_reg[8]);
    O_ptr[3] = *(float4*)(&o_reg[12]);
}

__device__ void load_global_to_shared_async(
    int stage, const float* src, float* dst, 
    int nrow, int ncol, int bc_idx, int tx
) {
    const float* src_ptr = src + nrow * ncol * bc_idx;
    float* dst_ptr = dst + stage * nrow * ncol;

    #pragma unroll
    for (int i = tx * 4; i < nrow * ncol; i += blockDim.x * 4) __pipeline_memcpy_async(&dst_ptr[i], &src_ptr[i], 16);
}

__global__ void flash_atten_v3_kernel(
    const float* Q, const float* K, const float* V, float* O,
    int N, int d, int Tc, int Tr, int Bc, int Br, float scale
) {
    int tx = threadIdx.x;
    int bx = blockIdx.x;

    int row_id = tx / 4; // 范围: 0-31（对应Br的行索引）
    int lane_id = tx % 4; // 范围: 0-3（对应一行内的四个分段）
    int col_offset = lane_id * 16; // 范围: [0, 16, 32, 48]（每个分段的起始位置）

    // 在pipeline读取当中，s_K和s_V的shape都是[2][Bc][d]
    extern __shared__ float sram[];
    float* s_Q = sram;
    float* s_K = s_Q + Br * d;
    float* s_V = s_K + 2 * Bc * d;
    int write_stage = 0;
    int read_stage = 0;

    float q_frag[16];
    float k_frag[16];
    float v_frag[16];
    float o_reg[16] = {0.0f};

    float m_i = -CUDART_INF_F;
    float l_i = 0.0f;

    load_global_to_shared(Q, s_Q, Br, d, bx, tx);
    __syncthreads();
    float4* s_Q_ptr = (float4*)(&s_Q[row_id * d + col_offset]);
    *(float4*)(&q_frag[0]) = s_Q_ptr[0];
    *(float4*)(&q_frag[4]) = s_Q_ptr[1];
    *(float4*)(&q_frag[8]) = s_Q_ptr[2];
    *(float4*)(&q_frag[12]) = s_Q_ptr[3];
    #pragma unroll
    for (int i = 0; i < 16; ++i) q_frag[i] *= scale;

    // preload: 发送第一条读K和V的请求
    load_global_to_shared_async(write_stage, K, s_K, Bc, d, 0, tx);
    load_global_to_shared_async(write_stage, V, s_V, Bc, d, 0, tx);
    __pipeline_commit();

    write_stage ^= 1;

    // 处理中间的K和V读取
    for (int j = 1; j < Tc; ++j) {
        load_global_to_shared_async(write_stage, K, s_K, Bc, d, j, tx);
        load_global_to_shared_async(write_stage, V, s_V, Bc, d, j, tx);
        __pipeline_commit();

        __pipeline_wait_prior(1);

        __syncthreads();

        // 对读取到的K和V进行计算
        for (int t = 0; t < Bc; ++t) {
            float4* s_K_ptr = (float4*)(s_K + read_stage * Bc * d + t * d + col_offset);
            *(float4*)(&k_frag[0]) = s_K_ptr[0];
            *(float4*)(&k_frag[4]) = s_K_ptr[1];
            *(float4*)(&k_frag[8]) = s_K_ptr[2];
            *(float4*)(&k_frag[12]) = s_K_ptr[3];

            
            float sum_partial = 0.0f;
            #pragma unroll
            for (int i = 0; i < 16; ++i) {
                sum_partial += q_frag[i] * k_frag[i];
            }

            float S_ij = sum_partial;
            S_ij += __shfl_xor_sync(0xffffffff, S_ij, 1);
            S_ij += __shfl_xor_sync(0xffffffff, S_ij, 2);

            float m_prev = m_i;
            m_i = max(m_prev, S_ij);
            
            float alpha = expf(m_prev - m_i);
            float exp_S = expf(S_ij - m_i);
            l_i = l_i * alpha + exp_S;

            float4* s_V_ptr = (float4*)(s_V + read_stage * Bc * d + t * d + col_offset);
            *(float4*)(&v_frag[0])  = s_V_ptr[0];
            *(float4*)(&v_frag[4])  = s_V_ptr[1];
            *(float4*)(&v_frag[8])  = s_V_ptr[2];
            *(float4*)(&v_frag[12]) = s_V_ptr[3];

            #pragma unroll
            for (int i = 0; i < 16; ++i) {
                o_reg[i] = o_reg[i] * alpha + exp_S * v_frag[i];
            }
        }

        __syncthreads();

        write_stage ^= 1;
        read_stage ^= 1;
    }

    // 处理最后的K和V读取
    __pipeline_wait_prior(0);
    __syncthreads();

    for (int t = 0; t < Bc; ++t) {
        float4* s_K_ptr = (float4*)(s_K + read_stage * Bc * d + t * d + col_offset);
        *(float4*)(&k_frag[0]) = s_K_ptr[0];
        *(float4*)(&k_frag[4]) = s_K_ptr[1];
        *(float4*)(&k_frag[8]) = s_K_ptr[2];
        *(float4*)(&k_frag[12]) = s_K_ptr[3];

        
        float sum_partial = 0.0f;
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            sum_partial += q_frag[i] * k_frag[i];
        }

        float S_ij = sum_partial;
        S_ij += __shfl_xor_sync(0xffffffff, S_ij, 1);
        S_ij += __shfl_xor_sync(0xffffffff, S_ij, 2);

        float m_prev = m_i;
        m_i = max(m_prev, S_ij);
        
        float alpha = expf(m_prev - m_i);
        float exp_S = expf(S_ij - m_i);
        l_i = l_i * alpha + exp_S;

        float4* s_V_ptr = (float4*)(s_V + read_stage * Bc * d + t * d + col_offset);
        *(float4*)(&v_frag[0])  = s_V_ptr[0];
        *(float4*)(&v_frag[4])  = s_V_ptr[1];
        *(float4*)(&v_frag[8])  = s_V_ptr[2];
        *(float4*)(&v_frag[12]) = s_V_ptr[3];

        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            o_reg[i] = o_reg[i] * alpha + exp_S * v_frag[i];
        }
    }

    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        o_reg[i] /= l_i;
    }

    // 2. 直接向量化写回 (HBM)
    // 对应 O 的 (bx*Br + row_id) 行，col_offset 列开始的 16 个元素
    float4* O_ptr = (float4*)(&O[(bx * Br + row_id) * d + col_offset]);
    O_ptr[0] = *(float4*)(&o_reg[0]);
    O_ptr[1] = *(float4*)(&o_reg[4]);
    O_ptr[2] = *(float4*)(&o_reg[8]);
    O_ptr[3] = *(float4*)(&o_reg[12]);
}

// =========================================================
// Launchers
// =========================================================
void launch_v0_cublas(cublasHandle_t handle, const void* Q_ptr, const void* K_ptr, const void* V_ptr, void* O_ptr, float* S, float* P, int N, int d) {
    const float* Q = reinterpret_cast<const float*>(Q_ptr);
    const float* K = reinterpret_cast<const float*>(K_ptr);
    const float* V = reinterpret_cast<const float*>(V_ptr);
    float* O = reinterpret_cast<float*>(O_ptr);
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

void launch_v1_flash_tiling(cublasHandle_t handle, const void* Q_ptr, const void* K_ptr, const void* V_ptr, void* O_ptr, float* S, float* P, int N, int d) {
    const float* Q = reinterpret_cast<const float*>(Q_ptr);
    const float* K = reinterpret_cast<const float*>(K_ptr);
    const float* V = reinterpret_cast<const float*>(V_ptr);
    float* O = reinterpret_cast<float*>(O_ptr);
    // 从v1开始，我们采用online计算中间矩阵的方式，所以传入的handle，S，P都是无用的变量
    const int Br = BR;
    const int Bc = BC;
    
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

void launch_v2_flash_vectorized(cublasHandle_t handle, const void* Q_ptr, const void* K_ptr, const void* V_ptr, void* O_ptr, float* S, float* P, int N, int d) {
    const float* Q = reinterpret_cast<const float*>(Q_ptr);
    const float* K = reinterpret_cast<const float*>(K_ptr);
    const float* V = reinterpret_cast<const float*>(V_ptr);
    float* O = reinterpret_cast<float*>(O_ptr);

    const int Br = BR;
    const int Bc = BC;

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
        printf("FlashAttn V3 Launch Error: %s\n", cudaGetErrorString(err));
    }
}

void launch_v3_flash_pipeline(cublasHandle_t handle, const void* Q_ptr, const void* K_ptr, const void* V_ptr, void* O_ptr, float* S, float* P, int N, int d) {
    const float* Q = reinterpret_cast<const float*>(Q_ptr);
    const float* K = reinterpret_cast<const float*>(K_ptr);
    const float* V = reinterpret_cast<const float*>(V_ptr);
    float* O = reinterpret_cast<float*>(O_ptr);

    const int Br = BR;
    const int Bc = BC;

    const int Tr = (N + Br - 1) / Br;
    const int Tc = (N + Bc - 1) / Bc;
    const float scale = 1.0f / sqrtf(d);

    dim3 grid(Tr);
    dim3 block(128);

    size_t shared_mem_size = (Br * d + 4 * Bc * d) * sizeof(float);

    if (shared_mem_size > 48 * 1024) {
        fprintf(stderr, "Error: Shared memory request (%zu bytes) exceeds 48KB limit!\n", shared_mem_size);
        exit(1);
    }

    flash_atten_v3_kernel<<<grid, block, shared_mem_size>>>(Q, K, V, O, N, d, Tc, Tr, Bc, Br, scale);

    cudaError err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("FlashAttn V3 Launch Error: %s\n", cudaGetErrorString(err));
    }
}

void launch_v4_flash_wmma(cublasHandle_t handle, const void* Q_ptr, const void* K_ptr, const void* V_ptr, void* O_ptr, float* S, float* P, int N, int d) {
    const half* Q = reinterpret_cast<const half*>(Q_ptr);
    const half* K = reinterpret_cast<const half*>(K_ptr);
    const half* V = reinterpret_cast<const half*>(V_ptr);
    half* O = reinterpret_cast<half*>(O_ptr);

    const int Br = BR;
    const int Bc = BC;

    const int Tr = (N + Br - 1) / Br;
    const int Tc = (N + Bc - 1) / Bc;
    const float scale = 1.0f / sqrtf(d);

    dim3 grid(Tr);
    dim3 block(128);

    size_t shared_mem_size = (Br * d + 4 * Bc * d) * sizeof(float);

    if (shared_mem_size > 48 * 1024) {
        fprintf(stderr, "Error: Shared memory request (%zu bytes) exceeds 48KB limit!\n", shared_mem_size);
        exit(1);
    }

    flash_atten_v4_kernel<<<grid, block, shared_mem_size>>>(Q, K, V, O, N, d, Tc, Tr, Bc, Br, scale);

    cudaError err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("FlashAttn V4 Launch Error: %s\n", cudaGetErrorString(err));
    }
}

// 注册列表
std::vector<KernelInfo> get_kernels() {
    std::vector<KernelInfo> kernels;
    // kernels.push_back({launch_v0_cublas, "V0_Multipass", true, false});
    // kernels.push_back({launch_v1_flash_tiling, "V1_flash_tiling", false, false});
    kernels.push_back({launch_v2_flash_vectorized, "V2_flash_vectorized", false, false});
    kernels.push_back({launch_v3_flash_pipeline, "V3_flash_pipeline", false, false});
    kernels.push_back({launch_v4_flash_wmma, "V4_flash_wmma", false, true});
    // 未来在这里添加 V1, V2...
    // kernels.push_back({launch_v1_naive, "01_V1_Naive_Tiled"});
    return kernels;
}