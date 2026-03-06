#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <math.h>
#include <math_constants.h>
#include <cuda_pipeline.h>
#include <mma.h>
#include <memory>         // 必须包含，用于 shared_ptr
#include <unordered_map>  // 必须包含
#include <iostream>
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
    extern __shared__ char dynamic_sram[];
    float* sram = (float*)dynamic_sram;
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

__device__ __forceinline__ void load_global_to_shared(const float* src, float* dst, int nrow, int ncol, int bx, int tx) {
    const float* src_ptr = src + nrow * ncol * bx;
    for (int i = tx * 4; i < nrow * ncol; i += blockDim.x * 4) *(float4*)(&dst[i]) = *(const float4*)(&src_ptr[i]);
}

__device__ __forceinline__ void load_global_to_shared_2(const float* src, float* dst, int nrow, int ncol, int bx, int tx, const float* src2 = nullptr, float* dst2 = nullptr) {
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

    extern __shared__ char dynamic_sram[];
    float* sram = (float*)dynamic_sram;
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

__device__ __forceinline__ void load_global_to_shared_async(
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
    extern __shared__ char dynamic_sram[];
    float* sram = (float*)dynamic_sram;
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


        write_stage ^= 1;
        read_stage ^= 1;
        __syncthreads();
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

__device__ __forceinline__ void load_global_to_shared_half(
    const half* src, half* dst, 
    int nrow, int ncol, int bc_idx, int tx
) {
    const half* src_ptr = src + bc_idx * nrow * ncol;
    #pragma unroll
    for (int i = tx * 8; i < nrow * ncol; i += blockDim.x * 8) *((uint4*)(&dst[i])) = *((const uint4*)(&src_ptr[i]));
}

__device__ __forceinline__ void load_global_to_shared_half_async(
    int stage, const half* src, half* dst,
    int nrow, int ncol, int bc_idx, int tx
) {
    const half* src_ptr = src + bc_idx * nrow * ncol;
    half* dst_ptr = dst + stage * nrow * ncol;
    #pragma unroll
    for (int i = tx * 8; i < nrow * ncol; i += blockDim.x * 8) __pipeline_memcpy_async(&dst_ptr[i], &src_ptr[i], 16);
}

__global__ void flash_atten_v4_kernel(
    const half* Q, const half* K, const half* V, half* O,
    int N, int d, int Tc, int Tr, int Bc, int Br, float scale
) {
    int tx = threadIdx.x;
    int bx = blockIdx.x;

    int warp_id = tx / 32;

    int row_offset_q = warp_id * 16;

    // WMMA frag声明
    // q_frag的shape就是(16,16)
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> q_frag;
    // k_frag的shape是(Bc, 16) = (32,16)，如果是K^T则是(16,32)，超出了16*16，所以要两个frag
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> k_frag[2];
    // s,p,v同理，16*32
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> s_frag[2];
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> p_frag[2];
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> v_frag[2];
    // O是16*d=16*64，分为四段
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> o_frag[4];
    
    for (int i = 0; i < 4; ++i) wmma::fill_fragment(o_frag[i], 0.0f);

    extern __shared__ char dynamic_sram[];
    half* sram = (half*)dynamic_sram;
    half* s_Q = sram;
    half* s_K = s_Q + Br * d;
    half* s_V = s_K + 2 * Bc * d;
    int write_stage = 0;
    int read_stage = 0;

    float m1 = -CUDART_INF_F, m2 = -CUDART_INF_F;
    float l1 = 0.0f, l2 = 0.0f;

    load_global_to_shared_half(Q, s_Q, Br, d, bx, tx);
    __syncthreads();

    // preload
    load_global_to_shared_half_async(write_stage, K, s_K, Bc, d, 0, tx);
    load_global_to_shared_half_async(write_stage, V, s_V, Bc, d, 0, tx);
    write_stage ^= 1;
    __pipeline_commit();

    for (int j = 1; j < Tc; ++j) {
        load_global_to_shared_half_async(write_stage, K, s_K, Bc, d, j, tx);
        load_global_to_shared_half_async(write_stage, V, s_V, Bc, d, j, tx);
        __pipeline_commit();

        __pipeline_wait_prior(1);

        __syncthreads();

        wmma::fill_fragment(s_frag[0], 0.0f);
        wmma::fill_fragment(s_frag[1], 0.0f);
        
        #pragma unroll
        for (int ki = 0; ki < d / WMMA_K; ++ki) {
            const half* q_tile_ptr = s_Q + row_offset_q * d + ki * 16;
            wmma::load_matrix_sync(q_frag, q_tile_ptr, d);

            // 分别读取K^T的左右两块
            const half* k_ptr_0 = s_K + read_stage * Bc * d + 0 * d + ki * 16;
            const half* k_ptr_1 = s_K + read_stage * Bc * d + 16 * d + ki * 16;
            wmma::load_matrix_sync(k_frag[0], k_ptr_0, d);
            wmma::load_matrix_sync(k_frag[1], k_ptr_1, d);

            wmma::mma_sync(s_frag[0], q_frag, k_frag[0], s_frag[0]);
            wmma::mma_sync(s_frag[1], q_frag, k_frag[1], s_frag[1]);
        }


        float m1_local = -CUDART_INF_F, m2_local = -CUDART_INF_F;

        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            float val0 = s_frag[0].x[i] * scale;
            float val1 = s_frag[1].x[i] * scale;
            if ((i % 4) < 2) {
                m1_local = fmaxf(m1_local, fmaxf(val0, val1));
            } else {
                m2_local = fmaxf(m2_local,  fmaxf(val0, val1));
            }
        }

        #pragma unroll
        for (int mask = 2; mask > 0; mask >>= 1) {
            m1_local = fmaxf(m1_local, __shfl_xor_sync(0xffffffff, m1_local, mask));
            m2_local = fmaxf(m2_local, __shfl_xor_sync(0xffffffff, m2_local, mask));
        }

        float m1_new = fmaxf(m1, m1_local);
        float m2_new = fmaxf(m2, m2_local);

        float sum1_local = 0.0f, sum2_local = 0.0f;

        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            float p0, p1;
            if ((i % 4) < 2) {
                p0 = expf(s_frag[0].x[i] * scale - m1_new);
                p1 = expf(s_frag[1].x[i] * scale - m1_new);
                sum1_local += (p0 + p1);
            } else {
                p0 = expf(s_frag[0].x[i] * scale - m2_new);
                p1 = expf(s_frag[1].x[i] * scale - m2_new);
                sum2_local += (p0 + p1);
            }
            p_frag[0].x[i] = __float2half(p0);
            p_frag[1].x[i] = __float2half(p1);
        }

        #pragma unroll
        for (int mask = 2; mask > 0; mask >>= 1) {
            sum1_local += __shfl_xor_sync(0xffffffff, sum1_local, mask);
            sum2_local += __shfl_xor_sync(0xffffffff, sum2_local, mask);
        }

        float scale1_o = expf(m1 - m1_new);
        float scale2_o = expf(m2 - m2_new);

        l1 = l1 * scale1_o + sum1_local;
        l2 = l2 * scale2_o + sum2_local;
        m1 = m1_new;
        m2 = m2_new;

        for(int vi=0; vi<4; ++vi) {
            #pragma unroll
            for(int i=0; i<8; ++i) {
                if ((i % 4) < 2) o_frag[vi].x[i] *= scale1_o;
                else             o_frag[vi].x[i] *= scale2_o;
            }
        }

        const half* cur_s_V = s_V + read_stage * Bc * d;
        for(int vi = 0; vi < 4; ++vi) { // 遍历 V 的 4 个列块 (d=64 / 16 = 4)
            // 加载 V 的上半部 (行 0-15) 和下半部 (行 16-31)
            wmma::load_matrix_sync(v_frag[0], cur_s_V + 0 * d + vi * 16, d);
            wmma::load_matrix_sync(v_frag[1], cur_s_V + 16 * d + vi * 16, d);

            // 矩阵分块乘加：O = P_left * V_top + P_right * V_bottom
            wmma::mma_sync(o_frag[vi], p_frag[0], v_frag[0], o_frag[vi]);
            wmma::mma_sync(o_frag[vi], p_frag[1], v_frag[1], o_frag[vi]);
        }

        write_stage ^= 1;
        read_stage ^= 1;
        __syncthreads();
    }

    __pipeline_wait_prior(0);
    __syncthreads();

    wmma::fill_fragment(s_frag[0], 0.0f);
    wmma::fill_fragment(s_frag[1], 0.0f);
        
    #pragma unroll
    for (int ki = 0; ki < d / WMMA_K; ++ki) {
        const half* q_tile_ptr = s_Q + row_offset_q * d + ki * 16;
        wmma::load_matrix_sync(q_frag, q_tile_ptr, d);

        // 分别读取K^T的左右两块
        const half* k_ptr_0 = s_K + read_stage * Bc * d + 0 * d + ki * 16;
        const half* k_ptr_1 = s_K + read_stage * Bc * d + 16 * d + ki * 16;
        wmma::load_matrix_sync(k_frag[0], k_ptr_0, d);
        wmma::load_matrix_sync(k_frag[1], k_ptr_1, d);

        wmma::mma_sync(s_frag[0], q_frag, k_frag[0], s_frag[0]);
        wmma::mma_sync(s_frag[1], q_frag, k_frag[1], s_frag[1]);
    }

    float m1_local = -CUDART_INF_F, m2_local = -CUDART_INF_F;

    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        float val0 = s_frag[0].x[i] * scale;
        float val1 = s_frag[1].x[i] * scale;
        if ((i % 4) < 2) {
            m1_local = fmaxf(m1_local, fmaxf(val0, val1));
        } else {
            m2_local = fmaxf(m2_local,  fmaxf(val0, val1));
        }
    }

    #pragma unroll
    for (int mask = 2; mask > 0; mask >>= 1) {
        m1_local = fmaxf(m1_local, __shfl_xor_sync(0xffffffff, m1_local, mask));
        m2_local = fmaxf(m2_local, __shfl_xor_sync(0xffffffff, m2_local, mask));
    }

    float m1_new = fmaxf(m1, m1_local);
    float m2_new = fmaxf(m2, m2_local);

    float sum1_local = 0.0f, sum2_local = 0.0f;

    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        float p0, p1;
        if ((i % 4) < 2) {
            p0 = expf(s_frag[0].x[i] * scale - m1_new);
            p1 = expf(s_frag[1].x[i] * scale - m1_new);
            sum1_local += (p0 + p1);
        } else {
            p0 = expf(s_frag[0].x[i] * scale - m2_new);
            p1 = expf(s_frag[1].x[i] * scale - m2_new);
            sum2_local += (p0 + p1);
        }
        p_frag[0].x[i] = __float2half(p0);
        p_frag[1].x[i] = __float2half(p1);
    }

    #pragma unroll
    for (int mask = 2; mask > 0; mask >>= 1) {
        sum1_local += __shfl_xor_sync(0xffffffff, sum1_local, mask);
        sum2_local += __shfl_xor_sync(0xffffffff, sum2_local, mask);
    }

    float scale1_o = expf(m1 - m1_new);
    float scale2_o = expf(m2 - m2_new);

    l1 = l1 * scale1_o + sum1_local;
    l2 = l2 * scale2_o + sum2_local;
    m1 = m1_new;
    m2 = m2_new;

    for(int vi=0; vi<4; ++vi) {
        #pragma unroll
        for(int i=0; i<8; ++i) {
            if ((i % 4) < 2) o_frag[vi].x[i] *= scale1_o;
            else             o_frag[vi].x[i] *= scale2_o;
        }
    }

    const half* cur_s_V = s_V + read_stage * Bc * d;
    for(int vi = 0; vi < 4; ++vi) { // 遍历 V 的 4 个列块 (d=64 / 16 = 4)
        // 加载 V 的上半部 (行 0-15) 和下半部 (行 16-31)
        wmma::load_matrix_sync(v_frag[0], cur_s_V + 0 * d + vi * 16, d);
        wmma::load_matrix_sync(v_frag[1], cur_s_V + 16 * d + vi * 16, d);

        // 矩阵分块乘加：O = P_left * V_top + P_right * V_bottom
        wmma::mma_sync(o_frag[vi], p_frag[0], v_frag[0], o_frag[vi]);
        wmma::mma_sync(o_frag[vi], p_frag[1], v_frag[1], o_frag[vi]);
    }

    // ==========================================
    // 最终阶段：归一化与写回 (Normalization & Store)
    // ==========================================

    // 1. 在寄存器内部进行除以 l_i 的归一化操作
    for(int vi = 0; vi < 4; ++vi) {
        #pragma unroll
        for(int i = 0; i < 8; ++i) {
            if ((i % 4) < 2) {
                o_frag[vi].x[i] /= l1;
            } else {
                o_frag[vi].x[i] /= l2;
            }
        }
    }

    // 2. 物理地址映射与内存写回
    // 计算当前 Warp 负责的全局 O 矩阵的起始行号
    int global_row_idx = bx * Br + row_offset_q;

    // 3. 声明 half 类型的目标中转 Fragment
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> o_frag_half;

    // 4. 遍历并写回 4 个 O 分块
    for(int vi = 0; vi < 4; ++vi) {
        
        // 核心修正：在寄存器内部手动完成 float 到 half 的精度截断
        #pragma unroll
        for(int i = 0; i < 8; ++i) {
            o_frag_half.x[i] = __float2half(o_frag[vi].x[i]);
        }

        // 计算当前子块在 HBM 中的起始指针
        half* O_ptr = O + global_row_idx * d + vi * 16;

        // 此时指针 (half*) 与 Fragment (half) 类型严格匹配，安全写入
        wmma::store_matrix_sync(O_ptr, o_frag_half, d, wmma::mem_row_major);
    }
}

__global__ void flash_decoding_partial_kernel(
    const float* Q, const float* K, const float* V,
    float* partial_O, float* partial_L, float* partial_M,
    int N, int d, int Bc, float scale
) {
    int tx = threadIdx.x;
    int split_idx = blockIdx.x;

    int start_n = split_idx * Bc;
    int end_n = min(start_n + Bc, N);

    // 1. 加载单行 Q 到 Shared Memory (所有线程共用)
    extern __shared__ float sram[];
    float* s_Q = sram; 
    for (int k = tx; k < d; k += blockDim.x) {
        s_Q[k] = Q[k];
    }
    __syncthreads();

    // 2. 线程局部状态初始化
    float m_i = -CUDART_INF_F;
    float l_i = 0.0f;
    float o_reg[64] = {0.0f};

    // -----------------------------------------------------------
    // 核心：处理 Bc > threads 的情况。每个线程通过循环处理多个 Token
    // -----------------------------------------------------------
    for (int i = start_n + tx; i < end_n; i += blockDim.x) {
        float sum = 0.0f;
        // 使用 float4 优化 K 的读取 (假设 d=64 且对齐)
        #pragma unroll
        for (int k = 0; k < d; k += 4) {
            float4 k_val = *(const float4*)(&K[i * d + k]);
            sum += s_Q[k + 0] * k_val.x;
            sum += s_Q[k + 1] * k_val.y;
            sum += s_Q[k + 2] * k_val.z;
            sum += s_Q[k + 3] * k_val.w;
        }
        sum *= scale;

        // Online Softmax 更新
        float m_prev = m_i;
        m_i = fmaxf(m_prev, sum);
        float alpha = expf(m_prev - m_i);
        float beta = expf(sum - m_i);

        l_i = l_i * alpha + beta;
        #pragma unroll
        for (int k = 0; k < d; k += 4) {
            float4 v_val = *(const float4*)(&V[i * d + k]);
            o_reg[k + 0] = o_reg[k + 0] * alpha + beta * v_val.x;
            o_reg[k + 1] = o_reg[k + 1] * alpha + beta * v_val.y;
            o_reg[k + 2] = o_reg[k + 2] * alpha + beta * v_val.z;
            o_reg[k + 3] = o_reg[k + 3] * alpha + beta * v_val.w;
        }
    }

    // -----------------------------------------------------------
    // 3. 块内归约 (Block-level Reduction)
    // -----------------------------------------------------------
    
    // 第一步：Warp Shuffle 归约 (32 线程合并) [cite: 2026-03-03]
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        float m_other = __shfl_xor_sync(0xffffffff, m_i, offset);
        float l_other = __shfl_xor_sync(0xffffffff, l_i, offset);

        float m_max = fmaxf(m_i, m_other);
        float a = expf(m_i - m_max);
        float b = expf(m_other - m_max);

        m_i = m_max;
        l_i = l_i * a + l_other * b;
        #pragma unroll
        for (int k = 0; k < d; ++k) {
            float o_other = __shfl_xor_sync(0xffffffff, o_reg[k], offset);
            o_reg[k] = o_reg[k] * a + o_other * b;
        }
    }

    // 第二步：跨 Warp 归约 (使用 Shared Memory 交换各 Warp 领头线程的结果)
    __shared__ float s_m[32]; // 最多支持 1024 线程/32 = 32 个 Warp
    __shared__ float s_l[32];
    __shared__ float s_o[32][64];

    int lane = tx % 32;
    int wid = tx / 32;
    int num_warps = blockDim.x / 32;

    if (lane == 0) {
        s_m[wid] = m_i;
        s_l[wid] = l_i;
        #pragma unroll
        for (int k = 0; k < d; ++k) s_o[wid][k] = o_reg[k];
    }
    __syncthreads();

    // 第三步：由 tx=0 线程对 4 个 Warp 的结果进行最终合并
    if (tx == 0) {
        float M_block = s_m[0];
        float L_block = s_l[0];
        float O_block[64];
        #pragma unroll
        for (int k = 0; k < d; ++k) O_block[k] = s_o[0][k];

        for (int w = 1; w < num_warps; ++w) {
            float m_w = s_m[w];
            float l_w = s_l[w];
            float m_max = fmaxf(M_block, m_w);
            float a = expf(M_block - m_max);
            float b = expf(m_w - m_max);

            L_block = L_block * a + l_w * b;
            M_block = m_max;
            #pragma unroll
            for (int k = 0; k < d; ++k) O_block[k] = O_block[k] * a + s_o[w][k] * b;
        }

        // 4. 写入 Workspace
        partial_M[split_idx] = M_block;
        partial_L[split_idx] = L_block;
        #pragma unroll
        for (int k = 0; k < d; ++k) partial_O[split_idx * d + k] = O_block[k];
    }
}

__global__ void flash_decoding_reduction_kernel(
    const float* partial_O, const float* partial_L, const float* partial_M,
    float* O, int d, int num_splits
) {
    int tx = threadIdx.x; // 每个线程负责结果向量的一个维度
    if (tx >= d) return;

    // 1. 寻找全局最大值 M_global
    float m_global = -CUDART_INF_F;
    for (int i = 0; i < num_splits; ++i) {
        m_global = fmaxf(m_global, partial_M[i]);
    }

    // 2. 重放缩并累加
    float l_global = 0.0f;
    float o_final = 0.0f;

    for (int i = 0; i < num_splits; ++i) {
        float alpha = expf(partial_M[i] - m_global); // 重放缩因子
        l_global += partial_L[i] * alpha;
        o_final += partial_O[i * d + tx] * alpha;
    }

    // 3. 归一化并写回
    O[tx] = o_final / l_global;
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

    const int Br = 64;
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

void launch_v5_flash_decoding(
    cublasHandle_t handle, const void* Q_ptr, const void* K_ptr, const void* V_ptr, 
    void* O_ptr, float* S, float* P, int N, int d
) {
    const float* Q = (const float*)Q_ptr;
    const float* K = (const float*)K_ptr;
    const float* V = (const float*)V_ptr;
    float* O = (float*)O_ptr;

    int Bc = 256; // 分片大小，支持 Bc > threads (128)
    int threads = 128;
    int num_splits = (N + Bc - 1) / Bc;
    float scale = 1.0f / sqrtf(d);

    // 1. 分配 Workspace 显存 (生产环境应预分配以避免 overhead)
    float *d_pO, *d_pL, *d_pM;
    cudaMalloc(&d_pO, num_splits * d * sizeof(float));
    cudaMalloc(&d_pL, num_splits * sizeof(float));
    cudaMalloc(&d_pM, num_splits * sizeof(float));

    // 2. 执行 Partial Kernel
    // SRAM 大小：s_Q(d) + 额外空间
    size_t shared_mem = d * sizeof(float); 
    flash_decoding_partial_kernel<<<num_splits, threads, shared_mem>>>(
        Q, K, V, d_pO, d_pL, d_pM, N, d, Bc, scale
    );

    // 3. 执行 Reduction Kernel
    // 一个 Block 处理所有分片的归约，线程数等于向量维度 d
    flash_decoding_reduction_kernel<<<1, d>>>(d_pO, d_pL, d_pM, O, d, num_splits);

    // 4. 清理
    cudaFree(d_pO); cudaFree(d_pL); cudaFree(d_pM);
}



// 注册列表
std::vector<KernelInfo> get_kernels() {
    std::vector<KernelInfo> kernels;
    // kernels.push_back({launch_v0_cublas, "V0_Multipass", true, false, false});
    // kernels.push_back({launch_v1_flash_tiling, "V1_flash_tiling", false, false, false});
    // kernels.push_back({launch_v2_flash_vectorized, "V2_flash_vectorized", false, false, false});
    // kernels.push_back({launch_v3_flash_pipeline, "V3_flash_pipeline", false, false, false});
    kernels.push_back({launch_v4_flash_wmma, "V4_flash_wmma", false, true, false});
    #ifdef USE_CUDNN
    // 注册 NVIDIA 官方 Baseline
    kernels.push_back({launch_cudnn_baseline, "Baseline_cuDNN_SDPA", false, false, false});
    #endif
    // kernels.push_back({launch_v5_flash_decoding, "V5_flash_decoding", false, false, true});
    return kernels;
}