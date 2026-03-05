#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "attention.h"
#include <cuda_fp16.h>

#define START_N 1024
#define END_N 62464
#define STEP_N 4096
#define HEAD_DIM 64

#define CHECK_CUDA(call) { \
    const cudaError_t error = call; \
    if (error != cudaSuccess) { \
        printf("Error: %s:%d, code: %d, reason: %s\n", __FILE__, __LINE__, error, cudaGetErrorString(error)); \
        exit(1); \
    } \
}

bool verify_result(const float* gpu_res, const float* cpu_res, int check_rows, int d, float tolerance) {
    for (int i = 0; i < check_rows * d; ++i) {
        float diff = std::abs(gpu_res[i] - cpu_res[i]);
        float rel_diff = diff / (std::abs(cpu_res[i]) + 1e-5f);
        if (rel_diff > tolerance) {
            printf("Error at index %d: GPU=%f, CPU=%f, RelDiff=%f\n", i, gpu_res[i], cpu_res[i], rel_diff);
            return false;
        }
    }
    return true;
}

__global__ void float2half_kernel(const float* src, half* dst, size_t num_elements) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        dst[idx] = __float2half(src[idx]);
    }
}

__global__ void half2float_kernel(const half* src, float* dst, size_t num_elements) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        dst[idx] = __half2float(src[idx]);
    }
}

int main() {
    srand(time(NULL));

    cublasHandle_t handle;
    cublasCreate(&handle);
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    int max_N = END_N;
    size_t max_bytes_QKV = (size_t)max_N * HEAD_DIM * sizeof(float);

    // ==========================================
    // 1. 全局静态显存分配 (FP32 & FP16)
    // ==========================================
    std::vector<float> h_Q(max_N * HEAD_DIM);
    std::vector<float> h_K(max_N * HEAD_DIM);
    std::vector<float> h_V(max_N * HEAD_DIM);

    for (int i = 0; i < h_Q.size(); ++i) {
        h_Q[i] = static_cast<float>(rand()) / RAND_MAX;
        h_K[i] = static_cast<float>(rand()) / RAND_MAX;
        h_V[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    float *d_Q, *d_K, *d_V, *d_O;
    CHECK_CUDA(cudaMalloc(&d_Q, max_bytes_QKV));
    CHECK_CUDA(cudaMalloc(&d_K, max_bytes_QKV));
    CHECK_CUDA(cudaMalloc(&d_V, max_bytes_QKV));
    CHECK_CUDA(cudaMalloc(&d_O, max_bytes_QKV));

    CHECK_CUDA(cudaMemcpy(d_Q, h_Q.data(), max_bytes_QKV, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_K, h_K.data(), max_bytes_QKV, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_V, h_V.data(), max_bytes_QKV, cudaMemcpyHostToDevice));

    // 核心修复：FP16 显存一次性全量分配，杜绝循环内碎片化申请
    size_t max_bytes_QKV_half = (size_t)max_N * HEAD_DIM * sizeof(half);
    half *d_Q_half, *d_K_half, *d_V_half, *d_O_half;
    CHECK_CUDA(cudaMalloc(&d_Q_half, max_bytes_QKV_half));
    CHECK_CUDA(cudaMalloc(&d_K_half, max_bytes_QKV_half));
    CHECK_CUDA(cudaMalloc(&d_V_half, max_bytes_QKV_half));
    CHECK_CUDA(cudaMalloc(&d_O_half, max_bytes_QKV_half));

    std::vector<KernelInfo> kernels = get_kernels();

    // ==========================================
    // 2. 暴力热身，强制跨越 P-State 墙
    // ==========================================
    float* dummy_d;
    cudaMalloc(&dummy_d, 1024);
    for(int i = 0; i < 5000; i++) {
        cudaMemset(dummy_d, 0, 1024); 
    }
    cudaDeviceSynchronize();
    cudaFree(dummy_d);
    printf("%-12s, %-20s, %10s, %10s\n", "Seq_Len(N)", "KernelName", "GFLOPS", "Time(ms)");

    // ==========================================
    // 3. 性能测试主循环
    // ==========================================
    for (int N = START_N; N <= END_N; N += STEP_N) {

        for (const auto& kernel : kernels) {
            float* d_S = nullptr, *d_P = nullptr;

            // O(N^2) 显存的 Multipass 特殊处理（由于过大，保留动态尝试）
            if (kernel.is_multipass) {
                size_t bytes_NxN = (size_t)N * N * sizeof(float);
                cudaGetLastError(); // Clear previous errors
                cudaError_t err1 = cudaMalloc(&d_S, bytes_NxN);
                cudaError_t err2 = cudaMalloc(&d_P, bytes_NxN);

                if (err1 != cudaSuccess || err2 != cudaSuccess) {
                    if (d_S) cudaFree(d_S);
                    if (d_P) cudaFree(d_P);
                    printf("%-12d, %-20s, %10s, %10s\n", N, kernel.name.c_str(), "OOM", "-");
                    cudaGetLastError();
                    continue; 
                }
            }   

            void *launch_Q = d_Q, *launch_K = d_K, *launch_V = d_V, *launch_O = d_O;
            size_t elements_QKV = (size_t)N * HEAD_DIM;

            // FP16 指针切换与数据准备
            if (kernel.is_halfacc) {
                int threads = 256;
                int blocks = (elements_QKV + threads - 1) / threads;
                float2half_kernel<<<blocks, threads>>>(d_Q, d_Q_half, elements_QKV);
                float2half_kernel<<<blocks, threads>>>(d_K, d_K_half, elements_QKV);
                float2half_kernel<<<blocks, threads>>>(d_V, d_V_half, elements_QKV);
                cudaDeviceSynchronize();

                launch_Q = d_Q_half;
                launch_K = d_K_half;
                launch_V = d_V_half;
                launch_O = d_O_half;
            }

            // ==========================================
            // 核心修复：自适应测速 (Adaptive Benchmark)
            // ==========================================
            // a) 单次摸底执行
            cudaEventRecord(start);
            kernel.func(handle, launch_Q, launch_K, launch_V, launch_O, d_S, d_P, N, HEAD_DIM);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);

            float warmup_ms = 0;
            cudaEventElapsedTime(&warmup_ms, start, stop);

            // b) 根据单次耗时，反推 200ms 窗口所需的 REPEAT 次数
            float safe_warmup = std::max(warmup_ms, 0.005f); 
            int REPEAT = static_cast<int>(200.0f / safe_warmup);
            REPEAT = std::max(3, std::min(500, REPEAT)); // 限制在 3~500 次之间

            // c) 正式执行与计时
            cudaEventRecord(start);
            for(int r = 0; r < REPEAT; ++r) {
                kernel.func(handle, launch_Q, launch_K, launch_V, launch_O, d_S, d_P, N, HEAD_DIM);
            }
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);

            float milliseconds = 0;
            cudaEventElapsedTime(&milliseconds, start, stop);
            float avg_time = milliseconds / REPEAT;

            // 计算与打印
            double ops = kernel.is_decoding ? (4.0 * N * HEAD_DIM) : (4.0 * N * N * HEAD_DIM);
            double gflops = ops / (avg_time * 1e-3) / 1e9;

            printf("%-12d, %-20s, %10.2f, %10.3f\n", N, kernel.name.c_str(), gflops, avg_time);
            fflush(stdout);

            // FP16 结果转回 FP32 用于校验
            if (kernel.is_halfacc) {
                int threads = 256;
                int blocks = (elements_QKV + threads - 1) / threads;
                half2float_kernel<<<blocks, threads>>>(d_O_half, d_O, elements_QKV); 
                cudaDeviceSynchronize();
            }

            // ==========================================
            // 4. 精确 CPU 校验逻辑
            // ==========================================
            int check_rows = kernel.is_decoding ? 1 : std::min(N, 100); 
            std::vector<float> h_O_gpu(check_rows * HEAD_DIM);
            std::vector<float> h_O_cpu(check_rows * HEAD_DIM);

            // 只拷贝需要校验的行数，避免 PCIe 总线阻塞
            CHECK_CUDA(cudaMemcpy(h_O_gpu.data(), d_O, check_rows * HEAD_DIM * sizeof(float), cudaMemcpyDeviceToHost));

            cpu_attention_reference_partial(h_Q.data(), h_K.data(), h_V.data(), h_O_cpu.data(), N, HEAD_DIM, check_rows);

            float current_tolerance = kernel.is_halfacc ? 5e-2f : 1e-3f;

            if (!verify_result(h_O_gpu.data(), h_O_cpu.data(), check_rows, HEAD_DIM, current_tolerance)) {
                fprintf(stderr, "\n[ERROR] Numerical verification FAILED!\n");
                fprintf(stderr, "Sequence Length (N): %d\n", N);
                fprintf(stderr, "Kernel Implementation: %s\n", kernel.name.c_str());
                fprintf(stderr, "Status: Numerical deviation exceeds threshold.\n");
                
                if (kernel.is_multipass) {
                    if (d_S) cudaFree(d_S);
                    if (d_P) cudaFree(d_P);
                }
                cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_O);
                cudaFree(d_Q_half); cudaFree(d_K_half); cudaFree(d_V_half); cudaFree(d_O_half);
                cublasDestroy(handle);
                exit(1); 
            }

            if (kernel.is_multipass) {
                cudaFree(d_S);
                cudaFree(d_P);
            }
        }
    }

    // ==========================================
    // 5. 全局清理
    // ==========================================
    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_O);
    cudaFree(d_Q_half); cudaFree(d_K_half); cudaFree(d_V_half); cudaFree(d_O_half);
    cublasDestroy(handle);
    return 0;
}