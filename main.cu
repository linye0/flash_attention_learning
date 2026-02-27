#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "attention.h"

#define START_N 1024
#define END_N 46080
#define STEP_N 4096
#define HEAD_DIM 64

#define CHECK_CUDA(call) { \
    const cudaError_t error = call; \
    if (error != cudaSuccess) { \
        printf("Error: %s:%d, code: %d, reason: %s\n", __FILE__, __LINE__, error, cudaGetErrorString(error)); \
        exit(1); \
    } \
}

bool verify_result(const float* gpu_res, const float* cpu_res, int check_rows, int d) {
    for (int i = 0; i < check_rows * d; ++i) {
        float diff = std::abs(gpu_res[i] - cpu_res[i]);
        // 使用相对误差来评估
        float rel_diff = diff / (std::abs(cpu_res[i]) + 1e-5f);
        if (rel_diff > 1e-3f) { // 阈值设为 0.1%，对于 FP32 算子是合理的
            printf("Error at index %d: GPU=%f, CPU=%f, RelDiff=%f\n", i, gpu_res[i], cpu_res[i], rel_diff);
            return false;
        }
    }
    return true;
}

int main() {
    srand(time(NULL));
    printf("%-12s, %-20s, %10s, %10s\n", "Seq_Len(N)", "KernelName", "GFLOPS", "Time(ms)");

    cublasHandle_t handle;
    cublasCreate(&handle);
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    int max_N = END_N;
    size_t max_bytes_QKV = (size_t)max_N * HEAD_DIM * sizeof(float);

    // 1. 分配 Q, K, V, O (这些是 O(N)，不会轻易爆显存)
    // Q, K, V, O的最大shape都是(maxN, HEAD_DIM), maxN是最大序列长度，HEAD_DIM是单个注意力头的维度
    std::vector<float> h_Q(max_N * HEAD_DIM);
    std::vector<float> h_K(max_N * HEAD_DIM);
    std::vector<float> h_V(max_N * HEAD_DIM);

    // 填充随机数 (避免全 0 导致计算逻辑被硬件优化跳过)
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

    std::vector<KernelInfo> kernels = get_kernels();

    // 2. 开始 N 的爬升循环
    for (int N = START_N; N <= END_N; N += STEP_N) {

        for (const auto& kernel : kernels) {
            float* d_S = nullptr, *d_P = nullptr;

            if (kernel.is_multipass) {
                size_t bytes_NxN = (size_t)N * N * sizeof(float);

                cudaGetLastError();

                cudaError_t err1 = cudaMalloc(&d_S, bytes_NxN);
                cudaError_t err2 = cudaMalloc(&d_P, bytes_NxN);

                if (err1 != cudaSuccess || err2 != cudaSuccess) {
                    if (d_S) cudaFree(d_S);
                    if (d_P) cudaFree(d_P);

                    printf("%-12d, %-20s, %10s, %10s\n", N, kernel.name.c_str(), "OOM", "-");

                    cudaGetLastError();
                    continue; // 跳过这个爆了的 V0，继续测下一个（比如 V1）
                }
            }   

            // 热身
            kernel.func(handle, d_Q, d_K, d_V, d_O, d_S, d_P, N, HEAD_DIM);
            cudaDeviceSynchronize();

            cudaEventRecord(start);
            int REPEAT = 5;
            for(int r = 0; r < REPEAT; ++r) {
                kernel.func(handle, d_Q, d_K, d_V, d_O, d_S, d_P, N, HEAD_DIM);
            }
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);

            float milliseconds = 0;
            cudaEventElapsedTime(&milliseconds, start, stop);
            float avg_time = milliseconds / REPEAT;

            double gflops = (4.0 * N * N * HEAD_DIM) / (avg_time * 1e-3) / 1e9;

            printf("%-12d, %-20s, %10.2f, %10.3f\n", N, kernel.name.c_str(), gflops, avg_time);

            fflush(stdout);

            // --- 校验逻辑开始 ---
            int check_rows = std::min(N, 100); // 统一校验前 100 行，如果 N 小于 100 则校验全量
            std::vector<float> h_O_gpu(N * HEAD_DIM);
            std::vector<float> h_O_cpu(check_rows * HEAD_DIM);

            // 1. 将 GPU 计算出的最终结果 O 拷回 Host
            CHECK_CUDA(cudaMemcpy(h_O_gpu.data(), d_O, N * HEAD_DIM * sizeof(float), cudaMemcpyDeviceToHost));

            // 2. 调用我们在 reference.cpp 里定义的局部校验函数
            cpu_attention_reference_partial(h_Q.data(), h_K.data(), h_V.data(), h_O_cpu.data(), N, HEAD_DIM, check_rows);

            // 3. 执行比对并打印结果
            if (!verify_result(h_O_gpu.data(), h_O_cpu.data(), check_rows, HEAD_DIM)) {
                fprintf(stderr, "\n[ERROR] Numerical verification FAILED!\n");
                fprintf(stderr, "Sequence Length (N): %d\n", N);
                fprintf(stderr, "Kernel Implementation: %s\n", kernel.name.c_str());
                fprintf(stderr, "Status: Numerical deviation exceeds threshold (1e-3).\n");
                
                // 彻底清理资源以防显存残留
                if (kernel.is_multipass) {
                    if (d_S) cudaFree(d_S);
                    if (d_P) cudaFree(d_P);
                }
                cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_O);
                cublasDestroy(handle);
                
                exit(1); // 立即退出程序 
            }
            // --- 校验逻辑结束 ---

            if (kernel.is_multipass) {
                cudaFree(d_S);
                cudaFree(d_P);
            }
        }
    }

    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_O);
    cublasDestroy(handle);
    return 0;
}