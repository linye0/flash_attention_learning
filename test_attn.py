import torch
import my_flash_attn  
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
import os
import csv

# ==========================================
# 1. 评测超参数配置
# ==========================================
START_N = 1024
END_N = 62464
STEP_N = 4096
HEAD_DIM = 64

# ==========================================
# 2. 自适应测速函数 (Adaptive Benchmark)
# ==========================================
def benchmark_kernel(func, *args, **kwargs):
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    for _ in range(3):
        func(*args, **kwargs)
    torch.cuda.synchronize()

    start_event.record()
    func(*args, **kwargs)
    end_event.record()
    torch.cuda.synchronize()
    warmup_ms = start_event.elapsed_time(end_event)

    safe_warmup = max(warmup_ms, 0.005)
    repeats = int(200.0 / safe_warmup)
    repeats = max(3, min(100, repeats))

    start_event.record()
    for _ in range(repeats):
        func(*args, **kwargs)
    end_event.record()
    torch.cuda.synchronize()

    return start_event.elapsed_time(end_event) / repeats

def main():
    os.makedirs('result', exist_ok=True)
    csv_file_path = 'result/torch_benchmark_data.csv'
    
    print(f"Starting Benchmark... Data will be saved to {csv_file_path}\n")

    with open(csv_file_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Seq_Len(N)', 'KernelName', 'GFLOPS', 'Time(ms)'])

        # ====================================================
        # 阶段一：集中测试 V4_flash_wmma
        # ====================================================
        print("="*55)
        print("Phase 1: Benchmarking V4_flash_wmma")
        print("="*55)
        # 完美对齐表头
        print(f"{'Seq_Len(N)':<12}, {'KernelName':<20}, {'GFLOPS':>10}, {'Time(ms)':>10}")
        
        for N in range(START_N, END_N + 1, STEP_N):
            Q = torch.randn(1, 1, N, HEAD_DIM, dtype=torch.float16, device='cuda')
            K = torch.randn(1, 1, N, HEAD_DIM, dtype=torch.float16, device='cuda')
            V = torch.randn(1, 1, N, HEAD_DIM, dtype=torch.float16, device='cuda')
            ops = 4.0 * N * N * HEAD_DIM
            
            def run_my_v4():
                return my_flash_attn.run_v4(Q[0,0], K[0,0], V[0,0])
            
            v4_time = benchmark_kernel(run_my_v4)
            v4_gflops = ops / (v4_time * 1e-3) / 1e9
            
            kernel_name = "V4_flash_wmma"
            # 完美对齐数据行 (对应 C++ 的 %-12d, %-20s, %10.2f, %10.3f)
            print(f"{N:<12}, {kernel_name:<20}, {v4_gflops:10.2f}, {v4_time:10.3f}")
            writer.writerow([N, kernel_name, round(v4_gflops, 2), round(v4_time, 3)])
            f.flush()

        print("\n")

        # ====================================================
        # 阶段二：集中测试 Baseline_PyTorch
        # ====================================================
        print("="*55)
        print("Phase 2: Benchmarking PyTorch Official SDPA")
        print("="*55)
        print(f"{'Seq_Len(N)':<12}, {'KernelName':<20}, {'GFLOPS':>10}, {'Time(ms)':>10}")

        for N in range(START_N, END_N + 1, STEP_N):
            Q = torch.randn(1, 1, N, HEAD_DIM, dtype=torch.float16, device='cuda')
            K = torch.randn(1, 1, N, HEAD_DIM, dtype=torch.float16, device='cuda')
            V = torch.randn(1, 1, N, HEAD_DIM, dtype=torch.float16, device='cuda')
            ops = 4.0 * N * N * HEAD_DIM
            
            def run_official_sdpa():
                with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                    return F.scaled_dot_product_attention(Q, K, V)
                    
            official_time = benchmark_kernel(run_official_sdpa)
            official_gflops = ops / (official_time * 1e-3) / 1e9
            
            kernel_name = "Baseline_PyTorch"
            # 完美对齐数据行
            print(f"{N:<12}, {kernel_name:<20}, {official_gflops:10.2f}, {official_time:10.3f}")
            writer.writerow([N, kernel_name, round(official_gflops, 2), round(official_time, 3)])
            f.flush()

if __name__ == "__main__":
    dummy = torch.zeros(1024, device='cuda')
    for _ in range(5000): dummy += 1
    torch.cuda.synchronize()
    
    main()