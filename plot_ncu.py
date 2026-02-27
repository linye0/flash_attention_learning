import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import sys
import io
import os
import numpy as np

# ================= 配置区域 =================
INPUT_FILE = 'result/profile_log.csv'
OUTPUT_FILE = 'result/ncu_analysis.png'

# 想要绘制的算法列表 (模糊匹配：包含这些关键词的 Kernel 会被画出来)
TARGET_KERNELS = [] 
# ===========================================

def parse_ncu_csv(filepath):
    if not os.path.exists(filepath):
        print(f"Error: File '{filepath}' not found.")
        sys.exit(1)
    with open(filepath, 'r') as f:
        lines = f.readlines()
    csv_lines = [line for line in lines if not line.strip().startswith("==")]
    header_idx = -1
    for i, line in enumerate(csv_lines):
        if '"Metric Name"' in line and '"Metric Value"' in line:
            header_idx = i
            break
    if header_idx == -1:
        print("Error: Could not find valid CSV header.")
        sys.exit(1)
    csv_content = "".join(csv_lines[header_idx:])
    df = pd.read_csv(io.StringIO(csv_content), thousands=',') 
    df['Metric Value'] = pd.to_numeric(df['Metric Value'], errors='coerce')
    return df

def process_data(df):
    # 1. 提取序列长度 N (假设 Grid Size 的 X 维度对应 N/BlockSize)
    def parse_dim(dim_str):
        if not isinstance(dim_str, str): return 0
        m = re.search(r'\((\d+),', dim_str)
        return int(m.group(1)) if m else 0

    # 注意：在 Attention 项目中，我们的横轴应该是 Seq_Len (N)
    # 我们通过 Grid 尺寸和 Block 尺寸反推 N
    df['Grid_X'] = df['Grid Size'].apply(parse_dim)
    df['Block_X'] = df['Block Size'].apply(parse_dim)
    
    # 临时估算 Seq_Len (仅供参考，实际可根据你的 Kernel 逻辑调整)
    df['Seq_Len'] = df['Grid_X'] * df['Block_X']

    def clean_name(name):
        name = str(name).split('(')[0]
        # 移除一些通用前缀方便绘图
        return name.replace('launch_', '').replace('_kernel', '')
    df['Kernel Short'] = df['Kernel Name'].apply(clean_name)

    # 2. 模糊匹配过滤
    if TARGET_KERNELS:
        pattern = '|'.join(map(re.escape, TARGET_KERNELS))
        mask = df['Kernel Name'].str.contains(pattern, case=False, na=False)
        df = df[mask]

    # 3. 更新指标映射 (加入显存流量监控) [cite: 2026-01-19]
    metrics_map = {
        'gpu__time_duration.sum': 'Time_ns',
        'sm__throughput.avg.pct_of_peak_sustained_elapsed': 'SM_Efficiency',
        'gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed': 'DRAM_BW_Util',
        'l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum': 'Global_Read_Bytes',
        'sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active': 'FMA_Util',
        'smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct': 'Stall_SharedMem'
    }
    
    df_filtered = df[df['Metric Name'].isin(metrics_map.keys())].copy()
    df_filtered['Metric Name'] = df_filtered['Metric Name'].map(metrics_map)

    pivot_df = df_filtered.pivot_table(
        index=['Seq_Len', 'Kernel Short'], 
        columns='Metric Name', 
        values='Metric Value',
        aggfunc='mean'
    ).reset_index()

    # 4. 计算算术强度 (Arithmetic Intensity) [cite: 2026-01-19]
    # Attention FLOPs 约为 4 * N^2 * d (对于 V0 这种显式存 N^2 的情况)
    HEAD_DIM = 64 # 这里硬编码为你的 HEAD_DIM
    if 'Time_ns' in pivot_df.columns:
        pivot_df['Time_ns'] = pivot_df['Time_ns'].replace(0, np.nan)
        # 这里的 TFLOPS 估算基于标准 Attention 公式
        pivot_df['TFLOPS'] = (4.0 * pivot_df['Seq_Len']**2 * HEAD_DIM) / (pivot_df['Time_ns']) / 1000.0
    
    return pivot_df

def plot_analysis(df, output_path):
    sns.set_theme(style="whitegrid")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), sharex=True)

    # Part 1: TFLOPS (保持不变)
    sns.lineplot(data=df, x='Seq_Len', y='TFLOPS', hue='Kernel Short', marker='o', ax=ax1)
    ax1.set_title('Attention Performance: TFLOPS (Higher is Better)', fontweight='bold')

    # Part 2: 硬件瓶颈分析
    valid_metrics = ['DRAM_BW_Util', 'SM_Efficiency', 'Stall_SharedMem']
    # 检查这些列是否存在
    available_metrics = [m for m in valid_metrics if m in df.columns]
    
    # --- 修正点：显式命名 var_name 和 value_name ---
    df_melt = df.melt(id_vars=['Seq_Len', 'Kernel Short'], 
                      value_vars=available_metrics,
                      var_name='Metric',    # 显式命名为 Metric
                      value_name='Value')   # 显式命名为 Value
    
    # 绘图调用也要对应修改
    sns.lineplot(data=df_melt, x='Seq_Len', y='Value', 
                 hue='Kernel Short', style='Metric', ax=ax2)
    # ----------------------------------------------
    
    ax2.set_title('Hardware Bottleneck Analysis (Utilization %)', fontweight='bold')
    ax2.set_ylim(-5, 105)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)

if __name__ == "__main__":
    df = parse_ncu_csv(INPUT_FILE)
    df_clean = process_data(df)
    plot_analysis(df_clean, OUTPUT_FILE)