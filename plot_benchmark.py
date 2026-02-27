import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys

# 默认路径
input_file = 'result/benchmark_data.csv'
output_file = 'result/performance_comparison.png'

# --- [新增] 配置需要画出的模型列表 ---
# 支持模糊匹配：只要 KernelName 包含列表中的关键词就匹配
# 例如：target_kernels = ["Double_Buffer", "TensorCore", "cuBLAS"]
# 如果列表为空 []，则默认画出 CSV 中的所有模型
target_kernels = [] 

if len(sys.argv) > 1:
    input_file = sys.argv[1]
if len(sys.argv) > 2:
    output_file = sys.argv[2]

# 1. 读取数据
try:
    df = pd.read_csv(input_file, skipinitialspace=True)
    df.columns = df.columns.str.strip()
    # 去除数据中 KernelName 列可能存在的空格
    df['KernelName'] = df['KernelName'].str.strip()
except Exception as e:
    print(f"Error reading CSV: {e}")
    exit(1)

# --- [核心修改] 模糊匹配过滤逻辑 ---
if target_kernels:
    # 构建正则表达式：匹配包含任意一个关键词的行（不区分大小写）
    # 例如：["Double_Buffer", "cuBLAS"] -> "Double_Buffer|cuBLAS"
    pattern = '|'.join(target_kernels)
    
    # 使用 str.contains 进行模糊匹配，case=False 表示不区分大小写
    mask = df['KernelName'].str.contains(pattern, case=False, na=False)
    df = df[mask]
    
    if df.empty:
        print(f"Warning: No data found for kernels matching {target_kernels}")
        print(f"Available kernels: {df['KernelName'].unique().tolist()}")
        exit(1)
    
    print(f"Matched kernels: {df['KernelName'].unique().tolist()}")

# 2. 设置绘图风格
sns.set_theme(style="whitegrid")
plt.figure(figsize=(12, 8))

# 3. 画折线图
try:
    sns.lineplot(data=df, x="Seq_Len(N)", y="Time(ms)", hue="KernelName", 
                 style="KernelName", markers=True, dashes=False, linewidth=2.5)
except Exception as e:
    print(f"Error plotting data: {e}")
    exit(1)

# 4. 设置标题和标签
plt.title("Matrix Multiplication Performance Comparison", fontsize=16)
plt.xlabel("Seq_Len (N)", fontsize=12)
plt.ylabel("Performance (ms)", fontsize=12)
plt.legend(title="Kernel Implementation", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()

# 5. 保存图片
plt.savefig(output_file, dpi=300)
print(f"Plot saved to {output_file}")