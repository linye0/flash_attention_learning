# ===================================================
# 编译开关：运行 make USE_CUDNN=1 即可开启 cuDNN 基准测试
# ===================================================
USE_CUDNN ?= 0

NVCC = nvcc

# 注意：为了兼容 cudnn_frontend，建议将标准升级为 c++17
NVCC_FLAGS = -O3 -std=c++17 --use_fast_math --ptxas-options=-v -lcublas
ARCH_FLAGS = -gencode arch=compute_86,code=sm_86

# 动态注入 cuDNN 依赖
ifeq ($(USE_CUDNN), 1)
    NVCC_FLAGS += -DUSE_CUDNN -I./cudnn-frontend/include -lcudnn
    $(info >>> cuDNN Baseline is ENABLED)
else
    $(info >>> cuDNN Baseline is DISABLED)
endif

BUILD_DIR = build
RESULT_DIR = result
TARGET_NAME = attention
TARGET = $(BUILD_DIR)/$(TARGET_NAME)

SRCS = $(wildcard *.cu)
OBJS = $(patsubst %.cu, $(BUILD_DIR)/%.o, $(SRCS))

DATA_FILE = $(RESULT_DIR)/benchmark_data.csv
PLOT_FILE = $(RESULT_DIR)/performance_comparison.png
PLOT_SCRIPT = plot_benchmark.py

PROFILE_CSV = $(RESULT_DIR)/profile_log.csv
PROFILE_IMG = $(RESULT_DIR)/ncu_analysis.png
PROFILE_SCRIPT = plot_ncu.py 

# --- Nsight Compute 基础配置 ---
NCU_CMD      = sudo env "PATH=$(PATH)" ncu

NCU_COMMON   = --csv --log-file $(PROFILE_CSV) --force-overwrite \
               --kernel-name-base function \
               --clock-control none \
               -k "regex:attention"

NCU_FLAGS = $(NCU_COMMON) \
			--metric gpu__time_duration.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed,gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,sm__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_active,sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active,sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active,smsp__inst_executed_pipe_tensor.avg.pct_of_peak_sustained_active,smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct

all: $(TARGET)

$(TARGET): $(OBJS)
	@echo "Linking $@"
	$(NVCC) $(ARCH_FLAGS) $(NVCC_FLAGS) -o $@ $(OBJS)

$(BUILD_DIR)/%.o: %.cu | $(BUILD_DIR)
	@echo "Compiling $<"
	$(NVCC) $(ARCH_FLAGS) $(NVCC_FLAGS) -c $< -o $@

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

clean:
	rm -rf $(BUILD_DIR) 

run: $(TARGET)
	./$(TARGET)

plot: $(TARGET)
	@mkdir -p $(RESULT_DIR)
	@echo "1. Running Benchmark (This may take a while)..."
	stdbuf -oL ./$(TARGET) | tee $(DATA_FILE)
	@echo "   Data saved to $(DATA_FILE)"
	@echo "2. Generating Plot..."
	python3 $(PLOT_SCRIPT) $(DATA_FILE) $(PLOT_FILE)
	@echo "   Plot saved to $(PLOT_FILE)"
	@echo "Done!"

profile_pipe: $(TARGET)
	@mkdir -p $(RESULT_DIR)
	@echo "==================================================="
	@echo "Running PIPELINE Profile (LSU vs FMA)..."
	@echo "==================================================="
	$(NCU_CMD) $(NCU_FLAGS) ./$(TARGET)
	@echo "\n>>> Generating Plot..."
	@python3 $(PROFILE_SCRIPT)