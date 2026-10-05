CUDA_HOME ?= /usr/local/cuda
NVCC ?= $(CUDA_HOME)/bin/nvcc
NCU ?= $(CUDA_HOME)/bin/ncu
PYTHON ?= python3
CUDA_ARCH ?= 86
USE_CUDNN ?= 0

BUILD_DIR := build
RESULT_DIR := result
TARGET := $(BUILD_DIR)/attention_bench
SOURCES := main.cu attention.cu reference.cu
OBJECTS := $(SOURCES:%.cu=$(BUILD_DIR)/%.o)

NVCCFLAGS ?= -O3 -std=c++17 --use_fast_math --ptxas-options=-v
ARCH_FLAGS := -gencode arch=compute_$(CUDA_ARCH),code=sm_$(CUDA_ARCH)
CPPFLAGS :=
LDLIBS := -lcublas

ifeq ($(USE_CUDNN),1)
CPPFLAGS += -DUSE_CUDNN -I./cudnn-frontend/include
LDLIBS += -lcudnn
endif

.DEFAULT_GOAL := all

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(NVCC) $(ARCH_FLAGS) $(NVCCFLAGS) $^ -o $@ $(LDLIBS)

$(BUILD_DIR)/%.o: %.cu attention.h | $(BUILD_DIR)
	$(NVCC) $(ARCH_FLAGS) $(NVCCFLAGS) $(CPPFLAGS) -c $< -o $@

$(BUILD_DIR):
	mkdir -p $@

run: $(TARGET)
	./$(TARGET)

smoke: $(TARGET)
	./$(TARGET) --n 1024 --kernel V4 --target-ms 20 --max-repeats 20 --check-rows 16

test-stable: $(TARGET)
	./$(TARGET) --n 1024 --target-ms 20 --max-repeats 20 --check-rows 16

test-v5: $(TARGET)
	./$(TARGET) --n 64 --seed 1 --kernel V5 --target-ms 5 --max-repeats 3 --check-rows 64
	./$(TARGET) --n 256 --seed 7 --kernel V5 --target-ms 5 --max-repeats 3 --check-rows 64
	./$(TARGET) --n 1024 --seed 42 --kernel V5 --target-ms 5 --max-repeats 3 --check-rows 64
	./$(TARGET) --n 4096 --seed 2026 --kernel V5 --target-ms 5 --max-repeats 3 --check-rows 64

benchmark: $(TARGET)
	mkdir -p $(RESULT_DIR)
	./$(TARGET) --min-n 1024 --max-n 62464 --step 4096 --kernel V4 \
		--output $(RESULT_DIR)/benchmark_data.csv

plot: benchmark
	$(PYTHON) plot_benchmark.py $(RESULT_DIR)/benchmark_data.csv $(RESULT_DIR)/performance_comparison.png

profile: $(TARGET)
	mkdir -p $(RESULT_DIR)
	$(NCU) --kernel-name-base demangled --filter-mode per-launch-config \
		--section SpeedOfLight --section LaunchStats --section Occupancy \
		--section ComputeWorkloadAnalysis --section MemoryWorkloadAnalysis_Tables \
		--export $(RESULT_DIR)/attention_profile --force-overwrite \
		./$(TARGET) --n 8192 --kernel V4 --target-ms 20 --max-repeats 3 --no-validate

extension:
	$(PYTHON) setup.py build_ext --inplace

test-python: extension
	$(PYTHON) -m pytest -q tests

clean:
	rm -rf $(BUILD_DIR)

.PHONY: all run smoke test-stable test-v5 benchmark plot profile extension test-python clean
