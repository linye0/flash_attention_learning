NVCC = nvcc

NVCC_FLAGS = -O3 -std=c++14 --ptxas-options=-v -lcublas

ARCH_FLAGS = -gencode arch=compute_86,code=sm_86

BUILD_DIR = build
RESULT_DIR = result
TARGET_NAME = attention
TARGET = $(BUILD_DIR)/$(TARGET_NAME)

DATA_FILE = $(RESULT_DIR)/benchmark_data.csv
PLOT_FILE = $(RESULT_DIR)/performance_comparison.png
PLOT_SCRIPT = plot_benchmark.py

SRCS = $(wildcard *.cu *.cpp)
OBJS = $(patsubst %.cu, $(BUILD_DIR)/%.o, $(SRCS))

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
	rm -rf $(BUILD_DIR) $(RESULT_DIR)

run: $(TARGET)
	./$(TARGET)

plot: $(TARGET)
	@mkdir -p $(RESULT_DIR)
	@echo "1. Running Benchmark (This may take a while)..."
	./$(TARGET) > $(DATA_FILE)
	@echo "   Data saved to $(DATA_FILE)"
	@echo "2. Generating Plot..."
	python3 $(PLOT_SCRIPT) $(DATA_FILE) $(PLOT_FILE)
	@echo "   Plot saved to $(PLOT_FILE)"
	@echo "Done!"