#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "attention.h"

namespace {

struct Options {
    int min_n = 1024;
    int max_n = 8192;
    int step = 1024;
    int exact_n = 0;
    int head_dim = 64;
    int warmup = 2;
    int check_rows = 32;
    int min_repeats = 3;
    int max_repeats = 200;
    int seed = 42;
    float target_ms = 200.0f;
    bool validate = true;
    bool list_kernels = false;
    bool include_experimental = false;
    std::string kernel_filter;
    std::string output_path;
};

struct ErrorStats {
    int mismatches = 0;
    float max_abs = 0.0f;
    float max_rel = 0.0f;
};

#define CHECK_CUDA(call) do { \
    const cudaError_t status = (call); \
    if (status != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ':' << __LINE__ \
                  << ": " << cudaGetErrorString(status) << '\n'; \
        std::exit(EXIT_FAILURE); \
    } \
} while (0)

#define CHECK_CUBLAS(call) do { \
    const cublasStatus_t status = (call); \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        std::cerr << "cuBLAS error " << status << " at " << __FILE__ << ':' << __LINE__ << '\n'; \
        std::exit(EXIT_FAILURE); \
    } \
} while (0)

int positive_int(const char* option, const char* text) {
    char* end = nullptr;
    const long value = std::strtol(text, &end, 10);
    if (*text == '\0' || *end != '\0' || value <= 0 || value > std::numeric_limits<int>::max()) {
        std::cerr << "Invalid value for " << option << ": " << text << '\n';
        std::exit(EXIT_FAILURE);
    }
    return static_cast<int>(value);
}

float positive_float(const char* option, const char* text) {
    char* end = nullptr;
    const float value = std::strtof(text, &end);
    if (*text == '\0' || *end != '\0' || value <= 0.0f) {
        std::cerr << "Invalid value for " << option << ": " << text << '\n';
        std::exit(EXIT_FAILURE);
    }
    return value;
}

void print_help(const char* program) {
    std::cout
        << "FlashAttention kernel benchmark and validator\n\n"
        << "Usage: " << program << " [options]\n\n"
        << "Shapes:\n"
        << "  --n N                  Benchmark one sequence length\n"
        << "  --min-n N              First sweep length (default: 1024)\n"
        << "  --max-n N              Last sweep length (default: 8192)\n"
        << "  --step N               Sweep increment (default: 1024)\n"
        << "  --head-dim D           Head dimension; currently 64 only\n\n"
        << "Measurement:\n"
        << "  --warmup N             Warm-up launches (default: 2)\n"
        << "  --target-ms MS         Adaptive timing window (default: 200)\n"
        << "  --min-repeats N        Lower repeat bound (default: 3)\n"
        << "  --max-repeats N        Upper repeat bound (default: 200)\n"
        << "  --check-rows N         Rows checked against CPU (default: 32)\n"
        << "  --kernel TEXT          Filter kernel names by substring\n"
        << "  --include-experimental Include V5/decoding prototypes\n"
        << "  --no-validate          Skip numerical validation\n"
        << "  --output FILE          Write CSV to FILE instead of stdout\n"
        << "  --seed N               Deterministic input seed (default: 42)\n"
        << "  --list-kernels         List implementations and exit\n"
        << "  --help                  Show this help\n";
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        const auto value = [&](const char* name) -> const char* {
            if (i + 1 >= argc) {
                std::cerr << "Missing value for " << name << '\n';
                std::exit(EXIT_FAILURE);
            }
            return argv[++i];
        };
        if (arg == "--n") options.exact_n = positive_int(arg.c_str(), value(arg.c_str()));
        else if (arg == "--min-n") options.min_n = positive_int(arg.c_str(), value(arg.c_str()));
        else if (arg == "--max-n") options.max_n = positive_int(arg.c_str(), value(arg.c_str()));
        else if (arg == "--step") options.step = positive_int(arg.c_str(), value(arg.c_str()));
        else if (arg == "--head-dim") options.head_dim = positive_int(arg.c_str(), value(arg.c_str()));
        else if (arg == "--warmup") options.warmup = positive_int(arg.c_str(), value(arg.c_str()));
        else if (arg == "--check-rows") options.check_rows = positive_int(arg.c_str(), value(arg.c_str()));
        else if (arg == "--min-repeats") options.min_repeats = positive_int(arg.c_str(), value(arg.c_str()));
        else if (arg == "--max-repeats") options.max_repeats = positive_int(arg.c_str(), value(arg.c_str()));
        else if (arg == "--seed") options.seed = positive_int(arg.c_str(), value(arg.c_str()));
        else if (arg == "--target-ms") options.target_ms = positive_float(arg.c_str(), value(arg.c_str()));
        else if (arg == "--kernel") options.kernel_filter = value(arg.c_str());
        else if (arg == "--output") options.output_path = value(arg.c_str());
        else if (arg == "--include-experimental") options.include_experimental = true;
        else if (arg == "--no-validate") options.validate = false;
        else if (arg == "--list-kernels") options.list_kernels = true;
        else if (arg == "--help" || arg == "-h") {
            print_help(argv[0]);
            std::exit(EXIT_SUCCESS);
        } else {
            std::cerr << "Unknown option: " << arg << "\nUse --help for usage.\n";
            std::exit(EXIT_FAILURE);
        }
    }
    if (options.head_dim != 64) {
        std::cerr << "Current kernels require --head-dim 64.\n";
        std::exit(EXIT_FAILURE);
    }
    if (options.min_n > options.max_n || options.min_repeats > options.max_repeats) {
        std::cerr << "Invalid range in benchmark options.\n";
        std::exit(EXIT_FAILURE);
    }
    return options;
}

std::vector<int> sequence_lengths(const Options& options) {
    if (options.exact_n > 0) return {options.exact_n};
    std::vector<int> values;
    for (int n = options.min_n; n <= options.max_n; n += options.step) {
        values.push_back(n);
        if (n > options.max_n - options.step) break;
    }
    return values;
}

ErrorStats compare(const float* actual, const float* expected, size_t count, float atol, float rtol) {
    ErrorStats stats;
    for (size_t i = 0; i < count; ++i) {
        const float abs_error = std::abs(actual[i] - expected[i]);
        const float rel_error = abs_error / (std::abs(expected[i]) + 1e-6f);
        stats.max_abs = std::max(stats.max_abs, abs_error);
        stats.max_rel = std::max(stats.max_rel, rel_error);
        if (abs_error > atol + rtol * std::abs(expected[i])) ++stats.mismatches;
    }
    return stats;
}

__global__ void float_to_half(const float* source, half* destination, size_t count) {
    const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) destination[index] = __float2half(source[index]);
}

__global__ void half_to_float(const half* source, float* destination, size_t count) {
    const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) destination[index] = __half2float(source[index]);
}

float measure_once(const KernelInfo& kernel, cublasHandle_t handle,
                   const void* q, const void* k, const void* v, void* output,
                   float* scores, float* probabilities, int n, int d,
                   cudaEvent_t start, cudaEvent_t stop) {
    CHECK_CUDA(cudaEventRecord(start));
    kernel.func(handle, q, k, v, output, scores, probabilities, n, d);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float milliseconds = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
    return milliseconds;
}

}  // namespace

int main(int argc, char** argv) {
    const Options options = parse_options(argc, argv);
    const std::vector<KernelInfo> registry = get_kernels();

    if (options.list_kernels) {
        for (const auto& kernel : registry) {
            std::cout << kernel.name;
            if (kernel.experimental) std::cout << " [experimental]";
            if (kernel.is_halfacc) std::cout << " [fp16]";
            if (kernel.is_decoding) std::cout << " [decode]";
            std::cout << '\n';
        }
        return EXIT_SUCCESS;
    }

    std::vector<KernelInfo> kernels;
    for (const auto& kernel : registry) {
        const bool name_match = options.kernel_filter.empty() || kernel.name.find(options.kernel_filter) != std::string::npos;
        if (name_match && (options.include_experimental || !kernel.experimental)) kernels.push_back(kernel);
    }
    if (kernels.empty()) {
        std::cerr << "No kernel matched the requested filter/stability level.\n";
        return EXIT_FAILURE;
    }

    const std::vector<int> lengths = sequence_lengths(options);
    for (const int n : lengths) {
        if (n % 64 != 0) {
            std::cerr << "Sequence lengths must be multiples of 64; got " << n << ".\n";
            return EXIT_FAILURE;
        }
    }
    const int max_n = *std::max_element(lengths.begin(), lengths.end());
    const size_t max_elements = static_cast<size_t>(max_n) * options.head_dim;

    cudaDeviceProp device{};
    CHECK_CUDA(cudaGetDeviceProperties(&device, 0));
    std::cerr << "GPU: " << device.name << " (sm_" << device.major << device.minor << ")\n"
              << "Selected kernels: " << kernels.size() << ", shapes: " << lengths.size()
              << ", validation: " << (options.validate ? "on" : "off") << '\n';

    std::ofstream output_file;
    std::ostream* output = &std::cout;
    if (!options.output_path.empty()) {
        output_file.open(options.output_path);
        if (!output_file) {
            std::cerr << "Cannot open output file: " << options.output_path << '\n';
            return EXIT_FAILURE;
        }
        output = &output_file;
    }
    *output << "SequenceLength,HeadDim,KernelName,DType,GFLOPS,TimeMs,Repeats,Validated\n";

    std::mt19937 generator(options.seed);
    std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
    std::vector<float> h_q(max_elements), h_k(max_elements), h_v(max_elements);
    for (float& value : h_q) value = distribution(generator);
    for (float& value : h_k) value = distribution(generator);
    for (float& value : h_v) value = distribution(generator);

    float *d_q = nullptr, *d_k = nullptr, *d_v = nullptr, *d_output = nullptr;
    half *d_q_half = nullptr, *d_k_half = nullptr, *d_v_half = nullptr, *d_output_half = nullptr;
    CHECK_CUDA(cudaMalloc(&d_q, max_elements * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_k, max_elements * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_v, max_elements * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_output, max_elements * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_q_half, max_elements * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_k_half, max_elements * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_v_half, max_elements * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_output_half, max_elements * sizeof(half)));

    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    for (const int n : lengths) {
        const size_t elements = static_cast<size_t>(n) * options.head_dim;
        CHECK_CUDA(cudaMemcpy(d_q, h_q.data(), elements * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_k, h_k.data(), elements * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_v, h_v.data(), elements * sizeof(float), cudaMemcpyHostToDevice));
        const int conversion_blocks = static_cast<int>((elements + 255) / 256);
        float_to_half<<<conversion_blocks, 256>>>(d_q, d_q_half, elements);
        float_to_half<<<conversion_blocks, 256>>>(d_k, d_k_half, elements);
        float_to_half<<<conversion_blocks, 256>>>(d_v, d_v_half, elements);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());

        const int checked_rows = std::min(n, options.check_rows);
        std::vector<float> reference(static_cast<size_t>(checked_rows) * options.head_dim);
        if (options.validate) {
            cpu_attention_reference_partial(h_q.data(), h_k.data(), h_v.data(), reference.data(),
                                            n, options.head_dim, checked_rows);
        }

        for (const auto& kernel : kernels) {
            float* scores = nullptr;
            float* probabilities = nullptr;
            if (kernel.is_multipass) {
                const size_t matrix_bytes = static_cast<size_t>(n) * n * sizeof(float);
                cudaError_t first = cudaMalloc(&scores, matrix_bytes);
                cudaError_t second = cudaMalloc(&probabilities, matrix_bytes);
                if (first != cudaSuccess || second != cudaSuccess) {
                    if (scores) cudaFree(scores);
                    if (probabilities) cudaFree(probabilities);
                    cudaGetLastError();
                    std::cerr << "SKIP " << kernel.name << " @ N=" << n << " (O(N^2) workspace OOM)\n";
                    continue;
                }
            }

            const void* launch_q = kernel.is_halfacc ? static_cast<const void*>(d_q_half) : static_cast<const void*>(d_q);
            const void* launch_k = kernel.is_halfacc ? static_cast<const void*>(d_k_half) : static_cast<const void*>(d_k);
            const void* launch_v = kernel.is_halfacc ? static_cast<const void*>(d_v_half) : static_cast<const void*>(d_v);
            void* launch_output = kernel.is_halfacc ? static_cast<void*>(d_output_half) : static_cast<void*>(d_output);

            for (int i = 0; i < options.warmup; ++i) {
                kernel.func(handle, launch_q, launch_k, launch_v, launch_output,
                            scores, probabilities, n, options.head_dim);
            }
            CHECK_CUDA(cudaGetLastError());
            CHECK_CUDA(cudaDeviceSynchronize());

            const float probe_ms = measure_once(kernel, handle, launch_q, launch_k, launch_v,
                                                launch_output, scores, probabilities,
                                                n, options.head_dim, start, stop);
            const int repeats = std::max(options.min_repeats,
                std::min(options.max_repeats, static_cast<int>(options.target_ms / std::max(probe_ms, 0.005f))));

            CHECK_CUDA(cudaEventRecord(start));
            for (int i = 0; i < repeats; ++i) {
                kernel.func(handle, launch_q, launch_k, launch_v, launch_output,
                            scores, probabilities, n, options.head_dim);
            }
            CHECK_CUDA(cudaGetLastError());
            CHECK_CUDA(cudaEventRecord(stop));
            CHECK_CUDA(cudaEventSynchronize(stop));
            float elapsed_ms = 0.0f;
            CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start, stop));
            const float average_ms = elapsed_ms / repeats;
            const double operations = kernel.is_decoding
                ? 4.0 * n * options.head_dim
                : 4.0 * n * n * options.head_dim;
            const double performance = operations / (average_ms * 1e-3) / 1e9;

            bool validated = false;
            if (options.validate) {
                if (kernel.is_halfacc) {
                    half_to_float<<<conversion_blocks, 256>>>(d_output_half, d_output, elements);
                    CHECK_CUDA(cudaGetLastError());
                }
                const int validation_rows = kernel.is_decoding ? 1 : checked_rows;
                std::vector<float> actual(static_cast<size_t>(validation_rows) * options.head_dim);
                CHECK_CUDA(cudaMemcpy(actual.data(), d_output, actual.size() * sizeof(float), cudaMemcpyDeviceToHost));
                const float atol = kernel.is_halfacc ? 5e-2f : 1e-3f;
                const float rtol = kernel.is_halfacc ? 5e-2f : 1e-3f;
                const ErrorStats stats = compare(actual.data(), reference.data(), actual.size(), atol, rtol);
                validated = stats.mismatches == 0;
                std::cerr << (validated ? "PASS " : "FAIL ") << kernel.name << " @ N=" << n
                          << " max_abs=" << stats.max_abs << " max_rel=" << stats.max_rel << '\n';
                if (!validated) return EXIT_FAILURE;
            }

            *output << n << ',' << options.head_dim << ',' << kernel.name << ','
                    << (kernel.is_halfacc ? "FP16" : "FP32") << ','
                    << performance << ',' << average_ms << ',' << repeats << ','
                    << (options.validate ? (validated ? "true" : "false") : "skipped") << '\n';
            output->flush();
            if (scores) CHECK_CUDA(cudaFree(scores));
            if (probabilities) CHECK_CUDA(cudaFree(probabilities));
        }
    }

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUBLAS(cublasDestroy(handle));
    CHECK_CUDA(cudaFree(d_q));
    CHECK_CUDA(cudaFree(d_k));
    CHECK_CUDA(cudaFree(d_v));
    CHECK_CUDA(cudaFree(d_output));
    CHECK_CUDA(cudaFree(d_q_half));
    CHECK_CUDA(cudaFree(d_k_half));
    CHECK_CUDA(cudaFree(d_v_half));
    CHECK_CUDA(cudaFree(d_output_half));
    return EXIT_SUCCESS;
}
