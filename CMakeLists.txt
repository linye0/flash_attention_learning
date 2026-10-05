cmake_minimum_required(VERSION 3.24)
project(flash_attention_lab LANGUAGES CXX CUDA)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

find_package(CUDAToolkit REQUIRED)
add_executable(attention_bench main.cu attention.cu reference.cu)
target_compile_features(attention_bench PRIVATE cxx_std_17 cuda_std_17)
target_compile_options(attention_bench PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-O3;--use_fast_math;--ptxas-options=-v>)
target_link_libraries(attention_bench PRIVATE CUDA::cublas)

include(CTest)
if(BUILD_TESTING)
  add_test(NAME smoke COMMAND attention_bench --n 1024 --kernel V4
           --target-ms 20 --max-repeats 20 --check-rows 16)
  set_tests_properties(smoke PROPERTIES LABELS "gpu")
  add_test(NAME v5_correctness COMMAND attention_bench --n 4096 --kernel V5
           --seed 2026 --target-ms 5 --max-repeats 3 --check-rows 64)
  set_tests_properties(v5_correctness PROPERTIES LABELS "gpu")
endif()
