from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


setup(
    name="my-flash-attn",
    version="0.3.0",
    description="Educational fused FlashAttention CUDA forward kernel",
    python_requires=">=3.10",
    ext_modules=[
        CUDAExtension(
            name="my_flash_attn",
            sources=["wrapper.cpp", "attention.cu"],
            libraries=["cublas"],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": ["-O3", "-std=c++17", "--use_fast_math", "-lineinfo"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension.with_options(no_python_abi_suffix=True)},
)
