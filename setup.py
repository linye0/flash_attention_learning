from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='my_flash_attn',
    ext_modules=[
        CUDAExtension('my_flash_attn', [
            'wrapper.cpp',
            'attention.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)