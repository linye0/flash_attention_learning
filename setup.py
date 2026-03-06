from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='my_flash_attn',
    ext_modules=[
        CUDAExtension('my_flash_attn', [
            'wrapper.cpp',
            'attention.cu', # 你呕心沥血写出的内核文件
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)