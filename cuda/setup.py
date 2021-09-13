from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os
if 'CUDA_HOME' not in os.environ:
    os.environ['CUDA_HOME'] = '/usr/local/cuda'
    
setup(
    name='xccp_decode',
    ext_modules=[
        CUDAExtension('xccp_decode', [
            'xccp_decode.cpp',
            'xccp_decode_kernel.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })