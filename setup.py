import os
import glob

import torch
from setuptools import setup
from packaging import version
from torch.utils.cpp_extension import CUDA_HOME
from torch.utils.cpp_extension import CppExtension
from torch.utils.cpp_extension import CUDAExtension
from torch.utils.cpp_extension import BuildExtension

this_dir = os.path.dirname(os.path.abspath(__file__))
extensions_dir = os.path.join(this_dir, "csrc")

main_source = glob.glob(os.path.join(extensions_dir, "*.cpp"))
source_cpu = glob.glob(os.path.join(extensions_dir, "cpu", "*.cpp"))
source_cuda = glob.glob(os.path.join(extensions_dir, "cuda", "*.cu"))

sources = main_source + source_cpu
extension = CppExtension
define_macros = []

if torch.cuda.is_available() and CUDA_HOME is not None:
    extension = CUDAExtension
    sources += source_cuda
    define_macros += [("WITH_CUDA", None)]

if version.parse(torch.__version__) >= version.parse('1.6.0'):
    define_macros += [("TORCH_1_6", None)]

setup(
    name='torch_sampling',
    author="LeviViana",
    description="Efficient random sampling extension for Pytorch",
    ext_modules=[extension(
                    'torch_sampling',
                    sources,
                    define_macros=define_macros,
                    include_dirs=[extensions_dir],
                )],
    cmdclass={'build_ext': BuildExtension},
)
