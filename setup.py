import os
import glob

from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

this_dir = os.path.dirname(os.path.abspath(__file__))
extensions_dir = os.path.join(this_dir, "csrc")

main_source = glob.glob(os.path.join(extensions_dir, "*.cpp"))
source_cpu = glob.glob(os.path.join(extensions_dir, "cpu", "*.cpp"))

setup(
    name='torch_sampling',
    author="LeviViana",
    description="Efficient random sampling extension for Pytorch",
    ext_modules=[CppExtension(
            'torch_sampling',
            main_source + source_cpu,
            include_dirs=[extensions_dir],
    )],
    cmdclass={'build_ext': BuildExtension},
)
