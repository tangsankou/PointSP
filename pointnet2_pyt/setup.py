'''
Description: 
Autor: Jiachen Sun
Date: 2022-02-16 22:23:16
LastEditors: Jiachen Sun
LastEditTime: 2022-02-24 23:16:38
'''
from __future__ import division, absolute_import, with_statement, print_function
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import glob

try:
    import builtins
except:
    import __builtin__ as builtins

builtins.__POINTNET2_SETUP__ = True
import pointnet2

# _ext_src_root = "pointnet2/_ext-src"
# _ext_sources = glob.glob("{}/src/*.cpp".format(_ext_src_root)) + glob.glob(
#     "{}/src/*.cu".format(_ext_src_root)
# )
# _ext_headers = glob.glob("{}/include/*".format(_ext_src_root))

requirements = ["etw_pytorch_utils", "h5py", "enum34", "future"]

# setup(
#     name="pointnet2",
#     version=pointnet2.__version__,
#     author="Erik Wijmans",
#     packages=find_packages(),
#     install_requires=requirements,
#     # ext_modules=[
#     #     CUDAExtension(
#     #         name="pointnet2._ext",
#     #         sources=_ext_sources,
#     #         extra_compile_args={
#     #             "cxx": ["-O2", "-I{}".format("{}/include".format(_ext_src_root))],
#     #             "nvcc": ["-O2", "-I{}".format("{}/include".format(_ext_src_root))],
#     #         },
#     #     )
#     # ],
#     # cmdclass={"build_ext": BuildExtension},
# )

# # Copyright (c) Facebook, Inc. and its affiliates.
# # 
# # This source code is licensed under the MIT license found in the
# # LICENSE file in the root directory of this source tree.

# from setuptools import setup
# from torch.utils.cpp_extension import BuildExtension, CUDAExtension
# import glob
# import os

# _ext_src_root = "pointnet2/_ext-src"
# # _ext_src_root = "_ext_src"
# _ext_sources = glob.glob("{}/src/*.cpp".format(_ext_src_root)) + glob.glob(
#     "{}/src/*.cu".format(_ext_src_root)
# )
# _ext_headers = glob.glob("{}/include/*".format(_ext_src_root))

# headers = "-I" + os.path.join(os.path.dirname(os.path.abspath(__file__)), '_ext_src', 'include')

# setup(
#     name='pointnet2',
#     ext_modules=[
#         CUDAExtension(
#             name='pointnet2._ext',
#             sources=_ext_sources,
#             extra_compile_args={
#                 "cxx": ["-O2", headers],
#                 "nvcc": ["-O2", headers]
#             },
#         )
#     ],
#     cmdclass={
#         'build_ext': BuildExtension
#     }
# )

import glob
import os
import os.path as osp
from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

this_dir = osp.dirname(osp.abspath(__file__))
_ext_src_root = osp.join("pointnet2", "_ext-src")
_ext_sources = glob.glob(osp.join(_ext_src_root, "src", "*.cpp")) + glob.glob(
    osp.join(_ext_src_root, "src", "*.cu")
)
_ext_headers = glob.glob(osp.join(_ext_src_root, "include", "*"))

requirements = ["torch>=1.4"]

exec(open(osp.join("pointnet2", "_version.py")).read())

os.environ["TORCH_CUDA_ARCH_LIST"] = "3.7+PTX;5.0;6.0;6.1;6.2;7.0;7.5;8.0"
setup(
    name="pointnet2",
    version=__version__,
    author="Erik Wijmans",
    packages=find_packages(),
    install_requires=requirements,
    ext_modules=[
        CUDAExtension(
            name="pointnet2._ext",
            sources=_ext_sources,
            extra_compile_args={
                "cxx": ["-O3"],
                # "nvcc": ["-O3", "-Xfatbin", "-compress-all","-lineinfo"],
                "nvcc": ["-O3", "-Xfatbin", "-compress-all"],#'pointnet2_ops/_ext-src/src/sampling_gpu.cu'
            },
            include_dirs=[osp.join(this_dir, _ext_src_root, "include")],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    include_package_data=True,
)
