import glob
import os.path as osp
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

ROOT_DIR = osp.dirname(osp.abspath(__file__))
include_dirs = [osp.join(ROOT_DIR), "include"]

sources = glob.glob('*.cpp')+glob.glob('*.cu')


setup(
       name='cppcuda_tutorial',
       version='1.0',
       author='wenboji',
       author_email='wenboji0420@gmail.com',
       description='cppcuda example',
       long_description='example of accelerating pytorch with cppcuda',
       ext_modules=[
           CUDAExtension(
               name='cppcuda_tutorial',
               sources=sources,
               include_dirs=include_dirs,
               extra_compile_args={'cxx':['-02'],
                                    'nvcc':['-02']}
            )
        ],
       cmdclass={
           'build_ext':BuildExtension
        }
    )

