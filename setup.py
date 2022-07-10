from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

setup(
       name='cppcuda_tutorial',
       version='1.0',
       author='wenboji',
       author_email='wenboji0420@gmail.com',
       description='cppcuda example',
       long_description='example of accelerating pytorch with cppcuda',
       ext_modules=[
           CppExtension(
               name='cppcuda_tutorial',
               sources=['interpolation.cpp']
            )
        ],
       cmdclass={
           'build_ext':BuildExtension
        }
    )

