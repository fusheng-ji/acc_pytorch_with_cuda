# Accelerate pytorch with cuda and c++

## General use case

1. lots of sequential computation

   ```
   x = f1(x)
   
   x = f2(x)
   
   ...
   
   x = fn(x)
   ```

2. non-parallel computation

   e.g. For each batch, do an operation that depends on the data length (e.g. volume rendering)

## How it work

```mermaid
graph LR
1[pytorch]-->2[c++]-->3[cuda]
3-->2-->1
```

Combine together all the C++ and CUDA files we'll need and use PyBind11 to build the interface we want; 

Fortunately, [PyBind11](https://github.com/pybind/pybind11) is included with Pytorch.

## Example: Trilinear interpolation

### Writing a C++ Extension

C++ extensions come in two flavors: They can be built “ahead of time” with `setuptools`, or “just in time” via `torch.utils.cpp_extension.load()`. We’ll begin with the first approach and discuss the latter later.

```mermaid
graph LR
5["C++ extensions"]-->1
5-->2
1[ahead of time]-->3[setuptools]
2[just in time]-->4["torch.utils.cpp_extension.load()"]
```

#### Building with `setuptools`

For the “ahead of time” flavor, we build our C++ extension by writing a `setup.py` script that uses `setuptools` to compile our C++ code. 

[setup.py](./setup.py)

```python
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
```

`CppExtension` is a convenience wrapper around `setuptools.Extension` that passes the correct include paths and sets the language of the extension to C++. 

`BuildExtension` performs a number of required configuration steps and checks and also manages mixed compilation in the case of mixed C++/CUDA extensions. And that’s all we really need to know about building C++ extensions for now! 

Let’s now take a look at the implementation of our C++ extension, which goes into `interpolation.cpp`.

#### Writing the C++ Op

[interpolation.cpp](./interpolation.cpp)

```c++
#include <torch/extension.h>
#include <pybind11/pybind11.h>

torch::Tensor trilinear_interpolation(
		torch::Tensor feats,
		torch::Tensor point){
	return feats;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
	m.def("trilinear_interpolation", &trilinear_interpolation, R"pbdoc(
		m.def("name_in_python",&name_in_cpp)
		)pbdoc");
}

```

`<torch/extension.h>` is the one-stop header to include all the necessary PyTorch bits to write C++ extensions. It includes:

- The ATen library, which is our primary API for tensor computation,
- [pybind11](https://github.com/pybind/pybind11), which is how we create Python bindings for our C++ code,
- Headers that manage the details of interaction between ATen and pybind11.

#### Binding to Python

Once you have your operation written in C++ and ATen, you can use pybind11 to bind your C++ functions or classes into Python in a very simple manner. Questions or issues you have about this part of PyTorch C++ extensions will largely be addressed by [pybind11 documentation](https://pybind11.readthedocs.io/en/stable/).

For our extensions, the necessary binding code spans only four lines:

```c++
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
	m.def("trilinear_interpolation", &trilinear_interpolation, R"pbdoc(
		m.def("name_in_python",&name_in_cpp)
		)pbdoc");
}
```

One bit to note here is the macro `TORCH_EXTENSION_NAME`. The torch extension build will define it as the name you give your extension in the `setup.py` script. In this case, the value of `TORCH_EXTENSION_NAME` would be “interpolation_cpp”. This is to avoid having to maintain the name of the extension in two places (the build script and your C++ code), as a mismatch between the two can lead to nasty and hard to track issues.

#### Using Your Extension

Put the `interpolation.cpp` and `setup.py` under same dir. Then run `python setup.py install`  to build and install your extension. 

Once your extension is built, you can simply import it in Python, using the name you specified in your `setup.py` script. Just be sure to `import torch` first, as this will resolve some symbols that the dynamic linker must see:

[test.py](./test.py)

```python
import torch
import cppcuda_tutorial # import your installed c++ extension

feats = torch.ones(2)
point = torch.zeros(2)

out = cppcuda_tutorial.trilinear_interpolation(feats, point)

print(out)
```

Now we test if the extension can run correctly.

run `python test.py`. This should look something like this:

```python
tensor([1., 1.])
```

That means pytorch call the c++ extension correctly.

## Acknowledgement

Thanks for all the contributors below!

Pytorch official tutorial: https://pytorch.org/tutorials/advanced/cpp_extension.html

kwea123's tutorial: [YouTube](https://www.youtube.com/watch?v=l_Rpk6CRJYI&list=PLDV2CyUo4q-LKuiNltBqCKdO9GH4SS_ec) and [repo](https://github.com/kwea123/pytorch-cppcuda-tutorial)

PyBind11: https://github.com/pybind/pybind11

Example pybind11 module built with a CMake-based build system: https://github.com/pybind/cmake_example

Examples for the usage of "pybind11": https://github.com/tdegeus/pybind11_examples
