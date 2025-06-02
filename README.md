# Arrayx - A Deep Learning Framework

## Overview
Arrayx is a lightweight deep learning framework designed to provide Metal-accelerated array(or tensor) operations similar to PyTorch. Future releases will hopefully include CUDA support for NVIDIA GPUs.

## Requirements
A virtual environment(e.g., Conda) is recommended before installing Python packages. The following software is needed to run Arrayx:
- Python 3.12+
- C++ 23
- NumPy (>=2.0 but should work for any version)
- Nanobind >= 2.7.0 (Python package for C++ bindings)
- Metal 3.2 and metal-cpp for macOS 15.2 and iOS 18.2
- Pytest >= 8.3.4 (optional, mainly for testing arrayx in Python)

## Installation
1. Clone the repository
```bash
git clone https://github.com/dashluu/arrayx.git
cd arrayx
```

2. Build the project
Do the following on macOS from the project root:
* Build the project and generate `.so` and `.pyi` file using the following commands:
```bash
cd arrayx
cmake -S . -B build
cmake --build build
```
The two files are automatically generated in the `python` directory.


## Usage
The following assumes you will be writing code in the `python` directory. If you want to write your code elsewhere, move `.so` and `.pyi` file to the directory you are writing your code in.

1. Import `arrayx` in your Python code:
```python
from arrayx import Array, f32, Backend
```

2. Initialize backend and arrays, remember to call `Backend.cleanup()` at the end of the program:
```python
Backend.init()
shape = [2, 3, 4]
x1 = Array.from_numpy(np.random.randn(*shape).astype(np.float32))
x2 = Array.from_numpy(np.random.randn(*shape).astype(np.float32))
x3 = Array.full(shape, 2.0, dtype=f32)
```

3. Define computations and run:
```python
# Define computation graph
# Make sure the value is a single element before evaluation and backpropagation
out = ((x1 + x2) * x3).exp().sum()
# Execute both forward and backward pass once backward is called
out.backward()
Backend.cleanup()
```

Check out an example of `main.py` inside `python` directory. Cleaner code can also be achieved by using context manager defined in `ax.py`:
```python
from arrayx import Array, f32, Backend
import ax
import numpy as np
import torch


with ax.context():
    arr1 = Array.zeros([2, 5, 3])
    arr2 = arr1[:, 1:4]
    arr2 += Array.ones([2, 3, 3])
	# print triggers the computational graph to be compiled and executed
    print(arr2)
```

## Features
- Metal-accelerated array operations
- Automatic differentiation
- Full computational graph forward and backward propagation
- Well supported operations:
  - Initialization operations: `full`, `arange`, `ones`, `zeros`, `from_numpy`, `numpy`, `torch`
  - Array transformation operations: `reshape`, `permute`, `slice`, `transpose`
  - Matrix multiplication `matmul`
  - Element-wise operations: `add`, `sub`, `mul`, `div`, `exp`, `log`, `neg`(negation), `recip`(reciprocal), `sqrt`, `sq`(square)
  - Reduction operations: `sum`, `mean`, `max`, `min`, `argmax`, `argmin`
- NumPy, PyTorch integration
- The only data types currently supported are `f32`(float32), `i32`(int32), and `b8`(bool).

## Examples
Check out the `python/tests` directory for example implementations of:
- Array operations
- Gradient calculations
- Metal backend usage
- PyTorch comparison tests

## Acknowledgement
These resources inspired me to do the project:
- MLX by Apple (https://github.com/ml-explore/mlx)
- PyTorch by Meta (https://github.com/pytorch/pytorch)
- Tinygrad by George Hotz (https://github.com/tinygrad/tinygrad)
- Micrograd by Karpathy (https://github.com/karpathy/micrograd)
