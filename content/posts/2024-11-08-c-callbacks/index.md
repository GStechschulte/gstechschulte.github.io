+++
title = 'Creating C Callbacks with Numba and Calling Them From Rust'
date = 2024-11-08
author = 'Gabriel Stechschulte'
categories = ['rust']
draft = true
+++

When interfacing with libraries written in C/C++ from Rust, it may require writing native callbacks to provide functionality or logic to the library. A C Callback is a function pointer that is passed as an argument to another function, allowing that function to "call back" and execute the passed function at runtime.

When writing Rust code with Python bindings, there may be scenarios where the Rust code also needs to be able to call a Python function. `pyo3` in fact lets you do this. However, when calling Python from Rust, you take a performance hit as you invoke the Python interpretor. What if you do not want to invoke the Python interpretor? Enter Numba. You can use Numba to create a C callback, pass this function pointer to Rust to call the callback, all while not incurring the overhead of Python.

## Creating C Callbacks with Numba

To create a C callback of a Python function using Numba, the `cfunc` function is used. Alternatively, the `@cfunc` decorator can be used. Regardless, of what technique is used, passing a signature of the Python function is required as it determines the visible signature of the C callback. The C function object exposes the address of the compiled C callback as the address attribute, so that you can pass it to any foreign C/C++/Rust library. The object is also callable from Python.

```python
import numpy as np
import scipy.integrate as si

from numba import cfunc

def integrate(t):
    return np.exp(-t) / t ** 2

def do_integrate(func):
    return si.quad(func, 1, np.inf)

cb_integrate = cfunc("float64(float64)")(integrate)
do_integrate(cb_integrate)
```

Even though we are executing the code in Python, the `cb_integrate` does not invoke the Python interpreter each time it evaluates the integral—making the code much faster.

### Dealing with pointers and array memory



## "Call back" from Rust

We can perform the "call back" from Rust using a foreign function interface (FFI).
