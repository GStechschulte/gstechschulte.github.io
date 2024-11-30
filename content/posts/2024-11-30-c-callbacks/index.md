+++
title = 'Creating C Callbacks with Numba and Calling Them From Rust'
date = 2024-11-30
author = 'Gabriel Stechschulte'
categories = ['rust']
draft = false
+++

When interfacing with libraries written in C/C++ from Rust, it may require writing native callbacks to provide functionality or logic to the library. A C Callback is a function pointer that is passed as an argument to another function, allowing that function to "call back" and execute the passed function at runtime.

When writing Rust code with Python bindings, there may be scenarios where the Rust code also needs to be able to call a Python function. Rust's foreign function interface (FFI) and `pyo3` crate in fact lets you do this. However, when calling Python from Rust, you take a performance hit as you invoke the Python interpreter. What if you do not want to invoke the Python interpreter? Enter Numba. You can use Numba to create a C callback, pass this function pointer to Rust to perform the callback, all while not incurring the overhead of Python.

This post will briefly explain how to create C callbacks using Numba, and how to pass and call them from within Rust.

## Creating C Callbacks with Numba

To create a C callback of a Python function using Numba, the `cfunc` function is used. Alternatively, one could use the `@cfunc` decorator. Regardless of the technique, passing a signature of the Python function is required as it determines the visible signature of the C callback. The C function object exposes the address of the compiled C callback as the address attribute, so that you can pass it to a foreign library. The object is also callable from Python.

```python
import ctypes
import numpy as np
from numba import cfunc, carray, types

from callback import initialize

# Define the C signature
c_sig = types.void(types.CPointer(types.double),
                   types.CPointer(types.double),
                   types.intc,
                   types.intc)

@cfunc(c_sig)
def my_callback(in_, out, m, n):
    in_array = carray(in_, (m, n))
    out_array = carray(out, (m, n))
    for i in range(m):
        for j in range(n):
            out_array[i, j] = 2 * in_array[i, j]

# Prepare input data
m, n = 3, 2
input_array = np.array([1., 2., 3., 4., 5., 6.])
output_array = np.zeros(m * n, dtype=np.float64)

# Get pointers to the data
input_ptr = input_array.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
output_ptr = output_array.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

# Call the function using ctypes
my_callback.ctypes(input_ptr, output_ptr, ctypes.c_int(m), ctypes.c_int(n))

print(f"Input array : {input_array}")
print(f"Output array: {output_array}")
```
```
Input array : [1. 2. 3. 4. 5. 6.]
Output array: [2. 4. 6. 8. 10. 12.]
```

Even though the code is being executed within Python, the `my_callback` does not invoke the Python interpreter each time it evaluates `my_callback`—making the code much faster.

## "Call back" from Rust

What if part of our library is written in Rust and needs to be able to call a Python function? Imagine performance is critical so Numba is used to create a C callback from the original Python function. This function pointer will then be passed to Rust where the "callback" is performed, i.e., `my_callback` is called from within Rust without ever invoking the Python interpreter!

The boundary between Rust and the C callback can be crossed using Rust's FFI. FFI lets Rust code call functions written in other programming languages (typically C/C++), and is ultimately all about accessing bytes that originate somewhere outside the Rust code. For that, Rust provides two primary building blocks:
1. **Symbols**. Names assigned to particular addresses in a given segment of your binary that allow you to share memory between the external origin and your Rust code.
2. **Calling convention**. Provides a common understanding of how to call functions stored in such shared memory.

Rust's `extern` keyword is used to link with external functions and variables defined outside of the Rust environment libraries. This is achieved by declaring external blocks where these functions and variables are specified. In our example, we need to define an external block with the function signature (calling convention) of `my_callback`. The call to this function is then wrapped in an `unsafe` block due to the potential risks associated with calling code that originates outside of Rust.

Below, the function signature of `my_callback` is defined in Rust as the type alias `Callback`. The alias represents a function pointer type that can be used to call C functions from Rust. The `ffi` and `os::raw` modules provide type definitions required for C-compatible data types.

```Rust
use std::{
    ffi::{c_double, c_void},
    os::raw::c_int,
};

// Declare calling convention of `my_callback` using C types from the `ffi` module
type Callback = unsafe extern "C" fn(in_: *const c_double, out: *mut c_double, m: c_int, n: c_int);
```
Notice how the function signature of `Callback` matches that of `my_callback`. Now, we need a way of passing the pointer of `my_callback` to Rust. To enable this interface, we will use `pyo3` to create a Python extension module. Details of how to use `pyo3` will not be given here. Rather, we will focus on the declaration of `my_python_fn`. Here, the `fn_ptr` parameter is cast to the `Callback` type using `std::mem::transmute`. This casting is an unsafe operation that converts `fn_ptr` from a `usize` type to the function pointer`Callback` type enabling the callback.

Now, we can perform the callback by passing the appropriate data to `my_python_fn`. The callback also happens in an unsafe block as it involves dereferencing raw pointers. Lastly, the output vector is returned as a Python object.

```Rust
#[pyfunction]
fn compute_from_rust(fn_ptr: usize) -> PyResult<(PyObject)> {
    // Cast `fn_ptr` from usize to the `Callback` type (aka the function pointer)
    let my_python_fn: Callback = unsafe { std::mem::transmute(fn_ptr as *const c_void) };

    // Create data to be passed to `my_python_fn`
    let m: c_int = 3;
    let n: c_int = 2;

    let in_ = vec![1., 2., 3., 4., 5., 6.];
    let mut out = vec![0.0; (m * n) as usize];

    // Perform callback within an unsafe block
    unsafe {
        (my_python_fn)(in_.as_ptr(), out.as_mut_ptr(), m, n);
    }

    // Return the data to Python
    Python::with_gil(|py| Ok(PyList::new_bound(py, &out).into()))
}

#[pymodule]
fn callback(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_from_rust, m)?)?;
    Ok(())
}
```

We can now compile the Rust code, and import and call the extension module in Python.

```Python
import ctypes
import numpy as np
from numba import cfunc, carray, types

from callback import compute_from_rust

c_sig = types.void(types.CPointer(types.double),
                   types.CPointer(types.double),
                   types.intc,
                   types.intc)

@cfunc(c_sig)
def my_callback(in_, out, m, n):
    in_array = carray(in_, (m, n))
    out_array = carray(out, (m, n))
    for i in range(m):
        for j in range(n):
            out_array[i, j] = 2 * in_array[i, j]

# Call `my_callback` within Rust using FFI
result = compute_from_rust(my_callback.address)
print(f"Rust output: {result}")
```
```
Rust output: [2.0, 4.0, 6.0, 8.0, 10.0, 12.0]
```

Voilà, the result is the same as above. The code in this blog post can be found [here](https://github.com/GStechschulte/gstechschulte.github.io/tree/main/content/posts/2024-11-30-c-callbacks/callback/).
