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

# Prepare input data
m, n = 3, 2
input_array = np.array([1., 2., 3., 4., 5., 6.])
output_array = np.zeros(m * n, dtype=np.float64)

# Get pointers to the data
input_ptr = input_array.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
output_ptr = output_array.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

# Call the function using ctypes
my_callback.ctypes(input_ptr, output_ptr, ctypes.c_int(m), ctypes.c_int(n))

print(f"Input Array : {input_array}")
print(f"Output Array: {output_array}")

# Call `my_callback` within Rust using FFI
result = compute_from_rust(my_callback.address)
print(f"Rust output: {result}")
