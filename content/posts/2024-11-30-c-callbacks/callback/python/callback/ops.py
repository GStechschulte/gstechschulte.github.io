import numpy as np
from numba import cfunc, types, carray

from callback import initialize

c_sig = types.void(types.CPointer(types.double),
                   types.CPointer(types.double),
                   types.intc,
                   types.intc
)

@cfunc(c_sig)
def my_callback(in_, out, m, n):
    in_array = carray(in_, (m, n))
    out_array = carray(out, (m, n))
    for i in range(m):
        for j in range(n):
            out_array[i, j] = 2 * in_array[i, j]

res = initialize(my_callback.address)
print(res)
