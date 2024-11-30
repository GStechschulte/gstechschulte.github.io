use std::{
    ffi::{c_double, c_void},
    os::raw::c_int,
};

use pyo3::prelude::*;
use pyo3::types::PyList;

type Callback = unsafe extern "C" fn(in_: *const c_double, out: *mut c_double, m: c_int, n: c_int);

#[pyfunction]
fn compute_from_rust(fn_ptr: usize) -> PyResult<(PyObject)> {
    let my_python_fn: Callback = unsafe { std::mem::transmute(fn_ptr as *const c_void) };

    let m: c_int = 3;
    let n: c_int = 2;

    let in_ = vec![1., 2., 3., 4., 5., 6.];
    let mut out = vec![0.0; (m * n) as usize];

    unsafe {
        (my_python_fn)(in_.as_ptr(), out.as_mut_ptr(), m, n);
    }

    Python::with_gil(|py| Ok(PyList::new_bound(py, &out).into()))
}

#[pymodule]
fn callback(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_from_rust, m)?)?;
    Ok(())
}
