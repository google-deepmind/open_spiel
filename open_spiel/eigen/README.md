# Integration with Eigen library

This is an integration with the
[Eigen library](http://eigen.tuxfamily.org/index.php?title=Main_Page), based on
the documentation of
[pybind](https://pybind11.readthedocs.io/en/stable/advanced/cast/eigen.html#)

This is an optional dependency and it can be enabled by `BUILD_WITH_EIGEN`
global variable (see `install.sh`).

Use the header `eigen/pyeig.h` to get basic `Matrix` and `Vector` types. The
types in this header file are tested for compatibility with numpy. Other Eigen
types might not be compatible (due to memory layout), so be careful if you use
them in the code and you'd like to expose them to Python.

There is an integration test with pybind: it creates an internal namespace
`open_spiel::eigen_test`, which is then invoked as part of the Python test suite
by loading module `pyspiel_eigen`.

## Known gotchas

Things to keep in mind.

-   Numpy stores vectors as 1D shape. Eigen however stores vectors as 2D shape,
    i.e. a matrix with one dimension equal to one. The default implementation in
    Eigen sets the column dimension to be equal to 1. However, to be compatible
    with numpy's memory layout, we need to use row layout, so by default **the
    row dimension** is equal to 1. See `test_square_vector_elements`
