// Copyright 2021 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "open_spiel/eigen/eigen_test_support.h"
#include "open_spiel/python/pybind11/pybind11.h"
// Make sure that we can convert Eigen types to proper bindings.
#include "pybind11/include/pybind11/eigen.h"

// This file contains OpenSpiel's Python API for Eigen.
// This is a python package intended for testing purposes.

namespace open_spiel {
namespace {

namespace py = ::pybind11;

// Definition of our Python module.
PYBIND11_MODULE(pyspiel_eigen_test, m) {
  m.doc() = "OpenSpiel Eigen testing module";

  // Register bits of the testing API.
  m.def("square", &eigen_test::SquareElements,
        py::arg().noconvert(),  // Avoid silent copying on incorrect types.
        "Squares elements of a matrix.");
  m.def("matrix", &eigen_test::CreateSmallTestingMatrix,
        "Allocate a 2x2 testing matrix on C++ side.");

  py::class_<eigen_test::BigMatrixForTestingClass>(m, "BigMatrix")
      .def(py::init<>())
      .def("copy_matrix", &eigen_test::BigMatrixForTestingClass::getMatrix)
      .def("get_matrix", &eigen_test::BigMatrixForTestingClass::getMatrix,
           py::return_value_policy::reference_internal)
      .def("view_matrix", &eigen_test::BigMatrixForTestingClass::viewMatrix,
           py::return_value_policy::reference_internal);
}

}  // namespace
}  // namespace open_spiel
