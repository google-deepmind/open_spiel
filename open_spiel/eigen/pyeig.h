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

#ifndef OPEN_SPIEL_EIGEN_PYEIG_H_
#define OPEN_SPIEL_EIGEN_PYEIG_H_

#include "Eigen/Dense"

// Defines matrix types that use the library Eigen in a way that is compatible
// with numpy arrays. The aim is to use an arrangement of the C++ matrices
// so that no unncessary copying is done to expose them as numpy arrays.
// The known "gotchas" are listed in the README in this directory.
// If you want to use Eigen, include this file.
//
// Relevant docs (recommended reading):
// -
// https://pybind11.readthedocs.io/en/stable/advanced/cast/eigen.html#storage-orders
// - https://eigen.tuxfamily.org/dox/classEigen_1_1Ref.html
namespace open_spiel {

// Use this type for dynamically sized matrices of doubles.
using MatrixXd =
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

// Use this type for dynamically sized vectors of doubles.
using VectorXd = Eigen::VectorXd;

// Use this type for dynamically sized arrays of doubles.
using ArrayXd = Eigen::ArrayXd;

}  // namespace open_spiel

#endif  // OPEN_SPIEL_EIGEN_PYEIG_H_
