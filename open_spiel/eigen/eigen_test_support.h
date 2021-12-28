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

#ifndef OPEN_SPIEL_EIGEN_EIGEN_TEST_SUPPORT_H_
#define OPEN_SPIEL_EIGEN_EIGEN_TEST_SUPPORT_H_

#include "open_spiel/eigen/pyeig.h"

namespace open_spiel {
namespace eigen_test {

// A simple testing function that squares matrix elements.
inline MatrixXd SquareElements(const MatrixXd &xs) {
  return xs.cwiseProduct(xs);
}

// A simple function that allocates a matrix and returns a copy.
inline MatrixXd CreateSmallTestingMatrix() {
  MatrixXd m(2, 2);
  m(0, 0) = 1;
  m(0, 1) = 2;
  m(1, 0) = 3;
  m(1, 1) = 4;
  return m;
}

// From https://pybind11.readthedocs.io/en/stable/advanced/cast/eigen.html#returning-values-to-python
// An example of returning an owning copy or a
// non-owning (non)writeable reference.
class BigMatrixForTestingClass {
  MatrixXd big_mat = MatrixXd::Zero(10000, 10000);
 public:
  MatrixXd &getMatrix() { return big_mat; }
  const MatrixXd &viewMatrix() { return big_mat; }
};

}  // namespace eigen_test
}  // namespace open_spiel

#endif  // OPEN_SPIEL_EIGEN_EIGEN_TEST_SUPPORT_H_
