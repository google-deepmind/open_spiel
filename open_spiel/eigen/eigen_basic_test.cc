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

#include <iostream>

#include "open_spiel/eigen/pyeig.h"
#include "open_spiel/spiel.h"

// This is a simple test to check that Eigen works as intended.
// These tests do not involve python bindings, however the matrix types
// are compatible with numpy's arrays.
namespace open_spiel {
namespace {

void MatrixScalarMultiplicationTest() {
  MatrixXd m(2, 2);
  m(0, 0) = 1;
  m(1, 0) = 2;
  m(0, 1) = 3;
  m(1, 1) = 4;

  MatrixXd m2 = m * 2;
  std::cout << "Orig matrix\n" <<  m << std::endl;
  std::cout << "Multiplied matrix\n" << m2 << std::endl;
  SPIEL_CHECK_EQ(m2(0, 0), 2.0);
  SPIEL_CHECK_EQ(m2(1, 0), 4.0);
  SPIEL_CHECK_EQ(m2(0, 1), 6.0);
  SPIEL_CHECK_EQ(m2(1, 1), 8.0);
}

}  // namespace
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::MatrixScalarMultiplicationTest();
}
