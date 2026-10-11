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

// Examples of how to use the C++ API:
// - https://github.com/pytorch/examples/tree/master/cpp
// - https://github.com/prabhuomkar/pytorch-cpp

#include "open_spiel/spiel_utils.h"
#include "torch/torch.h"

namespace {

void TestMatrixMultiplication() {
  at::Tensor mat = torch::rand({3, 3});
  at::Tensor identity = torch::ones({3, 3});
  at::Tensor multiplied = mat * identity;
  int num_identical_elements = (mat == multiplied).sum().item().to<int>();
  SPIEL_CHECK_EQ(num_identical_elements, 9);
}

}  // namespace

int main() { TestMatrixMultiplication(); }
