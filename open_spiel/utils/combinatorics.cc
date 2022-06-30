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

#include "open_spiel/utils/combinatorics.h"

#include <algorithm>
#include <cassert>
#include <iostream>
#include <iterator>
#include <utility>
#include <vector>

namespace open_spiel {

bool NextPowerSetMask(std::vector<bool>* bs) {
  for (std::size_t i = 0; i != bs->size(); ++i) {
    (*bs)[i] = !(*bs)[i];
    if ((*bs)[i]) {
      return true;
    }
  }
  return false;  // overflow
}

std::vector<std::vector<int>> GenerateMasks(
    std::vector<int>& values, int k, std::vector<int>& permutation_stack) {
  if (k == permutation_stack.size()) {
    return {permutation_stack};
  }

  std::vector<std::vector<int>> vs;
  auto end_valid = values.size() - permutation_stack.size();
  permutation_stack.push_back(0);
  for (int i = 0; i < end_valid; ++i) {
    permutation_stack.back() = values[i];
    std::swap(values[i], values[end_valid - 1]);
    auto child_vs = GenerateMasks(values, k, permutation_stack);
    vs.insert(vs.begin(), child_vs.begin(), child_vs.end());
    std::swap(values[i], values[end_valid - 1]);
  }
  permutation_stack.pop_back();
  return vs;
}

int Factorial(int n) { return n <= 1 ? 1 : n * Factorial(n - 1); }

}  // namespace open_spiel
