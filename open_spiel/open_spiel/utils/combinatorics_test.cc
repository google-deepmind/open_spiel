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

#include <cstdlib>
#include <string>

namespace open_spiel {
namespace {

void CheckPermutation(
    std::vector<int> v, std::vector<std::vector<int>> expected) {
  std::vector<std::vector<int>> actual = Permutations(v);
  SPIEL_CHECK_EQ(actual, expected);
}

void TestPermutations() {
  CheckPermutation({}, {{}});
  CheckPermutation({1}, {{1}});
  CheckPermutation({1, 2, 3}, {{1, 2, 3}, {1, 3, 2}, {2, 1, 3},
                               {2, 3, 1}, {3, 1, 2}, {3, 2, 1}});
}

void CheckSubsetsOfSize(
    std::vector<int> v, int k, std::vector<std::vector<int>> expected) {
  std::vector<std::vector<int>> actual = SubsetsOfSize(v, k);
  SPIEL_CHECK_EQ(actual, expected);
}


void TestSubsetsOfSize() {
  CheckSubsetsOfSize({}, 0, {{}});
  CheckSubsetsOfSize({1}, 0, {{}});
  CheckSubsetsOfSize({1}, 1, {{1}});
  CheckSubsetsOfSize({1, 2, 3, 4}, 2, {{3, 4}, {2, 4}, {2, 3},
                                       {1, 4}, {1, 3}, {1, 2}});
}

void CheckPowerSet(
    std::vector<int> v, std::vector<std::vector<int>> expected) {
  std::vector<std::vector<int>> actual = PowerSet(v);
  SPIEL_CHECK_EQ(actual, expected);
}


void TestPowerSet() {
  CheckPowerSet({}, {{}});
  CheckPowerSet({1}, {{}, {1}});
  CheckPowerSet({1, 2, 3},
                {{}, {1}, {2}, {1, 2}, {3}, {1, 3}, {2, 3}, {1, 2, 3}});
}

void CheckVariationsWithoutRepetition(
    std::vector<int> v, int k, std::vector<std::vector<int>> expected) {
  std::vector<std::vector<int>> actual = VariationsWithoutRepetition(v, k);
  SPIEL_CHECK_EQ(actual, expected);
}


void TestVariationsWithoutRepetition() {
  CheckVariationsWithoutRepetition({}, 0, {{}});
  CheckVariationsWithoutRepetition({1}, 0, {{}});
  CheckVariationsWithoutRepetition({1}, 1, {{1}});
  CheckVariationsWithoutRepetition({1, 2, 3}, 2, {{3, 2}, {3, 1}, {2, 3},
                                                  {2, 1}, {1, 2}, {1, 3}});
}

void UnrankPermutationTest() {
  std::vector<std::vector<int>> all_perms = {
      {0, 1, 2, 3}, {0, 1, 3, 2}, {0, 2, 1, 3}, {0, 2, 3, 1}, {0, 3, 1, 2},
      {0, 3, 2, 1}, {1, 0, 2, 3}, {1, 0, 3, 2}, {1, 2, 0, 3}, {1, 2, 3, 0},
      {1, 3, 0, 2}, {1, 3, 2, 0}, {2, 0, 1, 3}, {2, 0, 3, 1}, {2, 1, 0, 3},
      {2, 1, 3, 0}, {2, 3, 0, 1}, {2, 3, 1, 0}, {3, 0, 1, 2}, {3, 0, 2, 1},
      {3, 1, 0, 2}, {3, 1, 2, 0}, {3, 2, 0, 1}, {3, 2, 1, 0}};

  std::vector<int> elements = {0, 1, 2, 3};
  for (int k = 0; k < 24; ++k) {
    std::vector<int> perm = UnrankPermutation(elements, k);
    SPIEL_CHECK_TRUE(perm == all_perms[k]);
  }
}

}  // namespace
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::TestPermutations();
  open_spiel::TestSubsetsOfSize();
  open_spiel::TestPowerSet();
  open_spiel::TestVariationsWithoutRepetition();
  open_spiel::UnrankPermutationTest();
}
