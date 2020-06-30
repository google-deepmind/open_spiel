// Copyright 2019 DeepMind Technologies Ltd. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
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

void TestPermutations() {
  {
    std::vector<int> v = {};
    std::vector<std::vector<int>> actual_vs = Permutations(v);
    std::vector<std::vector<int>> expected_vs = {{}};
    SPIEL_CHECK_EQ(actual_vs, expected_vs);
  }
  {
    std::vector<int> v = {1};
    std::vector<std::vector<int>> actual_vs = Permutations(v);
    std::vector<std::vector<int>> expected_vs = {{1}};
    SPIEL_CHECK_EQ(actual_vs, expected_vs);
  }
  {
    std::vector<int> v = {1, 2, 3};
    std::vector<std::vector<int>> actual_vs = Permutations(v);
    std::vector<std::vector<int>> expected_vs = {
        {1, 2, 3}, {1, 3, 2}, {2, 1, 3}, {2, 3, 1}, {3, 1, 2}, {3, 2, 1}};
    SPIEL_CHECK_EQ(actual_vs, expected_vs);
  }
  {
    std::vector<char> v = {'a', 'b', 'c'};
    std::vector<std::vector<char>> actual_vs = Permutations(v);
    std::vector<std::vector<char>> expected_vs = {
        {'a', 'b', 'c'}, {'a', 'c', 'b'}, {'b', 'a', 'c'},
        {'b', 'c', 'a'}, {'c', 'a', 'b'}, {'c', 'b', 'a'}};
    SPIEL_CHECK_EQ(actual_vs, expected_vs);
  }
}

void TestCombinations() {
  {
    std::vector<int> v = {};
    std::vector<std::vector<int>> actual_vs = SubsetsOfSize(v, 0);
    std::vector<std::vector<int>> expected_vs = {{}};
    SPIEL_CHECK_EQ(actual_vs, expected_vs);
  }
  {
    std::vector<int> v = {1};
    std::vector<std::vector<int>> actual_vs = SubsetsOfSize(v, 0);
    std::vector<std::vector<int>> expected_vs = {{}};
    SPIEL_CHECK_EQ(actual_vs, expected_vs);
  }
  {
    std::vector<int> v = {1};
    std::vector<std::vector<int>> actual_vs = SubsetsOfSize(v, 1);
    std::vector<std::vector<int>> expected_vs = {{1}};
    SPIEL_CHECK_EQ(actual_vs, expected_vs);
  }
  {
    std::vector<int> v = {1, 2, 3, 4};
    std::vector<std::vector<int>> actual_vs = SubsetsOfSize(v, 2);
    std::vector<std::vector<int>> expected_vs = {
        {3, 4}, {2, 4}, {2, 3}, {1, 4}, {1, 3}, {1, 2}};
    SPIEL_CHECK_EQ(actual_vs, expected_vs);
  }
  {
    std::vector<char> v = {'a', 'b', 'c', 'd'};
    std::vector<std::vector<char>> actual_vs = SubsetsOfSize(v, 2);
    std::vector<std::vector<char>> expected_vs = {
        {'c', 'd'}, {'b', 'd'}, {'b', 'c'}, {'a', 'd'}, {'a', 'c'}, {'a', 'b'}};
    SPIEL_CHECK_EQ(actual_vs, expected_vs);
  }
}

void TestPowerSet() {
  {
    std::vector<int> v = {};
    std::vector<std::vector<int>> actual_vs = PowerSet(v);
    std::vector<std::vector<int>> expected_vs = {{}};
    SPIEL_CHECK_EQ(actual_vs, expected_vs);
  }
  {
    std::vector<int> v = {1};
    std::vector<std::vector<int>> actual_vs = PowerSet(v);
    std::vector<std::vector<int>> expected_vs = {{}, {1}};
    SPIEL_CHECK_EQ(actual_vs, expected_vs);
  }
  {
    std::vector<int> v = {1, 2, 3};
    std::vector<std::vector<int>> actual_vs = PowerSet(v);
    std::vector<std::vector<int>> expected_vs = {
        {}, {1}, {2}, {1, 2}, {3}, {1, 3}, {2, 3}, {1, 2, 3}};
    SPIEL_CHECK_EQ(actual_vs, expected_vs);
  }
  {
    std::vector<char> v = {'a', 'b', 'c'};
    std::vector<std::vector<char>> actual_vs = PowerSet(v);
    std::vector<std::vector<char>> expected_vs = {
        {}, {'a'}, {'b'}, {'a', 'b'}, {'c'},
        {'a', 'c'}, {'b', 'c'}, {'a', 'b', 'c'}};
    SPIEL_CHECK_EQ(actual_vs, expected_vs);
  }
}

void TestVariationsWithoutRepetition() {
  {
    std::vector<int> v = {};
    std::vector<std::vector<int>> actual_vs = VariationsWithoutRepetition(v, 0);
    std::vector<std::vector<int>> expected_vs = {{}};
    SPIEL_CHECK_EQ(actual_vs, expected_vs);
  }
  {
    std::vector<int> v = {1};
    std::vector<std::vector<int>> actual_vs = VariationsWithoutRepetition(v, 0);
    std::vector<std::vector<int>> expected_vs = {{}};
    SPIEL_CHECK_EQ(actual_vs, expected_vs);
  }
  {
    std::vector<int> v = {1};
    std::vector<std::vector<int>> actual_vs = VariationsWithoutRepetition(v, 1);
    std::vector<std::vector<int>> expected_vs = {{1}};
    SPIEL_CHECK_EQ(actual_vs, expected_vs);
  }
  {
    std::vector<int> v = {1, 2, 3};
    std::vector<std::vector<int>> actual_vs = VariationsWithoutRepetition(v, 2);
    std::vector<std::vector<int>> expected_vs = {
        {3, 2}, {3, 1}, {2, 3}, {2, 1}, {1, 2}, {1, 3}};
    SPIEL_CHECK_EQ(actual_vs, expected_vs);
  }
  {
    std::vector<char> v = {'a', 'b', 'c'};
    std::vector<std::vector<char>> actual_vs =
        VariationsWithoutRepetition(v, 2);
    std::vector<std::vector<char>> expected_vs = {
        {'c', 'b'}, {'c', 'a'}, {'b', 'c'}, {'b', 'a'}, {'a', 'b'}, {'a', 'c'}};
    SPIEL_CHECK_EQ(actual_vs, expected_vs);
  }
}

}  // namespace
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::TestPermutations();
  open_spiel::TestCombinations();
  open_spiel::TestPowerSet();
  open_spiel::TestVariationsWithoutRepetition();
}
