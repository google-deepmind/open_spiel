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

#ifndef OPEN_SPIEL_UTILS_COMBINATORICS_H_
#define OPEN_SPIEL_UTILS_COMBINATORICS_H_

#include <algorithm>
#include <cassert>
#include <iostream>
#include <iterator>
#include <utility>
#include <vector>

#include "open_spiel/spiel_utils.h"

// A suite of basic combinatorial operations.

namespace open_spiel {

// Return all permutations of a vector.
// This returns n! vectors, where n is the size of the vector.
template <typename T>
std::vector<std::vector<T>> Permutations(std::vector<T> v) {
  std::vector<std::vector<T>> vs;
  int perm_size = 1;
  for (int i = 2; i <= v.size(); ++i) perm_size *= i;
  vs.reserve(perm_size);
  std::sort(v.begin(), v.end());
  do {
    vs.push_back(v);
  } while (std::next_permutation(v.begin(), v.end()));
  return vs;
}

// Return all subsets of size k from a vector of size n (all combinations).
// This implements "n choose k" (or also known as binomial coefficient).
// Returns (n k) = n! / ( k! * (n-k)! ) vectors.
template <typename T>
std::vector<std::vector<T>> SubsetsOfSize(const std::vector<T>& v, int k) {
  SPIEL_CHECK_LE(k, v.size());
  SPIEL_CHECK_GE(k, 0);
  std::vector<bool> bitset(v.size() - k, 0);
  bitset.resize(v.size(), 1);
  std::vector<std::vector<T>> vs;

  do {
    std::vector<T> x;
    x.reserve(k);
    for (std::size_t i = 0; i != v.size(); ++i) {
      if (bitset[i]) {
        x.push_back(v[i]);
      }
    }
    vs.push_back(x);
  } while (std::next_permutation(bitset.begin(), bitset.end()));

  return vs;
}

bool NextPowerSetMask(std::vector<bool>* bs);

// Return the power set of a vector of size n.
// Returns 2^n vectors.
template <typename T>
std::vector<std::vector<T>> PowerSet(const std::vector<T>& v) {
  std::vector<bool> bitset(v.size());
  std::vector<std::vector<T>> vs;
  do {
    std::vector<T> x;
    for (std::size_t i = 0; i != v.size(); ++i) {
      if (bitset[i]) {
        x.push_back(v[i]);
      }
    }
    vs.push_back(x);
  } while (NextPowerSetMask(&bitset));
  return vs;
}

std::vector<std::vector<int>> GenerateMasks(
    std::vector<int>& values, int k, std::vector<int>& permutation_stack);

// Return all k-variations without repetition of a vector with the size n.
// Also known as k-permutations of n.
// The input is assumed that it does not contain repetitions.
// This returns n! / (n-k)! vectors.
// TODO(author13): more efficient implementation with something like
//              NextVariationsWithoutRepetitionMask
template <typename T>
std::vector<std::vector<T>> VariationsWithoutRepetition(const std::vector<T>& v,
                                                        int k) {
  SPIEL_CHECK_LE(k, v.size());
  SPIEL_CHECK_GE(k, 0);

  // Generate masks -- avoid copying of T, as that might
  // be more expensive than juggling integers.
  std::vector<int> current_permutation;
  std::vector<int> rng(v.size());
  std::iota(rng.begin(), rng.end(), 0);
  auto masks = GenerateMasks(rng, k, current_permutation);

  // Apply the masks.
  std::vector<std::vector<T>> vs;
  for (auto& mask : masks) {
    std::vector<T> x;
    for (auto& i : mask) {
      x.push_back(v[i]);
    }
    vs.push_back(x);
  }
  return vs;
}

}  // namespace open_spiel

#endif  // OPEN_SPIEL_UTILS_COMBINATORICS_H_
