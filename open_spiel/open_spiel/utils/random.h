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

#ifndef OPEN_SPIEL_UTILS_RANDOM_H_
#define OPEN_SPIEL_UTILS_RANDOM_H_

#include <random>
#include <utility>
#include <vector>

#include "open_spiel/abseil-cpp/absl/random/uniform_real_distribution.h"

// A suite of utilities that wrap random number generators.
//
// It makes it easy to mock stochastic algorithms, as you can supply
// a fixed "random" sequence that will produce desired behaviour
// you can test against.

namespace open_spiel {

class Random {
 public:
  // Return a random value in the interval <0,1)
  virtual double RandomUniform() = 0;

  Random() = default;
  Random(const Random &) = default;
  virtual ~Random() = default;
};

// Random Mersenne Twister.
class RandomMT : public Random {
  std::mt19937 generator_;

 public:
  explicit RandomMT(int seed) : generator_(std::mt19937(seed)) {}
  explicit RandomMT(const std::mt19937 &generator) : generator_(generator) {}
  double RandomUniform() final;
};

// Helper class to provide fixed sampling, according to specified values.
// It keeps cycling through them when end of the list is reached.
// It is not "random", but we keep the prefix name for consistency.
class RandomFixedSequence : public Random {
  const std::vector<double> values_;
  int position_ = 0;

 public:
  // Return values from this specified list.
  RandomFixedSequence(std::initializer_list<double> l) : values_(l) {}

  double RandomUniform() final;
};

}  // namespace open_spiel

#endif  // OPEN_SPIEL_UTILS_RANDOM_H_
