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

#ifndef OPEN_SPIEL_UTILS_FUNCTIONAL_H_
#define OPEN_SPIEL_UTILS_FUNCTIONAL_H_

#include <string>
#include <utility>
#include <vector>

#include "open_spiel/spiel_utils.h"

// A suite of utilities common in functional programming languages.

namespace open_spiel {

// A helper to create a zipped vector from two vectors.
// The resulting vector has the size of xs, possibly omitting any longer ys.
template<typename X, typename Y>
std::vector<std::pair<X, Y>> Zip(
    const std::vector<X>& xs, const std::vector<Y>& ys) {
  SPIEL_CHECK_LE(xs.size(), ys.size());
  std::vector<std::pair<X, Y>> zipped;
  zipped.reserve(xs.size());
  for (int i = 0; i < xs.size(); ++i) {
    zipped.emplace_back(std::make_pair(xs[i], ys[i]));
  }
  return zipped;
}

}  // namespace open_spiel

#endif  // OPEN_SPIEL_UTILS_FUNCTIONAL_H_
