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

#ifndef OPEN_SPIEL_UTILS_FUNCTIONAL_H_
#define OPEN_SPIEL_UTILS_FUNCTIONAL_H_

#include <algorithm>
#include <iterator>
#include <string>
#include <utility>
#include <vector>


// A suite of utilities common in functional programming languages.

namespace open_spiel {

template <typename InputSequence1,
          typename InputSequence2,
          typename ZippedOutputIterator>
void Zip(const InputSequence1& first1, const InputSequence1& last1,
         const InputSequence2& first2, ZippedOutputIterator& output) {
  std::transform(
      first1, last1, first2, std::back_inserter(output),
      [](const auto& a, const auto& b) { return std::make_pair(a, b); });
}

}  // namespace open_spiel

#endif  // OPEN_SPIEL_UTILS_FUNCTIONAL_H_
