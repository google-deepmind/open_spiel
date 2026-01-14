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

#ifndef OPEN_SPIEL_UTILS_TENSOR_VIEW_H_
#define OPEN_SPIEL_UTILS_TENSOR_VIEW_H_

#include <algorithm>
#include <numeric>
#include <vector>

#include "open_spiel/abseil-cpp/absl/types/span.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {

// Treat a `absl::Span<float>` as a tensor of fixed shape. The rank (number of
// dimensions) must be known at compile time, though the actual sizes of the
// dimensions can be supplied at construction time. It then lets you index into
// the vector easily without having to compute the 1d-vector's indices manually.
template <int Rank>
class TensorView {
 public:
  constexpr TensorView(absl::Span<float> values,
                       const std::array<int, Rank>& shape, bool reset)
      : values_(values), shape_(shape) {
    SPIEL_CHECK_EQ(size(), values_.size());
    if (reset) std::fill(values.begin(), values.end(), 0);
  }

  constexpr int size() const {
    return std::accumulate(shape_.begin(), shape_.end(), 1,
                           std::multiplies<int>());
  }

  void clear() { std::fill(values_.begin(), values_.end(), 0.0); }

  constexpr int index(const std::array<int, Rank>& args) const {
    int ind = 0;
    for (int i = 0; i < Rank; ++i) {
      ind = ind * shape_[i] + args[i];
    }
    return ind;
  }

  constexpr float& operator[](const std::array<int, Rank>& args) {
    return values_[index(args)];
  }
  constexpr const float& operator[](const std::array<int, Rank>& args) const {
    return values_[index(args)];
  }

  constexpr int rank() const { return Rank; }
  constexpr const std::array<int, Rank> shape() const { return shape_; }
  constexpr int shape(int i) const { return shape_[i]; }

 private:
  absl::Span<float> values_;
  const std::array<int, Rank> shape_;
};

}  // namespace open_spiel

#endif  // OPEN_SPIEL_UTILS_TENSOR_VIEW_H_
