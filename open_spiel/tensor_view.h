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

#ifndef THIRD_PARTY_OPEN_SPIEL_TENSOR_VIEW_H_
#define THIRD_PARTY_OPEN_SPIEL_TENSOR_VIEW_H_

#include <algorithm>
#include <numeric>
#include <vector>

#include "open_spiel/spiel_utils.h"

namespace open_spiel {

// Treat a `std::vector<double>` as a tensor of fixed shape. The rank (number of
// dimensions) must be known at compile time, though the actual sizes of the
// dimensions can be supplied at construction time. It then lets you index into
// the vector easily without having to compute the 1d-vector's indices manually.
// Given the common use case is to fill the observations in
// ObservationTensor and InformationStateTensor it offers a way to resize and
// clear the vector to match the specified shape at construction.
template <int Rank>
class TensorView {
 public:
  constexpr TensorView(std::vector<double>* values,
                       const std::array<int, Rank>& shape, bool reset)
      : values_(values), shape_(shape) {
    if (reset) {
      int old_size = values_->size();
      int new_size = size();
      values_->resize(new_size, 0.0);
      std::fill(values_->begin(),
                values_->begin() + std::min(old_size, new_size), 0.0);
    } else {
      SPIEL_CHECK_EQ(size(), values_->size());
    }
  }

  constexpr int size() const {
    return std::accumulate(shape_.begin(), shape_.end(), 1,
                           std::multiplies<int>());
  }

  void clear() { std::fill(values_->begin(), values_->end(), 0.0); }

  constexpr int index(const std::array<int, Rank>& args) const {
    int ind = 0;
    for (int i = 0; i < Rank; ++i) {
      ind = ind * shape_[i] + args[i];
    }
    return ind;
  }

  constexpr double& operator[](const std::array<int, Rank>& args) {
    return (*values_)[index(args)];
  }
  constexpr const double& operator[](const std::array<int, Rank>& args) const {
    return (*values_)[index(args)];
  }

  constexpr int rank() const { return Rank; }
  constexpr const std::array<int, Rank> shape() const { return shape_; }
  constexpr int shape(int i) const { return shape_[i]; }

 private:
  std::vector<double>* values_;
  const std::array<int, Rank> shape_;
};

}  // namespace open_spiel

#endif  // THIRD_PARTY_OPEN_SPIEL_TENSOR_VIEW_H_
