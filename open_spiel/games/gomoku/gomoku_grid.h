// Copyright 2026 DeepMind Technologies Limited
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

#pragma once

#include <algorithm>   // std::fill
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace open_spiel {
namespace gomoku {

template <typename T>
class Grid {
 public:
  using Coord = std::vector<int>;
  using Direction = Coord;

  Grid(std::size_t size, std::size_t dims, bool wrap)
      : size_(size),
        dims_(dims),
        wrap_(wrap),
        strides_(dims),
        data_(ComputeTotalSize(size, dims)) {
    SPIEL_CHECK_GE(size, 1);
    SPIEL_CHECK_GE(dims, 1);
    // stride[d] = size^(dims - d - 1)
    strides_[dims_ - 1] = 1;
    for (std::size_t i = dims_ - 1; i > 0; --i) {
      strides_[i - 1] = strides_[i] * size_;
    }
  }

  void Fill(const T& value) { std::fill(data_.begin(), data_.end(), value); }

  std::size_t Flatten(const Coord& coord) const {
    SPIEL_CHECK_EQ(coord.size(), dims_);

    std::size_t index = 0;
    for (std::size_t d = 0; d < dims_; ++d) {
      int c = coord[d];

      if (wrap_) {
        c %= static_cast<int>(size_);
        if (c < 0) c += static_cast<int>(size_);
      } else {
        SPIEL_CHECK_GE(c, 0);
        SPIEL_CHECK_LE(c, static_cast<int>(size_));
      }

      index += static_cast<std::size_t>(c) * strides_[d];
    }
    return index;
  }

  Coord Unflatten(std::size_t index) const {
    SPIEL_CHECK_LE(index,  data_.size());

    Coord coord(dims_);
    for (std::size_t d = 0; d < dims_; ++d) {
      coord[d] = static_cast<int>(index / strides_[d]);
      index %= strides_[d];
    }
    return coord;
  }

  bool Step(Coord& coord, const Direction& dir) const {
    SPIEL_CHECK_EQ(coord.size(), dims_);
    SPIEL_CHECK_EQ(dir.size(), dims_);

    for (std::size_t d = 0; d < dims_; ++d) {
      int next = coord[d] + dir[d];

      if (wrap_) {
        next %= static_cast<int>(size_);
        if (next < 0) next += static_cast<int>(size_);
      } else {
        if (next < 0 || next >= static_cast<int>(size_)) {
          return false;
        }
      }

      coord[d] = next;
    }
    return true;
  }

  T& At(const Coord& coord) { return data_[Flatten(coord)]; }
  const T& At(const Coord& coord) const { return data_[Flatten(coord)]; }

  T& AtIndex(std::size_t i) { return data_[i]; }
  const T& AtIndex(std::size_t i) const { return data_[i]; }

  std::size_t NumCells() const { return data_.size(); }

  const std::vector<Direction>& Directions() const {
    if (!directions_.empty()) return directions_;

    Direction dir(dims_, 0);
    GenerateDirectionsRecursive(0, dir);
    return directions_;
  }

  static bool IsCanonical(const Direction& dir) {
    for (int v : dir) {
      if (v > 0) return true;
      if (v < 0) return false;
    }
    return false;  // zero vector should not be present
  }
  // These are the one-step basic rotations. To get all rotations apply these
  // 1, 2, or 3 times. In 3 or more dimensions we must also compose them.
  std::vector<std::pair<int, int>> GenRotations() const {
    std::vector<std::pair<int, int>> rotations;
      for (int i = 0; i < dims_; ++i) {
        for (int j = i + 1; j < dims_; ++j) {
          rotations.emplace_back(i, j);
        }
      }
    return rotations;
  }

  Grid<T> ApplyRotation(int i, int j) const {
    SPIEL_CHECK_LT(i, j);
    SPIEL_CHECK_LT(j, dims_);

    Grid<T> result(size_, dims_, wrap_);

    const int L = size_;

    for (int idx = 0; idx < NumCells(); ++idx) {
       Coord old = Unflatten(idx);
       Coord neu = old;

       const int x = old[i];
       const int y = old[j];

       neu[i] = (L - 1) - y;
       neu[j] = x;

       result.At(neu) =  At(old);
     }

    return result;
  }

  // Get reflections aligned with grid axes. Diagonal reflections can be formed
  // by composing these straigh reflections with rotations.
  std::vector<int> GenReflections() const {
    std::vector<int> refs;
    for (int k = 0; k < dims_; ++k) {
      refs.push_back(k);
    }
    return refs;
  }

  Grid<T> ApplyReflection(int k) const {
    SPIEL_CHECK_LT(k, dims_);

    Grid<T> result(size_, dims_, wrap_);
    const int L = size_;

    for (int idx = 0; idx < NumCells(); ++idx) {
       Coord old = Unflatten(idx);
       Coord neu = old;

       neu[k] = (L - 1) - old[k];

       result.At(neu) = At(old);
    }

    return result;
  }

  int Size() const { return size_; }

 protected:
  static std::size_t ComputeTotalSize(std::size_t size, std::size_t dims) {
    std::size_t total = 1;
    for (std::size_t i = 0; i < dims; ++i) {
      total *= size;
    }
    return total;
  }

 private:
  void GenerateDirectionsRecursive(std::size_t d, Direction& dir) const {
    if (d == dims_) {
      bool all_zero = true;
      for (int v : dir) {
        if (v != 0) {
          all_zero = false;
          break;
        }
      }
      if (!all_zero) directions_.push_back(dir);
      return;
    }

    for (int v = -1; v <= 1; ++v) {
      dir[d] = v;
      GenerateDirectionsRecursive(d + 1, dir);
    }
  }

  std::size_t size_;
  std::size_t dims_;
  bool wrap_;
  std::vector<std::size_t> strides_;
  std::vector<T> data_;

  mutable std::vector<Direction> directions_;
};

}  // namespace gomoku
}  // namespace open_spiel
