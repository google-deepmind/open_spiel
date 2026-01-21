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
    if (size_ == 0 || dims_ == 0) {
      throw std::invalid_argument("Grid size and dims must be >= 1");
    }

    // stride[d] = size^(dims - d - 1)
    strides_[dims_ - 1] = 1;
    for (std::size_t i = dims_ - 1; i > 0; --i) {
      strides_[i - 1] = strides_[i] * size_;
    }
  }

  void Fill(const T& value) { std::fill(data_.begin(), data_.end(), value); }

  std::size_t Flatten(const Coord& coord) const {
    if (coord.size() != dims_) {
      throw std::invalid_argument("Coordinate has wrong dimension");
    }

    std::size_t index = 0;
    for (std::size_t d = 0; d < dims_; ++d) {
      int c = coord[d];

      if (wrap_) {
        c %= static_cast<int>(size_);
        if (c < 0) c += static_cast<int>(size_);
      } else {
        if (c < 0 || c >= static_cast<int>(size_)) {
          throw std::out_of_range("Coordinate out of bounds");
        }
      }

      index += static_cast<std::size_t>(c) * strides_[d];
    }
    return index;
  }

  Coord Unflatten(std::size_t index) const {
    if (index >= data_.size()) {
      throw std::out_of_range("Index out of bounds");
    }

    Coord coord(dims_);
    for (std::size_t d = 0; d < dims_; ++d) {
      coord[d] = static_cast<int>(index / strides_[d]);
      index %= strides_[d];
    }
    return coord;
  }

  bool Step(Coord& coord, const Direction& dir) const {
    if (coord.size() != dims_ || dir.size() != dims_) {
      throw std::invalid_argument("Dimension mismatch in Step()");
    }

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
