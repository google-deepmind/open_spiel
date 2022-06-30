// Copyright 2019 DeepMind Technologies Limited
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

#ifndef OPEN_SPIEL_GAMES_IMPL_COMMON_CHESS_COMMON_H_
#define OPEN_SPIEL_GAMES_IMPL_COMMON_CHESS_COMMON_H_

#include <array>
#include <cstdint>
#include <random>
#include <string>

#include "open_spiel/abseil-cpp/absl/random/uniform_int_distribution.h"

namespace open_spiel {
namespace chess_common {

struct Offset {
  int8_t x_offset;
  int8_t y_offset;

  bool operator==(const Offset& other) const {
    return x_offset == other.x_offset && y_offset == other.y_offset;
  }
};

// x corresponds to file (column / letter)
// y corresponds to rank (row / number).
struct Square {
  Square& operator+=(const Offset& offset) {
    x += offset.x_offset;
    y += offset.y_offset;
    return *this;
  }

  bool operator==(const Square& other) const {
    return x == other.x && y == other.y;
  }

  bool operator!=(const Square& other) const { return !(*this == other); }

  int8_t x;
  int8_t y;
};

constexpr Square kInvalidSquare{-1, -1};

inline std::string SquareToString(const Square& square) {
  if (square == kInvalidSquare) {
    return "None";
  } else {
    std::string s;
    s.push_back('a' + square.x);
    s.push_back('1' + square.y);
    return s;
  }
}

inline Square operator+(const Square& sq, const Offset& offset) {
  int8_t x = sq.x + offset.x_offset;
  int8_t y = sq.y + offset.y_offset;
  return Square{x, y};
}

// This function takes an Offset which represents a relative chess move and
// encodes it into an integer: the DestinationIndex. The encoding enumerates the
// queen moves and then the knight moves. For chess, this results in the
// following mapping:
//  - [ 0, 13]: 14 vertical moves
//  - [14, 27]: 14 horizontal moves
//  - [28, 41]: 14 left downward or right upward diagonal moves
//  - [42, 55]: 14 left upward or right downward diagonal moves
//  - [56, 63]:  8 knight moves
int OffsetToDestinationIndex(const Offset& offset,
                             const std::array<Offset, 8>& knight_offsets,
                             int board_size);
int OffsetToDestinationIndex(const Offset& offset,
                             const std::array<Offset, 2>& knight_offsets,
                             int board_size);

// Inverse function of OffsetToDestinationIndex
Offset DestinationIndexToOffset(int destination_index,
                                const std::array<Offset, 8>& knight_offsets,
                                int board_size);
Offset DestinationIndexToOffset(int destination_index,
                                const std::array<Offset, 2>& knight_offsets,
                                int board_size);

// Encoding is:
// i = (x * board_size + y) * num_actions_destinations + destination_index
// where x,y are the square coordinates.
std::pair<Square, int> DecodeNetworkTarget(int i, int board_size,
                                           int num_actions_destinations);
int EncodeNetworkTarget(const Square& from_square, int destination_index,
                        int board_size, int num_actions_destinations);

// n-dimensional array of uniform random numbers.
// Example:
//   ZobristTable<int, 3, 4, 5> table;
//
//   table[a][b][c] is a random int where a < 3, b < 4, c < 5
//
template <typename T, std::size_t InnerDim, std::size_t... OtherDims>
class ZobristTable {
 public:
  using Generator = std::mt19937_64;
  using NestedTable = ZobristTable<T, OtherDims...>;

  ZobristTable(Generator::result_type seed) {
    Generator generator(seed);
    absl::uniform_int_distribution<Generator::result_type> dist;
    data_.reserve(InnerDim);
    for (std::size_t i = 0; i < InnerDim; ++i) {
      data_.emplace_back(dist(generator));
    }
  }

  const NestedTable& operator[](std::size_t inner_index) const {
    return data_[inner_index];
  }

 private:
  std::vector<NestedTable> data_;
};

// 1-dimensional array of uniform random numbers.
template <typename T, std::size_t InnerDim>
class ZobristTable<T, InnerDim> {
 public:
  using Generator = std::mt19937_64;

  ZobristTable(Generator::result_type seed) : data_(InnerDim) {
    Generator generator(seed);
    absl::uniform_int_distribution<T> dist;
    for (auto& field : data_) {
      field = dist(generator);
    }
  }

  T operator[](std::size_t index) const { return data_[index]; }

 private:
  std::vector<T> data_;
};

inline std::ostream& operator<<(std::ostream& stream, const Square& sq) {
  return stream << SquareToString(sq);
}

}  // namespace chess_common
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_IMPL_COMMON_CHESS_COMMON_H_
