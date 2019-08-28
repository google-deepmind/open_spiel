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

#ifndef THIRD_PARTY_OPEN_SPIEL_GAMES_GO_GO_BOARD_H_
#define THIRD_PARTY_OPEN_SPIEL_GAMES_GO_GO_BOARD_H_

#include <array>
#include <cstdint>
#include <ostream>
#include <vector>

namespace open_spiel {
namespace go {

enum class GoColor : uint8_t { kBlack = 0, kWhite = 1, kEmpty = 2, kGuard = 3 };

std::string GoColorToString(GoColor c);

std::ostream &operator<<(std::ostream &os, GoColor c);

GoColor OppColor(GoColor c);

// For simplicity and speed, we store the board in terms of a "virtual board",
// with a border of guard stones around all sides of the board.
// This allows us to skip bounds checking.
// We support boards up to size 19.
constexpr int kMaxBoardSize = 19;
constexpr int kVirtualBoardSize = kMaxBoardSize + 2;
constexpr int kVirtualBoardPoints = kVirtualBoardSize * kVirtualBoardSize;

using GoPoint = uint16_t;

constexpr GoPoint kInvalidPoint = 0;
constexpr GoPoint kPass = kVirtualBoardPoints + 1;

// Returns the GoPoint corresponding to the provided coordinates, e.g. "d4" or
// "f10".
GoPoint MakePoint(std::string s);

// Converts a GoPoint to a string representation.
std::string GoPointToString(GoPoint p);

std::ostream &operator<<(std::ostream &os, GoPoint p);

// Conversion functions between GoPoint and row/column representation.
std::pair<int, int> GoPointTo2DPoint(GoPoint p);
GoPoint GoPointFrom2DPoint(std::pair<int, int> row_col);

// Returns a reference to a vector that contains all points that are on a board
// of the specified size.
const std::vector<GoPoint> &BoardPoints(int board_size);

// Simple Go board that is optimized for speed.
// It only implements the minimum of functionality necessary to support the
// search and is optimized for speed and size. Importantly, it fits on the
// stack. For detailed numbers, run the benchmarks in go_board_test.
class GoBoard {
 public:
  explicit GoBoard(int board_size);

  void Clear();

  inline int board_size() const { return board_size_; }

  inline GoColor PointColor(GoPoint p) const { return board_[p].color; }

  inline bool IsEmpty(GoPoint p) const {
    return PointColor(p) == GoColor::kEmpty;
  }

  bool IsInBoardArea(GoPoint p) const;

  bool IsLegalMove(GoPoint p, GoColor c) const;

  bool PlayMove(GoPoint p, GoColor c);

  // kInvalidPoint if there is no ko, otherwise the point of the ko.
  inline GoPoint LastKoPoint() const { return last_ko_point_; }

  // Count of pseudo-liberties, i.e. each liberty is counted between 1 and 4
  // times, once for each stone of the group that borders it.
  // This is much faster than realLiberty(), so prefer it if possible.
  inline int PseudoLiberty(GoPoint p) const {
    return chain(p).num_pseudo_liberties == 0
               ? 0
               : (chain(p).in_atari() ? 1 : chain(p).num_pseudo_liberties);
  }

  inline bool InAtari(GoPoint p) const { return chain(p).in_atari(); }

  inline uint64_t HashValue() const { return zobrist_hash_; }

  // Actual liberty count, i.e. each liberty is counted exactly once.
  // This is computed on the fly by actually walking the group and checking the
  // neighbouring stones.
  inline int RealLiberty(GoPoint p) const {
    int num_lib = 0;
    for (auto it = LibIter(p); it; ++it) {
      ++num_lib;
    }
    return num_lib;
  }

  // Head of a chain; each chain has exactly one head that can be used to
  // uniquely identify it. Chain heads may change over successive playMove()s.
  inline GoPoint ChainHead(GoPoint p) const { return board_[p].chain_head; }

  class GroupIter {
   public:
    GroupIter(const GoBoard *board, GoPoint p, GoColor group_color)
        : board_(board), lib_i_(0), group_color_(group_color) {
      marked_.fill(false);
      chain_head_ = board->ChainHead(p);
      chain_cur_ = chain_head_;
      step();
    }

    inline explicit operator bool() const { return lib_i_ >= 0; }

    inline GoPoint operator*() const { return cur_libs_[lib_i_]; }

    GroupIter &operator++() {
      step();
      return *this;
    }

   private:
    void step();

    const GoBoard *board_;

    std::array<bool, kVirtualBoardPoints> marked_;
    std::array<GoPoint, 4> cur_libs_;
    int lib_i_;
    GoPoint chain_head_;
    GoPoint chain_cur_;
    GoColor group_color_;
  };

  GroupIter LibIter(GoPoint p) const {
    return GroupIter(this, p, GoColor::kEmpty);
  }
  GroupIter OppIter(GoPoint p) const {
    return GroupIter(this, p, OppColor(PointColor(p)));
  }

 private:
  void JoinChainsAround(GoPoint p, GoColor c);
  void SetStone(GoPoint p, GoColor c);
  void RemoveLibertyFromNeighbouringChains(GoPoint p);
  int CaptureDeadChains(GoPoint p, GoColor c);
  void RemoveChain(GoPoint p);
  void InitNewChain(GoPoint p);

  struct Vertex {
    GoPoint chain_head;
    GoPoint chain_next;
    GoColor color;
  };

  struct Chain {
    uint32_t liberty_vertex_sum_squared;
    uint16_t liberty_vertex_sum;
    uint16_t num_stones;
    uint16_t num_pseudo_liberties;

    void reset();
    void reset_border();
    void merge(const Chain &other);

    inline bool in_atari() const {
      return static_cast<uint32_t>(num_pseudo_liberties) *
                 liberty_vertex_sum_squared ==
             static_cast<uint32_t>(liberty_vertex_sum) *
                 static_cast<uint32_t>(liberty_vertex_sum);
    }
    void add_liberty(GoPoint p);
    void remove_liberty(GoPoint p);
  };

  Chain &chain(GoPoint p) { return chains_[ChainHead(p)]; }
  const Chain &chain(GoPoint p) const { return chains_[ChainHead(p)]; }

  std::array<Vertex, kVirtualBoardPoints> board_;
  std::array<Chain, kVirtualBoardPoints> chains_;

  uint64_t zobrist_hash_;

  // Chains captured in the last move, kInvalidPoint otherwise.
  std::array<GoPoint, 4> last_captures_;

  int board_size_;

  GoPoint last_ko_point_;
};

std::ostream &operator<<(std::ostream &os, const GoBoard &board);

// Score according to https://senseis.xmp.net/?TrompTaylorRules.
float TrompTaylorScore(const GoBoard &board, float komi, int handicap = 0);

}  // namespace go
}  // namespace open_spiel

#endif  // THIRD_PARTY_OPEN_SPIEL_GAMES_GO_GO_BOARD_H_
