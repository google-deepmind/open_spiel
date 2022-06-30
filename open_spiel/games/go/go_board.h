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

#ifndef OPEN_SPIEL_GAMES_GO_GO_BOARD_H_
#define OPEN_SPIEL_GAMES_GO_GO_BOARD_H_

#include <array>
#include <cstdint>
#include <ostream>
#include <vector>

#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace go {

enum class GoColor : uint8_t { kBlack = 0, kWhite = 1, kEmpty = 2, kGuard = 3 };

std::string GoColorToString(GoColor c);

std::ostream &operator<<(std::ostream &os, GoColor c);

GoColor OppColor(GoColor c);

// For simplicity and speed, we store the board in terms of a "virtual board",
// with a border of guard stones around all sides of the board.
// This allows us to skip bounds checking.
// In Virtual mode, an action (row, col) is row * 21 + col, and pass is 21*21+1.
// All functions in this file (except stated otherwise) use these virtual
// coordinates.
//
// However, in the OpenSpiel API (in go.{h, cc}), the actions are still exposed
// as actions within 0, board_size*boardsize) (with pass = board_size **2.
//
// We support boards up to size 19.
inline constexpr int kMaxBoardSize = 19;
inline constexpr int kVirtualBoardSize = kMaxBoardSize + 2;
inline constexpr int kVirtualBoardPoints =
    kVirtualBoardSize * kVirtualBoardSize;

using VirtualPoint = uint16_t;

inline constexpr VirtualPoint kInvalidPoint = 0;
inline constexpr VirtualPoint kVirtualPass = kVirtualBoardPoints + 1;

// Returns the VirtualPoint corresponding to the provided coordinates, e.g. "d4"
// or "f10".
VirtualPoint MakePoint(std::string s);

// Converts a VirtualPoint to a string representation.
std::string VirtualPointToString(VirtualPoint p);

std::ostream &operator<<(std::ostream &os, VirtualPoint p);

// Conversion functions between VirtualPoint and row/column representation.
std::pair<int, int> VirtualPointTo2DPoint(VirtualPoint p);
// Returns the point identifier in the Virtual 21*21 board from the (row, col)
// 0-index coordinate in the concrete board.
VirtualPoint VirtualPointFrom2DPoint(std::pair<int, int> row_col);

// Converts an OpenSpiel action in range [0, board_size **2] to the
// Virtual board range [0, kVirtualPass], and vice-versa.
Action VirtualActionToAction(int virtual_action, int board_size);
int ActionToVirtualAction(Action action, int board_size);

inline std::string GoActionToString(Action action, int board_size) {
  return VirtualPointToString(ActionToVirtualAction(action, board_size));
}

// Returns a reference to a vector that contains all points that are on a board
// of the specified size.
const std::vector<VirtualPoint> &BoardPoints(int board_size);

// To iterate over 4 neighbouring points, do
//
// VirtualPoint point;
// for (auto p = Neighbours4(point); p; ++p) {
//   // Do something on p..
// }
//
class Neighbours4 {
 public:
  explicit Neighbours4(const VirtualPoint p);

  Neighbours4 &operator++();
  const VirtualPoint operator*() const;
  explicit operator bool() const;

 private:
  VirtualPoint dir_;
  const VirtualPoint p_;
};

// Simple Go board that is optimized for speed.
// It only implements the minimum of functionality necessary to support the
// search and is optimized for speed and size. Importantly, it fits on the
// stack. For detailed numbers, run the benchmarks in go_board_test.
class GoBoard {
 public:
  explicit GoBoard(int board_size);

  void Clear();

  inline int board_size() const { return board_size_; }
  // Returns the concrete pass action.
  inline int pass_action() const { return pass_action_; }
  inline Action VirtualActionToAction(int virtual_action) const {
    return go::VirtualActionToAction(virtual_action, board_size_);
  }
  inline int ActionToVirtualAction(Action action) const {
    return go::ActionToVirtualAction(action, board_size_);
  }

  inline GoColor PointColor(VirtualPoint p) const { return board_[p].color; }

  inline bool IsEmpty(VirtualPoint p) const {
    return PointColor(p) == GoColor::kEmpty;
  }

  bool IsInBoardArea(VirtualPoint p) const;

  bool IsLegalMove(VirtualPoint p, GoColor c) const;

  bool PlayMove(VirtualPoint p, GoColor c);

  // kInvalidPoint if there is no ko, otherwise the point of the ko.
  inline VirtualPoint LastKoPoint() const { return last_ko_point_; }

  // Count of pseudo-liberties, i.e. each liberty is counted between 1 and 4
  // times, once for each stone of the group that borders it.
  // This is much faster than realLiberty(), so prefer it if possible.
  inline int PseudoLiberty(VirtualPoint p) const {
    return chain(p).num_pseudo_liberties == 0
               ? 0
               : (chain(p).in_atari() ? 1 : chain(p).num_pseudo_liberties);
  }

  inline bool InAtari(VirtualPoint p) const { return chain(p).in_atari(); }

  // If a chain has a single liberty (it is in Atari), return that liberty.
  VirtualPoint SingleLiberty(VirtualPoint p) const;

  // Actual liberty count, i.e. each liberty is counted exactly once.
  // This is computed on the fly by actually walking the group and checking the
  // neighbouring stones.
  inline int RealLiberty(VirtualPoint p) const {
    int num_lib = 0;
    for (auto it = LibIter(p); it; ++it) {
      ++num_lib;
    }
    return num_lib;
  }

  inline uint64_t HashValue() const { return zobrist_hash_; }

  // Head of a chain; each chain has exactly one head that can be used to
  // uniquely identify it. Chain heads may change over successive PlayMove()s.
  inline VirtualPoint ChainHead(VirtualPoint p) const {
    return board_[p].chain_head;
  }

  // Number of stones in a chain.
  inline int ChainSize(VirtualPoint p) const { return chain(p).num_stones; }

  std::string ToString();

  class GroupIter {
   public:
    GroupIter(const GoBoard *board, VirtualPoint p, GoColor group_color)
        : board_(board), lib_i_(0), group_color_(group_color) {
      marked_.fill(false);
      chain_head_ = board->ChainHead(p);
      chain_cur_ = chain_head_;
      step();
    }

    inline explicit operator bool() const { return lib_i_ >= 0; }

    inline VirtualPoint operator*() const { return cur_libs_[lib_i_]; }

    GroupIter &operator++() {
      step();
      return *this;
    }

   private:
    void step();

    const GoBoard *board_;

    std::array<bool, kVirtualBoardPoints> marked_;
    std::array<VirtualPoint, 4> cur_libs_;
    int lib_i_;
    VirtualPoint chain_head_;
    VirtualPoint chain_cur_;
    GoColor group_color_;
  };

  GroupIter LibIter(VirtualPoint p) const {
    return GroupIter(this, p, GoColor::kEmpty);
  }
  GroupIter OppIter(VirtualPoint p) const {
    return GroupIter(this, p, OppColor(PointColor(p)));
  }

 private:
  void JoinChainsAround(VirtualPoint p, GoColor c);
  void SetStone(VirtualPoint p, GoColor c);
  void RemoveLibertyFromNeighbouringChains(VirtualPoint p);
  int CaptureDeadChains(VirtualPoint p, GoColor c);
  void RemoveChain(VirtualPoint p);
  void InitNewChain(VirtualPoint p);

  struct Vertex {
    VirtualPoint chain_head;
    VirtualPoint chain_next;
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
    void add_liberty(VirtualPoint p);
    void remove_liberty(VirtualPoint p);
    VirtualPoint single_liberty() const;
  };

  Chain &chain(VirtualPoint p) { return chains_[ChainHead(p)]; }
  const Chain &chain(VirtualPoint p) const { return chains_[ChainHead(p)]; }

  std::array<Vertex, kVirtualBoardPoints> board_;
  std::array<Chain, kVirtualBoardPoints> chains_;

  uint64_t zobrist_hash_;

  // Chains captured in the last move, kInvalidPoint otherwise.
  std::array<VirtualPoint, 4> last_captures_;

  int board_size_;
  int pass_action_;

  VirtualPoint last_ko_point_;
};

std::ostream &operator<<(std::ostream &os, const GoBoard &board);

// Score according to https://senseis.xmp.net/?TrompTaylorRules.
float TrompTaylorScore(const GoBoard &board, float komi, int handicap = 0);

// Generates a go board from the given string, setting X to black stones and O
// to white stones. The first character of the first line is mapped to A1, the
// second character to B1, etc, as below:
//     ABCDEFGH
//   1 ++++XO++
//   2 XXXXXO++
//   3 OOOOOO++
//   4 ++++++++
// The board will always be 19x19.
// This exists mostly for test purposes.
// WARNING: This coordinate system is different from the representation in
// GoBoard in which A1 is at the bottom left.
GoBoard CreateBoard(const std::string &initial_stones);

}  // namespace go
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_GO_GO_BOARD_H_
