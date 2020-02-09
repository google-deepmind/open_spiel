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

#include "open_spiel/games/go/go_board.h"

#include <iomanip>

#include "open_spiel/abseil-cpp/absl/random/uniform_int_distribution.h"
#include "open_spiel/games/chess/chess_common.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace go {

namespace {

// Calls f for all 4 direct neighbours of p.
// f should have type void f(VirtualPoint n), but is passed as a template so we
// can elide the function call overhead.
template <typename F>
void Neighbours(VirtualPoint p, const F& f) {
  f(p + kVirtualBoardSize);
  f(p + 1);
  f(p - 1);
  f(p - kVirtualBoardSize);
}

std::vector<VirtualPoint> MakeBoardPoints(int board_size) {
  std::vector<VirtualPoint> points;
  points.reserve(board_size * board_size);
  for (int row = 0; row < board_size; ++row) {
    for (int col = 0; col < board_size; ++col) {
      points.push_back(VirtualPointFrom2DPoint({row, col}));
    }
  }
  return points;
}

template <int board_size>
const std::vector<VirtualPoint>& GetBoardPoints() {
  static std::vector<VirtualPoint> points = MakeBoardPoints(board_size);
  return points;
}

char GoColorToChar(GoColor c) {
  switch (c) {
    case GoColor::kBlack:
      return 'X';
    case GoColor::kWhite:
      return 'O';
    case GoColor::kEmpty:
      return '+';
    case GoColor::kGuard:
      return '#';
    default:
      SpielFatalError("Unknown color.");
      return '!';
  }
}

std::string MoveAsAscii(VirtualPoint p, GoColor c) {
  static std::string code = "0123456789abcdefghijklmnopqrstuvwxyz";
  static int mask = 31;
  // 1 bit for color, 9 bits for the point.
  uint16_t value = static_cast<int>(c) | (p << 1);
  // Encode in 2 characters of 5 bit each.
  std::string encoded;
  encoded.push_back(code[(value >> 5) & mask]);
  encoded.push_back(code[value & mask]);
  return encoded;
}

}  // namespace

std::pair<int, int> VirtualPointTo2DPoint(VirtualPoint p) {
  if (p == kInvalidPoint || p == kVirtualPass) return std::make_pair(-1, -1);

  const int row = static_cast<int>(p) / kVirtualBoardSize;
  const int col = static_cast<int>(p) % kVirtualBoardSize;
  return std::make_pair(row - 1, col - 1);
}

VirtualPoint VirtualPointFrom2DPoint(std::pair<int, int> row_col) {
  return static_cast<VirtualPoint>((row_col.first + 1) * kVirtualBoardSize +
                                   row_col.second + 1);
}

const std::vector<VirtualPoint>& BoardPoints(int board_size) {
#define CASE_GET_POINTS(n) \
  case n:                  \
    return GetBoardPoints<n>()

  switch (board_size) {
    CASE_GET_POINTS(2);
    CASE_GET_POINTS(3);
    CASE_GET_POINTS(4);
    CASE_GET_POINTS(5);
    CASE_GET_POINTS(6);
    CASE_GET_POINTS(7);
    CASE_GET_POINTS(8);
    CASE_GET_POINTS(9);
    CASE_GET_POINTS(10);
    CASE_GET_POINTS(11);
    CASE_GET_POINTS(12);
    CASE_GET_POINTS(13);
    CASE_GET_POINTS(14);
    CASE_GET_POINTS(15);
    CASE_GET_POINTS(16);
    CASE_GET_POINTS(17);
    CASE_GET_POINTS(18);
    CASE_GET_POINTS(19);
    default:
      SpielFatalError("unsupported board size");
  }

#undef CASE_GET_POINTS
}

GoColor OppColor(GoColor c) {
  switch (c) {
    case GoColor::kBlack:
      return GoColor::kWhite;
    case GoColor::kWhite:
      return GoColor::kBlack;
    case GoColor::kEmpty:
    case GoColor::kGuard:
      return c;
    default:
      SpielFatalError("Unknown color.");
      return c;
  }
}

std::ostream& operator<<(std::ostream& os, GoColor c) {
  return os << GoColorToString(c);
}

std::string GoColorToString(GoColor c) {
  switch (c) {
    case GoColor::kBlack:
      return "B";
    case GoColor::kWhite:
      return "W";
    case GoColor::kEmpty:
      return "EMPTY";
    case GoColor::kGuard:
      return "GUARD";
    default:
      SpielFatalError("Unknown color.");
      return "This will never return.";
  }
}

std::ostream& operator<<(std::ostream& os, VirtualPoint p) {
  return os << VirtualPointToString(p);
}

std::string VirtualPointToString(VirtualPoint p) {
  switch (p) {
    case kInvalidPoint:
      return "INVALID_POINT";
    case kVirtualPass:
      return "PASS";
    default: {
      auto row_col = VirtualPointTo2DPoint(p);
      char col = 'a' + row_col.second;
      if (col >= 'i') ++col;  // Go / SGF labeling skips 'i'.
      return absl::StrCat(std::string(1, col), row_col.first + 1);
    }
  }
}

VirtualPoint MakePoint(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(), ::tolower);

  if (s == "pass") return kVirtualPass;
  if (s.size() < 2 || s.size() > 3) return kInvalidPoint;

  int col = s[0] < 'i' ? s[0] - 'a' : s[0] - 'a' - 1;
  int row = s[1] - '0';
  if (s.size() == 3) {
    row *= 10;
    row += s[2] - '0';
  }
  return VirtualPointFrom2DPoint({row - 1, col});
}

GoBoard::GoBoard(int board_size) : board_size_(board_size) { Clear(); }

void GoBoard::Clear() {
  zobrist_hash_ = 0;

  for (int i = 0; i < board_.size(); ++i) {
    Vertex& v = board_[i];
    v.color = GoColor::kGuard;
    v.chain_head = static_cast<VirtualPoint>(i);
    v.chain_next = static_cast<VirtualPoint>(i);
    chains_[i].reset_border();
  }

  for (VirtualPoint p : BoardPoints(board_size_)) {
    board_[p].color = GoColor::kEmpty;
    chains_[p].reset();
  }

  for (VirtualPoint p : BoardPoints(board_size_)) {
    Neighbours(p, [this, p](VirtualPoint n) {
      if (IsEmpty(n)) chain(p).add_liberty(n);
    });
  }

  for (int i = 0; i < last_captures_.size(); ++i) {
    last_captures_[i] = kInvalidPoint;
  }

  last_ko_point_ = kInvalidPoint;
}

bool GoBoard::PlayMove(VirtualPoint p, GoColor c) {
  if (p == kVirtualPass) {
    last_ko_point_ = kInvalidPoint;
    return true;
  }

  SPIEL_CHECK_EQ(GoColor::kEmpty, board_[p].color);

  // Preparation for ko checking.
  bool played_in_enemy_eye = true;
  Neighbours(p, [this, c, &played_in_enemy_eye](VirtualPoint n) {
    GoColor s = PointColor(n);
    if (s == c || s == GoColor::kEmpty) {
      played_in_enemy_eye = false;
    }
  });

  JoinChainsAround(p, c);
  SetStone(p, c);
  RemoveLibertyFromNeighbouringChains(p);
  int stones_captured = CaptureDeadChains(p, c);

  if (played_in_enemy_eye && stones_captured == 1) {
    last_ko_point_ = last_captures_[0];
  } else {
    last_ko_point_ = kInvalidPoint;
  }

  SPIEL_CHECK_GT(chain(p).num_pseudo_liberties, 0);

  return true;
}

void GoBoard::SetStone(VirtualPoint p, GoColor c) {
  static const chess_common::ZobristTable<uint64_t, kVirtualBoardPoints, 2>
      zobrist_values(
          /*seed=*/2765481);

  zobrist_hash_ ^= zobrist_values[p][static_cast<int>(
      c == GoColor::kEmpty ? PointColor(p) : c)];

  board_[p].color = c;
}

// Combines the groups around the newly placed stone at vertex. If no groups
// are available for joining, the new stone is placed as a new group.
void GoBoard::JoinChainsAround(VirtualPoint p, GoColor c) {
  VirtualPoint largest_chain_head = kInvalidPoint;
  int largest_chain_size = 0;
  Neighbours(
      p, [this, c, &largest_chain_head, &largest_chain_size](VirtualPoint n) {
        if (PointColor(n) == c) {
          Chain& c = chain(n);
          if (c.num_stones > largest_chain_size) {
            largest_chain_size = c.num_stones;
            largest_chain_head = ChainHead(n);
          }
        }
      });
  if (largest_chain_size == 0) {
    InitNewChain(p);
    return;
  }

  Neighbours(p, [this, c, &largest_chain_head](VirtualPoint n) {
    if (PointColor(n) == c) {
      VirtualPoint chain_head = ChainHead(n);
      if (chain_head != largest_chain_head) {
        chain(largest_chain_head).merge(chain(n));

        // Set all stones in the smaller string to be part of the larger
        // chain.
        VirtualPoint cur = n;
        do {
          board_[cur].chain_head = largest_chain_head;
          cur = board_[cur].chain_next;
        } while (cur != n);

        // Connect the 2 linked lists representing the stones in the two
        // chains.
        std::swap(board_[largest_chain_head].chain_next, board_[n].chain_next);
      }
    }
  });

  board_[p].chain_next = board_[largest_chain_head].chain_next;
  board_[largest_chain_head].chain_next = p;
  board_[p].chain_head = largest_chain_head;
  chain(largest_chain_head).num_stones += 1;

  Neighbours(p, [this, largest_chain_head](VirtualPoint n) {
    if (IsEmpty(n)) {
      chain(largest_chain_head).add_liberty(n);
    }
  });
}

void GoBoard::RemoveLibertyFromNeighbouringChains(VirtualPoint p) {
  Neighbours(p, [this, p](VirtualPoint n) { chain(n).remove_liberty(p); });
}

int GoBoard::CaptureDeadChains(VirtualPoint p, GoColor c) {
  int stones_captured = 0;
  int capture_index = 0;
  Neighbours(p, [this, c, &capture_index, &stones_captured](VirtualPoint n) {
    if (PointColor(n) == OppColor(c) && chain(n).num_pseudo_liberties == 0) {
      last_captures_[capture_index++] = ChainHead(n);
      stones_captured += chain(n).num_stones;
      RemoveChain(n);
    }
  });

  for (; capture_index < last_captures_.size(); ++capture_index) {
    last_captures_[capture_index] = kInvalidPoint;
  }

  return stones_captured;
}

void GoBoard::RemoveChain(VirtualPoint p) {
  VirtualPoint this_chain_head = ChainHead(p);
  VirtualPoint cur = p;
  do {
    VirtualPoint next = board_[cur].chain_next;

    SetStone(cur, GoColor::kEmpty);
    InitNewChain(cur);

    Neighbours(cur, [this, this_chain_head, cur](VirtualPoint n) {
      if (ChainHead(n) != this_chain_head || IsEmpty(n)) {
        chain(n).add_liberty(cur);
      }
    });

    cur = next;
  } while (cur != p);
}

void GoBoard::InitNewChain(VirtualPoint p) {
  board_[p].chain_head = p;
  board_[p].chain_next = p;

  Chain& c = chain(p);
  c.reset();
  c.num_stones += 1;

  Neighbours(p, [this, &c](VirtualPoint n) {
    if (IsEmpty(n)) {
      c.add_liberty(n);
    }
  });
}

bool GoBoard::IsInBoardArea(VirtualPoint p) const {
  auto rc = VirtualPointTo2DPoint(p);
  return rc.first >= 0 && rc.first < board_size() && rc.second >= 0 &&
         rc.second < board_size();
}

bool GoBoard::IsLegalMove(VirtualPoint p, GoColor c) const {
  if (p == kVirtualPass) return true;
  if (!IsInBoardArea(p)) return false;
  if (!IsEmpty(p) || p == LastKoPoint()) return false;
  if (chain(p).num_pseudo_liberties > 0) return true;

  // For all checks below, the newly placed stone is completely surrounded by
  // enemy and friendly stones.

  // Allow to play if the placed stones connects to a group that still has at
  // least one other liberty after connecting.
  bool has_liberty = false;
  Neighbours(p, [this, c, &has_liberty](VirtualPoint n) {
    has_liberty |= (PointColor(n) == c && !chain(n).in_atari());
  });
  if (has_liberty) return true;

  // Allow to play if the placed stone will kill at least one group.
  bool kills_group = false;
  Neighbours(p, [this, c, &kills_group](VirtualPoint n) {
    kills_group |= (PointColor(n) == OppColor(c) && chain(n).in_atari());
  });
  if (kills_group) return true;

  return false;
}

void GoBoard::Chain::reset_border() {
  num_stones = 0;
  // Need to have values big enough that they can never go below 0 even if
  // all liberties are removed.
  num_pseudo_liberties = 4;
  liberty_vertex_sum = 32768;
  liberty_vertex_sum_squared = 2147483648;
}

void GoBoard::Chain::reset() {
  num_stones = 0;
  num_pseudo_liberties = 0;
  liberty_vertex_sum = 0;
  liberty_vertex_sum_squared = 0;
}

void GoBoard::Chain::merge(const Chain& other) {
  num_stones += other.num_stones;
  num_pseudo_liberties += other.num_pseudo_liberties;
  liberty_vertex_sum += other.liberty_vertex_sum;
  liberty_vertex_sum_squared += other.liberty_vertex_sum_squared;
}

void GoBoard::Chain::add_liberty(VirtualPoint p) {
  num_pseudo_liberties += 1;
  liberty_vertex_sum += p;
  liberty_vertex_sum_squared +=
      static_cast<uint32_t>(p) * static_cast<uint32_t>(p);
}

void GoBoard::Chain::remove_liberty(VirtualPoint p) {
  num_pseudo_liberties -= 1;
  liberty_vertex_sum -= p;
  liberty_vertex_sum_squared -=
      static_cast<uint32_t>(p) * static_cast<uint32_t>(p);
}

std::ostream& operator<<(std::ostream& os, const GoBoard& board) {
  os << "\n";
  for (int row = board.board_size() - 1; row >= 0; --row) {
    os << std::setw(2) << std::setfill(' ') << (row + 1) << " ";
    for (int col = 0; col < board.board_size(); ++col) {
      os << GoColorToChar(
          board.PointColor(VirtualPointFrom2DPoint({row, col})));
    }
    os << std::endl;
  }

  std::string columns = "ABCDEFGHJKLMNOPQRST";
  os << "   " << columns.substr(0, board.board_size()) << std::endl;

  // Encode the stones and print a URL that can be used to view the board.
  std::string encoded;
  for (VirtualPoint p : BoardPoints(board.board_size())) {
    if (!board.IsEmpty(p)) {
      encoded += MoveAsAscii(p, board.PointColor(p));
    }
  }

  // TODO(author9): Make this a public URL.
  // os << "http://jumper/goboard/" << encoded << "&size=" << board.board_size()
  //    << std::endl;

  return os;
}

void GoBoard::GroupIter::step() {
  --lib_i_;
  while (lib_i_ < 0 && !marked_[chain_cur_]) {
    Neighbours(chain_cur_, [this](VirtualPoint n) {
      VirtualPoint head = board_->ChainHead(n);
      if (board_->PointColor(head) == group_color_ && !marked_[head]) {
        cur_libs_[++lib_i_] = head;
        marked_[head] = true;
      }
    });
    marked_[chain_cur_] = true;
    chain_cur_ = board_->board_[chain_cur_].chain_next;
  }
}

// Returns the number of points surrounded entirely by one color.
// Aborts early and returns 0 if the area borders both black and white stones.
int NumSurroundedPoints(const GoBoard& board, const VirtualPoint p,
                        std::array<bool, kVirtualBoardPoints>* marked,
                        bool* reached_black, bool* reached_white) {
  if ((*marked)[p] || (*reached_black && *reached_white)) return 0;
  (*marked)[p] = true;

  int num_points = 1;
  Neighbours(p, [&board, &num_points, marked, reached_black,
                 reached_white](VirtualPoint n) {
    switch (board.PointColor(n)) {
      case GoColor::kBlack:
        *reached_black = true;
        break;
      case GoColor::kWhite:
        *reached_white = true;
        break;
      case GoColor::kEmpty:
        num_points +=
            NumSurroundedPoints(board, n, marked, reached_black, reached_white);
        break;
      case GoColor::kGuard:
        // Ignore the border.
        break;
    }
  });

  return num_points;
}

float TrompTaylorScore(const GoBoard& board, float komi, int handicap) {
  // The delta of how many points on the board black and white have occupied,
  // from black's point of view.
  int occupied_delta = 0;

  // We need to keep track of which empty points we've already counted as part
  // of a larger territory.
  std::array<bool, kVirtualBoardPoints> marked;
  marked.fill(false);

  for (VirtualPoint p : BoardPoints(board.board_size())) {
    switch (board.PointColor(p)) {
      case GoColor::kBlack:
        ++occupied_delta;
        break;
      case GoColor::kWhite:
        --occupied_delta;
        break;
      case GoColor::kEmpty: {
        if (marked[p]) continue;
        // If some empty points are surrounded entirely by one player, they
        // count as that player's territory.
        bool reached_black = false, reached_white = false;
        int n = NumSurroundedPoints(board, p, &marked, &reached_black,
                                    &reached_white);
        if (reached_black && !reached_white) {
          occupied_delta += n;
        } else if (!reached_black && reached_white) {
          occupied_delta -= n;
        }
        break;
      }
      case GoColor::kGuard:
        SpielFatalError("unexpected color");
    }
  }

  float score = occupied_delta - komi;
  if (handicap >= 2) {
    score -= handicap;
  }
  return score;
}

}  // namespace go
}  // namespace open_spiel
