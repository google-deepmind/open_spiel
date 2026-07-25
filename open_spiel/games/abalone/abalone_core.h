// Copyright 2023 DeepMind Technologies Limited
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

#ifndef OPEN_SPIEL_GAMES_ABALONE_ABALONE_CORE_H_
#define OPEN_SPIEL_GAMES_ABALONE_ABALONE_CORE_H_

#include <string>
#include <tuple>
#include <utility>

// Simple game of Abalone
// https://en.wikipedia.org/wiki/Abalone_(board_game)

namespace abalone_core {

enum Direction : int {  // counterclockwise order
  Direction_First = 0,

  RIGHT = Direction_First,
  UP_RIGHT = 1,
  UP_LEFT = 2,
  LEFT = 3,
  DOWN_LEFT = 4,
  DOWN_RIGHT = 5,

  Direction_Last = 6,
  Direction_Invalid = -1,
};

// Constants.
constexpr int kNumPlayers = 2;
constexpr int kNumRows = 9;
constexpr int kNumCols = 9;
constexpr int kNumCells = kNumRows * kNumCols;
// single move or slide move x2 or x3 from near or far left
constexpr int kNumActionsPerDirection = 5;
constexpr int kNumActionsPerCell = Direction_Last * kNumActionsPerDirection;
constexpr int kHistoryMax = 200;  // a game can't last longer than this
// Stop the game once a player has lost this many marbles (default: 6,
// blitz: 4).
constexpr int kMarblesToWin = 6;
constexpr double kMarbleReward = 0.0;  // check: kMarblesToWin*kMarbleReward<1
constexpr int kCellStates = 2 + kNumPlayers;  // empty, invalid, and players
const char kDefaultBoard[] = "classic";  // default board to play
// Invert the positions of player 1 and player 2.
constexpr bool kInvertBoard = false;
// The player with the most marbles wins at the end of the game (instead
// of a draw).
constexpr bool kMarbleAdvantage = false;

// State of a cell.
enum CellState : int8_t {
  Invalid = -2,
  Empty = -1,
  Player0 = 0,
  Player1 = 1,
};

std::string StateToString(CellState state);

// Hexagonal board represented as a square with some Invalid cells:
//
// I     2 2 2 2 2
// H    2 2 2 2 2 2
// G   0 0 2 2 2 0 0
// F  0 0 0 0 0 0 0 0
// E 0 0 0 0 0 0 0 0 0
// D  0 0 0 0 0 0 0 0 \9
// C   0 0 1 1 1 0 0 \8
// B    1 1 1 1 1 1 \7
// A     1 1 1 1 1 \6
//        \1\2\3\4\5
//
// Square/Memory representation:
//
// I X X X X 2 2 2 2 2
// H X X X 2 2 2 2 2 2
// G X X 0 0 2 2 2 0 0
// F X 0 0 0 0 0 0 0 0
// E 0 0 0 0 0 0 0 0 0
// D 0 0 0 0 0 0 0 0 X
// C 0 0 1 1 1 0 0 X X
// B 1 1 1 1 1 1 X X X
// A 1 1 1 1 1 X X X X
//   1 2 3 4 5 6 7 8 9

extern const CellState VALID_BOARD[kNumRows][kNumCols];
extern const CellState ABALONE_INIT_CLASSIC[kNumRows][kNumCols];

// cf https://abaloneonline.wordpress.com/variations/the-classics/
extern const CellState ABALONE_INIT_BELGIAN_DAISY[kNumRows][kNumCols];

constexpr std::pair<Direction, Direction> Sisters[] = {
    // eq to dir+1 and dir+2
    { Direction::UP_RIGHT, Direction::UP_LEFT },     // Direction::RIGHT
    { Direction::UP_LEFT, Direction::LEFT },         // Direction::UP_RIGHT
    { Direction::LEFT, Direction::DOWN_LEFT },       // Direction::UP_LEFT
    { Direction::DOWN_LEFT, Direction::DOWN_RIGHT },  // Direction::LEFT
    { Direction::DOWN_RIGHT, Direction::RIGHT },     // Direction::DOWN_LEFT
    { Direction::RIGHT, Direction::UP_RIGHT },       // Direction::DOWN_RIGHT
};
static_assert(sizeof(Sisters) / sizeof(Sisters[0]) == Direction_Last,
              "mismatch size");

struct Coordinate {
 public:
  int m_row;
  int m_column;

  bool operator==(const Coordinate& other) const {
    return m_row == other.m_row &&
           m_column == other.m_column;
  }

  Coordinate operator+(const Coordinate& other) const {
    Coordinate ret;
    ret.m_row = m_row + other.m_row;
    ret.m_column = m_column + other.m_column;
    return ret;
  }
};
static_assert(std::is_standard_layout<Coordinate>::value, "no vtable");

constexpr Coordinate Offsets[] = {
    // row, column
    { 0, 1 },   // Direction::RIGHT
    { 1, 1 },   // Direction.UP_RIGHT
    { 1, 0 },   // Direction.UP_LEFT
    { 0, -1 },  // Direction.LEFT
    { -1, -1 },  // Direction.DOWN_LEFT
    { -1, 0 },  // Direction.DOWN_RIGHT
};
static_assert(sizeof(Offsets) / sizeof(Offsets[0]) == Direction_Last,
              "mismatch size");

struct core_state {
 public:
  CellState board_[abalone_core::kNumRows][abalone_core::kNumCols];
  int turn_;
  CellState outcome_ = CellState::Invalid;  // winner (draw: empty)

  inline CellState ToPlay() const { return CellState(turn_ % 2); }
  void Reset(const CellState _init_pattern[kNumRows][kNumCols] =
             ABALONE_INIT_CLASSIC);
  // @param _marble_advantage: if no winner at end of game, winner is the
  //   player with more marbles
  // @return tuple<is_finished, new_outcome>
  std::tuple<bool, CellState> Eval(int _marbles_to_win = kMarblesToWin,
                                   int _game_length = -1,
                                   bool _marble_advantage = false) const;

  friend struct Move;
};
static_assert(std::is_standard_layout<core_state>::value, "no vtable");
std::ostream& operator<<(std::ostream& _os, core_state const& _arg);

typedef int core_Action;
constexpr core_Action kActionMin = 0;
constexpr core_Action kActionMax = kNumCells * kNumActionsPerCell;

// Valid single Move: m_end-m_start == Offset[m_direction].
// For slide moves: 1 <= length(end-start) <= 2. The slide direction is
// resolved as being to the left of the move direction (see Sisters[]).
struct Move {
 public:
  Direction m_direction;
  Coordinate m_start;
  Coordinate m_end;

  inline bool operator==(const Move& other) const {
    return this->m_direction == other.m_direction &&
           this->m_start == other.m_start &&
           this->m_end == other.m_end;
  }

  bool IsValid(const core_state& _state) const;

  void Apply(core_state& _state, int _marbles_to_win = kMarblesToWin,
             int _game_length = kHistoryMax,
             bool _marble_advantage = false) const;

  std::string ToString() const;

  static std::tuple<bool, Move> FromString(const std::string& _str);

  static Move ActionToMove(core_Action moveId);

  static core_Action MoveToAction(const Move& move);

 protected:
  bool _IsValidSlide(const core_state& _state) const;

  void _ApplyParallelMove(core_state& _state, int _dr, int _dc) const;

  void _ApplySingleMove(core_state& _state, int _dr, int _dc) const;
};
static_assert(std::is_standard_layout<Move>::value, "no vtable");
std::ostream& operator<<(std::ostream& os, const Move& _move);

}  // namespace abalone_core

#endif  // OPEN_SPIEL_GAMES_ABALONE_ABALONE_CORE_H_
