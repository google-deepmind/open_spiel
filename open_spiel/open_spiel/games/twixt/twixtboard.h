
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

#ifndef OPEN_SPIEL_GAMES_TWIXT_TWIXTBOARD_H_
#define OPEN_SPIEL_GAMES_TWIXT_TWIXTBOARD_H_

#include <map>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "open_spiel/games/twixt/twixtcell.h"
#include "open_spiel/spiel.h"

namespace open_spiel {
namespace twixt {

const int kMinBoardSize = 5;
const int kMaxBoardSize = 24;
const int kDefaultBoardSize = 8;

const bool kDefaultAnsiColorOutput = true;

// 8 link descriptors store the properties of a link direction
struct {
  Position offsets;  // offset of the target peg, e.g. (2, -1) for ENE
  std::vector<Link> blocking_links;
} typedef LinkDescriptor;

// Tensor has 2 * 6 planes of size bordSize * (boardSize-2)
// see ObservationTensor
const int kNumPlanes = 12;

enum Result { kOpen, kRedWin, kBlueWin, kDraw };

enum Color { kRedColor, kBlueColor, kEmpty, kOffBoard };

class Board {
 public:
  ~Board() {}
  Board() {}
  Board(int, bool);

  int size() const { return size_; }
  std::string ToString() const;
  int result() const { return result_; }
  int move_counter() const { return move_counter_; }
  std::vector<Action> GetLegalActions(Player player) const {
    return legal_actions_[player];
  }
  void ApplyAction(Player, Action);
  Cell& GetCell(Position position) { return cell_[position.x][position.y]; }
  const Cell& GetConstCell(Position position) const {
    return cell_[position.x][position.y];
  }
  Position ActionToPosition(Action action) const;
  Action PositionToAction(Position position) const;
  Position GetTensorPosition(Position position, bool turn) const;

 private:
  int move_counter_ = 0;
  bool swapped_ = false;
  Position move_one_;
  int result_ = kOpen;
  std::vector<std::vector<Cell>> cell_;
  int size_;  // length of a side of the board
  bool ansi_color_output_;
  std::vector<Action> legal_actions_[kNumPlayers];

  void set_size(int size) { size_ = size; }

  bool ansi_color_output() const { return ansi_color_output_; }
  void set_ansi_color_output(bool ansi_color_output) {
    ansi_color_output_ = ansi_color_output;
  }

  void set_result(int result) { result_ = result; }

  bool swapped() const { return swapped_; }
  void set_swapped(bool swapped) { swapped_ = swapped; }

  Position move_one() const { return move_one_; }
  void set_move_one(Position move) { move_one_ = move; }

  void IncMoveCounter() { move_counter_++; }

  bool HasLegalActions(Player player) const {
    return legal_actions_[player].size() > 0;
  }

  void RemoveLegalAction(Player, Position);

  void UpdateResult(Player, Position);
  void UndoFirstMove();

  void InitializeCells(bool);
  void InitializeNeighbors(Position, Cell&, bool);
  void InitializeBlockerMap(Position, int, const LinkDescriptor&);

  void InitializeLegalActions();

  void SetPegAndLinks(Player, Position);
  void ExploreLocalGraph(Player, Cell&, enum Border, std::set<Cell*>);

  void AppendLinkChar(std::string&, Position, enum Compass, std::string) const;
  void AppendColorString(std::string&, std::string, std::string) const;
  void AppendPegChar(std::string&, Position) const;

  void AppendBeforeRow(std::string&, Position) const;
  void AppendPegRow(std::string&, Position) const;
  void AppendAfterRow(std::string&, Position) const;

  bool PositionIsOnBorder(Player, Position) const;
  bool PositionIsOffBoard(Position) const;

  Action StringToAction(std::string s) const;
};

// used to construct new entries in BlockerMap
class LinkHashFunction {
 public:
  size_t operator()(const Link& link) const {
    return link.position.x * 10000 + link.position.y * 100 + link.direction;
  }
};

// stores for each link the set of links that could block it (i.e. cross it)
class BlockerMap {
 public:
  static const std::set<Link>& GetBlockers(Link link);
  static void PushBlocker(Link link, Link blocked_link);
  static void DeleteBlocker(Link link, Link blocked_link);
  static void ClearBlocker();

 private:
  static std::unordered_map<Link, std::set<Link>, LinkHashFunction> map_;
};

// twixt board:
// * the board has board_size_ * board_size_ cells
// * the x-axis (cols) points right,
// * the y axis (rows) points up
// * coord labels c3, f4, d2, etc. start at the upper left corner (a1)
// * player 0, 'x', red color, plays top/bottom
// * player 1, 'o', blue color, plays left/right
// * positions are labeled: col letter + row number, e.g. d4
// * moves are labeled: player label + col letter + row number, e.g. xd4
// * empty cell code = 2
// * corner cell code = 3
//
// example 8 x 8 board:
//   move: xc5, player 0 action: 19, red peg at [2,3]
//   move: of5, player 1 action: 43, blue peg at [5,3]
//   move: xd3, player 0 action: 29, red peg at [3,5]
//         link from [2,3] to [3,5]
//         cell[2][3].links = 00000001  (bit 1 set for NNE direction)
//         cell[3][5].links = 00010000  (bit 5 set for SSW direction)
//
//     a   b   c   d   e   f   g   h
//  7  3|  2   2   2   2   2   2 | 3  1
//    --|------------------------|--
//  6  2|  2   2   2   2   2   2 | 2  2
//      |                        |
//  5  2|  2   2  [0]  2   2   2 | 2  3
//      |                        |
//  4  2|  2   2   2   2   2   2 | 2  4
//      |                        |
//  3  2|  2  [0]  2   2  [1]  1 | 2  5
//      |                        |
//  2  2|  2   2   2   2   2   2 | 2  6
//      |                        |
//  1  2|  2   2   2   2   2   2 | 2  7
//    --|------------------------|--
//  0  3|  2   2   2   2   2   2 | 3  8
//     0   1   2   3   4   5   6   7
//
// Actions are indexed from 0 to (board_size_ ** 2) - 1
// except the corners (0, 7, 56, 63) which are not legal actions.
//
//     a   b   c   d   e   f   g   h
//  7   | 15  23  31  39  47  55 |    1
//    --|------------------------|--
//  6  6| 14  22  30  38  46  54 |62  2
//      |                        |
//  5  5| 13  21 [29] 37  45  53 |61  3
//      |                        |
//  4  4| 12  20  28  36  44  52 |60  4
//      |                        |
//  3  3| 11 [19] 27  35 [43] 51 |59  5
//      |                        |
//  2  2| 10  18  26  34  42  50 |58  6
//      |                        |
//  1  1|  9  17  25  33  41  49 |57  7
//    --|------------------------|--
//  0   |  8  16  24  32  40  48 |    8
//     0   1   2   3   4   5   6   7
//
//  mapping move to action: [c,r] => c * size + r
//  xd6 == [2,3] => 2 * 8 + 3 == 19

}  // namespace twixt
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_TWIXT_TWIXTBOARD_H_
