
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
#include <string>
#include <vector>
#include <utility>
#include <set>

#include "open_spiel/games/twixt/twixtcell.h"
#include "open_spiel/spiel.h"

namespace open_spiel {
namespace twixt {

const int kMinBoardSize = 5;
const int kMaxBoardSize = 24;
const int kDefaultBoardSize = 8;

const bool kDefaultAnsiColorOutput = true;

const double kMinDiscount = 0.0;
const double kMaxDiscount = 1.0;
const double kDefaultDiscount = kMaxDiscount;

// 8 link descriptors store the properties of a link direction
struct {
  Move offsets;  // offset of the target peg, e.g. (2, -1) for ENE
  std::vector<std::pair<Move, int>> blockingLinks;
} typedef LinkDescriptor;

// Tensor has 2 * 3 planes of size bordSize * (boardSize-2)
// see ObservationTensor
const int kNumPlanes = 6;

enum Result { kOpen, kRedWin, kBlueWin, kDraw };

enum Color { kRedColor, kBlueColor, kEmpty, kOffBoard };

// blockerMap stores set of blocking links for each link
static std::map<Link, std::set<Link>> blockerMap;

inline std::set<Link> *getBlockers(Link link) { return &blockerMap[link]; }

inline void pushBlocker(Link link, Link blockedLink) {
  blockerMap[link].insert(blockedLink);
}

inline void deleteBlocker(Link link, Link blockedLink) {
  blockerMap[link].erase(blockedLink);
}

inline void clearBlocker() { blockerMap.clear(); }

class Board {
 private:
  int mMoveCounter = 0;
  bool mSwapped = false;
  Move mMoveOne;
  int mResult = kOpen;
  std::vector<std::vector<Cell>> mCell;
  int mSize;  // length of a side of the board
  bool mAnsiColorOutput;
  std::vector<Action> mLegalActions[kNumPlayers];

  void setSize(int size) { mSize = size; }

  bool getAnsiColorOutput() const { return mAnsiColorOutput; }
  void setAnsiColorOutput(bool ansiColorOutput) {
    mAnsiColorOutput = ansiColorOutput;
  }

  void setResult(int result) { mResult = result; }

  bool getSwapped() const { return mSwapped; }
  void setSwapped(bool swapped) { mSwapped = swapped; }

  Move getMoveOne() const { return mMoveOne; }
  void setMoveOne(Move move) { mMoveOne = move; }

  void incMoveCounter() { mMoveCounter++; }

  bool hasLegalActions(Player player) const {
    return mLegalActions[player].size() > 0;
  }

  void removeLegalAction(Player, Move);

  void updateResult(Player, Move);
  void undoFirstMove();

  void initializeCells(bool);
  void initializeCandidates(Move, Cell *, bool);
  void initializeBlockerMap(Move, int, LinkDescriptor *);

  void initializeLegalActions();

  void setPegAndLinks(Player, Move);
  void exploreLocalGraph(Player, Cell *, enum Border);

  void appendLinkChar(std::string *, Move, enum Compass, std::string) const;
  void appendColorString(std::string *, std::string, std::string) const;
  void appendPegChar(std::string *, Move) const;

  void appendBeforeRow(std::string *, Move) const;
  void appendPegRow(std::string *, Move) const;
  void appendAfterRow(std::string *, Move) const;

  bool moveIsOnBorder(Player, Move) const;
  bool moveIsOffBoard(Move) const;

  Action stringToAction(std::string s) const;

 public:
  ~Board() {}
  Board() {}
  Board(int, bool);

  // std::string actionToString(Action) const;
  int getSize() const { return mSize; }
  std::string toString() const;
  int getResult() const { return mResult; }
  int getMoveCounter() const { return mMoveCounter; }
  std::vector<Action> getLegalActions(Player player) const {
    return mLegalActions[player];
  }
  void applyAction(Player, Action);
  Cell *getCell(Move move) { return &mCell[move.first][move.second]; }
  const Cell *getConstCell(Move move) const {
    return &mCell[move.first][move.second];
  }
  Move actionToMove(open_spiel::Player player, Action action) const;
  Action moveToAction(Player player, Move move) const;
  Move getTensorMove(Move move, int turn) const;
};

// twixt board:
// * the board has mBoardSize x mBoardSize cells
// * the x-axis (cols) points right,
// * the y axis (rows) points up
// * coords [col,row] start at the lower left corner [0,0]
// * coord labels c3, f4, d2, etc. start at the upper left corner (a1)
// * player 0 == 'x', red color, plays top/bottom
// * player 1 == 'o', blue color, plays left/right
// * move is labeled player + coord label, e.g. xd4
// * empty cell == 2
// * corner cell == 3
//
// example 8 x 8 board: red peg at [2,3] == xc5 == action=26
//                      red peg at [3,5] == xd3 == action=21
//                     blue peg at [5,3] == of5 == action=29
//
//     a   b   c   d   e   f   g   h
//    ------------------------------
// 1 | 3   2   2   2   2   2   2   3 |
//   |                               |
// 2 | 2   2   2   2   2   2   2   2 |
//   |                               |
// 3 | 2   2   2   0   2   2   2   2 |
//   |                               |
// 4 | 2   2   2   2   2   2   2   2 |
//   |                               |
// 5 | 2   2   0   2   2   1   2   2 |
//   |                               |
// 6 | 2   2   2   2   2   2   2   2 |
//   |                               |
// 7 | 2   2   2   2   2   2   2   2 |
//   |                               |
// 8 | 3   2   2   2   2   2   2   3 |
//     ------------------------------

// there's a red link from c5 to d3:
// cell[2][3].links = 00000001  (bit 1 set for NNE direction)
// cell[3][5].links = 00010000  (bit 5 set for SSW direction)

// Actions are indexed from 0 to boardSize * (boardSize-2) from the player's
// perspective:

// player 0 actions:
//     a   b   c   d   e   f   g   h
//    ------------------------------
// 1 |     7  15  23  31  39  47     |
//   |                               |
// 2 |     6  14  22  30  38  46     |
//   |                               |
// 3 |     5  13  21  29  37  45     |
//   |                               |
// 4 |     4  12  20  28  36  44     |
//   |                               |
// 5 |     3  11  19  27  35  43     |
//   |                               |
// 6 |     2  10  18  26  34  42     |
//   |                               |
// 7 |     1   9  17  25  33  41     |
//   |                               |
// 8 |     0   8  16  24  32  40     |
//     ------------------------------

// player 1 actions:
//     a   b   c   d   e   f   g   h
//    ------------------------------
// 1 |                               |
//   |                               |
// 2 | 0   1   2   3   4   5   6   7 |
//   |                               |
// 3 | 8   9  10  11  12  13  14  15 |
//   |                               |
// 4 |16  17  18  19  20  21  22  23 |
//   |                               |
// 5 |24  25  26  27  28  29  30  31 |
//   |                               |
// 6 |32  33  34  35  36  37  38  39 |
//   |                               |
// 7 |40  41  42  43  44  45  46  47 |
//   |                               |
// 8 |                               |
//     ------------------------------

//  mapping move to player 0 action:
//  [c,r] => (c-1) * size + r,
//  e.g.: xd6 == [3,2] => (3-1) * 8 + 2 == 18
//  xd6 == action 18 of player 0
//
//  mapping move to player 1 action:
//  [c,r] => (size-r-2) * size + c,
//  e.g.: od6 == [3,2] => (8-2-2) * 8 + 3 == 35
//  od6 == action 35 of player 1

}  // namespace twixt
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_TWIXT_TWIXTBOARD_H_
