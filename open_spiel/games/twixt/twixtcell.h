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

#ifndef OPEN_SPIEL_GAMES_TWIXT_TWIXTCELL_H_
#define OPEN_SPIEL_GAMES_TWIXT_TWIXTCELL_H_

#include <utility>
#include "open_spiel/spiel.h"

typedef std::pair<int, int> Move;
typedef std::pair<Move, int> Link;

namespace open_spiel {
namespace twixt {

enum Border { kStart, kEnd, kMaxBorder };

const open_spiel::Player kRedPlayer = 0;
const open_spiel::Player kBluePlayer = 1;
const int kNumPlayers = 2;

// eight directions of links from 0 to 7
enum Compass {
  kNNE,  // North-North-East, 1 right, 2 up
  kENE,  // East-North-East,  2 right, 1 up
  kESE,  // East-South-East,  2 right, 1 down
  kSSE,  // South-South-East, 1 right, 2 down
  kSSW,  // South-South-West, 1 left,  2 down
  kWSW,  // West-South-West,  2 left,  1 down
  kWNW,  // West-North-West,  2 left,  1 up
  kNNW,  // North-North-West, 1 left,  2 up
  kMaxCompass
};

class Cell {
 private:
  int mColor;
  // bitmap of outgoing links from this cell
  int mLinks = 0;
  // bitmap of candidates of a player
  // (neighbors that are empty or have same color)
  int mCandidates[kNumPlayers] = {0, 0};
  // bitmap of neighbors of same color that are blocked
  int mBlockedNeighbors = 0;
  // array of neighbor tuples
  // (cells in knight's move distance that are on board)
  Move mNeighbors[kMaxCompass];
  // indicator if cell is linked to START|END border of player 0|1
  bool mLinkedToBorder[kNumPlayers][kMaxBorder] = {{false, false},
                                                   {false, false}};

 public:
  int getColor() const { return mColor; }
  void setColor(int color) { mColor = color; }

  void setLink(int dir) { mLinks |= (1UL << dir); }
  int getLinks() const { return mLinks; }
  bool isLinked(int cand) const { return mLinks & cand; }
  bool hasLink(int dir) const { return mLinks & (1UL << dir); }
  bool hasLinks() const { return mLinks > 0; }

  int getCandidates(int player) { return mCandidates[player]; }
  bool isCandidate(int player, int cand) const {
    return mCandidates[player] & cand;
  }
  void setCandidate(int player, int dir) {
    mCandidates[player] |= (1UL << dir);
  }
  void deleteCandidate(int player, int cand) {
    mCandidates[player] &= ~(cand);
  }
  void deleteCandidate(int dir) {
    mCandidates[kRedPlayer] &= ~(1UL << dir);
    mCandidates[kBluePlayer] &= ~(1UL << dir);
  }

  void setBlockedNeighbor(int dir) { mBlockedNeighbors |= (1UL << dir); }
  bool hasBlockedNeighbors() const { return mBlockedNeighbors > 0; }

  Move getNeighbor(int dir) const { return mNeighbors[dir]; }
  void setNeighbor(int dir, Move c) { mNeighbors[dir] = c; }

  void setLinkedToBorder(int player, int border) {
    mLinkedToBorder[player][border] = true;
  }

  bool isLinkedToBorder(int player, int border) const {
    return mLinkedToBorder[player][border];
  }
};

}  // namespace twixt
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_TWIXT_TWIXTCELL_H_
