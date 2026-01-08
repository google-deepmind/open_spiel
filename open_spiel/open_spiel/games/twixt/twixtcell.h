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

#include "open_spiel/spiel_utils.h"

struct Position {
  int x;
  int y;
  Position operator+(const Position &p) { return {x + p.x, y + p.y}; }
  bool operator==(const Position &p) const { return x == p.x && y == p.y; }
  bool operator<(const Position &p) const {
    return x < p.x || (x == p.x && y < p.y);
  }
};

struct Link {
  Position position;
  int direction;
  bool operator==(const Link &l) const {
    return position == l.position && direction == l.direction;
  }
  bool operator<(const Link &l) const {
    return position < l.position ||
           (position == l.position && direction < l.direction);
  }
};

namespace open_spiel {
namespace twixt {

enum Border { kStart, kEnd, kMaxBorder };

const open_spiel::Player kRedPlayer = 0;
const open_spiel::Player kBluePlayer = 1;
const int kNumPlayers = 2;

// eight directions of links from 0 to 7:q!

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
 public:
  int color() const { return color_; }
  void set_color(int color) { color_ = color; }
  void set_link(int dir) { links_ |= (1UL << dir); }
  int links() const { return links_; }

  bool HasLink(int dir) const { return links_ & (1UL << dir); }
  bool HasLinks() const { return links_ > 0; }

  void SetBlockedNeighbor(int dir) { blocked_neighbors_ |= (1UL << dir); }
  bool HasBlockedNeighbors() const { return blocked_neighbors_ > 0; }
  bool HasBlockedNeighborsEast() const {
    return (blocked_neighbors_ & 15UL) > 0;
  }

  Position GetNeighbor(int dir) const { return neighbors_[dir]; }
  void SetNeighbor(int dir, Position c) { neighbors_[dir] = c; }

  void SetLinkedToBorder(int player, int border) {
    linked_to_border_[player][border] = true;
  }

  bool IsLinkedToBorder(int player, int border) const {
    return linked_to_border_[player][border];
  }

 private:
  int color_;
  // bitmap of outgoing links from this cell
  int links_ = 0;
  // bitmap of neighbors same color that are blocked
  int blocked_neighbors_ = 0;
  // array of neighbor tuples
  // (cells in knight's move distance that are on board)
  Position neighbors_[kMaxCompass];
  // indicator if cell is linked to START|END border of player 0|1
  bool linked_to_border_[kNumPlayers][kMaxBorder] = {{false, false},
                                                     {false, false}};
};

}  // namespace twixt
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_TWIXT_TWIXTCELL_H_
