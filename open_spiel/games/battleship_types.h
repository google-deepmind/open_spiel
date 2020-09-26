// Copyright 2020 DeepMind Technologies Ltd. All rights reserved.
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
// limitations under the License.#include "open_spiel/games/battleship.h"

#ifndef OPEN_SPIEL_GAMES_BATTLESHIP_TYPES_H_
#define OPEN_SPIEL_GAMES_BATTLESHIP_TYPES_H_

#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/abseil-cpp/absl/strings/str_format.h"
#include "open_spiel/abseil-cpp/absl/types/variant.h"
#include "open_spiel/spiel.h"

namespace open_spiel {
namespace battleship {

// A cell of a generic player's board.
struct Cell {
  int row;
  int col;

  std::string ToString() const { return absl::StrFormat("%d_%d", row, col); }
  bool operator==(const Cell& other) const {
    return row == other.row && col == other.col;
  }
  bool operator<(const Cell& other) const {
    return (row <= other.row) || (row == other.row && col <= other.col);
  }
};

// Represents a shot action. We store the coordinates of the cell that was hit.
using Shot = Cell;

struct Ship {
  int id;  // Unique identifier of the ship
  int length;
  double value;
};

// Represents the placement of a ship.
//
// Ships can be placed either horizontal or vertical on the board. When the ship
// is placed horizontally, we store the *leftmost* cell of the placement as the
// corner. When the ship is placed vertically, the corner is the *topmost*.
class ShipPlacement final {
 public:
  enum Direction { Horizontal = 0, Vertical = 1 };

  ShipPlacement(const Direction direction, const Ship& ship,
                const Cell& tl_corner);

  // Returns true if the the ship falls over a specific cell.
  bool CoversCell(const Cell& cell) const;

  Cell TopLeftCorner() const;
  Cell BottomRightCorner() const;

  // Checks whether two ship placements intersect on at least one cell.
  bool OverlapsWith(const ShipPlacement& other) const;

  // Gives a string representation of the ship placement, useful to the
  // ActionToString method.
  //
  // For a ship placed horizontally with the top left corner in (2,3), the
  // string representation is `h_2_3`. For vertical placements, the first
  // character is a `v` instead of an `h`.
  std::string ToString() const;

  Direction direction;
  Ship ship;

 private:
  Cell tl_corner_;
};

struct GameMove {
  Player player;
  absl::variant<ShipPlacement, Shot> action;
};

struct BattleshipConfiguration {
  int board_width;
  int board_height;

  // It is assumed that each agent has the same set of ships. So, each ship is
  // only included once instead of being duplicated for each player.
  std::vector<Ship> ships;

  // Number of shots **each player** can use.
  int num_shots;

  // If false, players are forbidden from shooting the same cell more than once.
  bool allow_repeated_shots;

  // See the description of the game in `battleship.h` for details of how
  // the payoffs of the players are computed.
  double loss_multiplier;
};
}  // namespace battleship
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_BATTLESHIP_TYPES_H_
