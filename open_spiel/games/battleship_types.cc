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

#include "open_spiel/games/battleship_types.h"

namespace open_spiel {
namespace battleship {

ShipPlacement::ShipPlacement(const Direction direction, const Ship& ship,
                             const Cell& tl_corner)
    : direction(direction), ship(ship), tl_corner_(tl_corner) {
  SPIEL_CHECK_GE(tl_corner.row, 0);
  SPIEL_CHECK_GE(tl_corner.col, 0);
  SPIEL_CHECK_GE(ship.length, 1);
}

bool ShipPlacement::CoversCell(const Cell& cell) const {
  if (direction == Direction::Horizontal) {
    return cell.row == tl_corner_.row && cell.col >= tl_corner_.col &&
           cell.col < tl_corner_.col + ship.length;
  } else {
    SPIEL_CHECK_EQ(direction, Direction::Vertical);
    return cell.col == tl_corner_.col && cell.row >= tl_corner_.row &&
           cell.row < tl_corner_.row + ship.length;
  }
}

Cell ShipPlacement::TopLeftCorner() const { return tl_corner_; }
Cell ShipPlacement::BottomRightCorner() const {
  if (direction == Direction::Horizontal) {
    return Cell{tl_corner_.row, tl_corner_.col + ship.length - 1};
  } else {
    SPIEL_CHECK_EQ(direction, Direction::Vertical);
    return Cell{tl_corner_.row + ship.length - 1, tl_corner_.col};
  }
}

bool ShipPlacement::OverlapsWith(const ShipPlacement& other) const {
  if (other.BottomRightCorner().row < TopLeftCorner().row) {
    // `other` is completely above `this`.
    return false;
  } else if (other.TopLeftCorner().row > BottomRightCorner().row) {
    // `other` is completely below `this`.
    return false;
  } else if (other.BottomRightCorner().col < TopLeftCorner().col) {
    // `other` is completely to the left of `this`.
    return false;
  } else if (other.TopLeftCorner().col > BottomRightCorner().col) {
    // `other` is completely to the right of `this`.
    return false;
  }
  return true;
}

std::string ShipPlacement::ToString() const {
  const char direction_char = direction == Direction::Horizontal ? 'h' : 'v';

  return absl::StrFormat("%c_%d_%d", direction_char, tl_corner_.row,
                         tl_corner_.col);
}

}  // namespace battleship
}  // namespace open_spiel
