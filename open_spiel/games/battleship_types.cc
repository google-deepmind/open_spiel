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

#include "open_spiel/games/battleship_types.h"

namespace open_spiel {
namespace battleship {

namespace {
bool IsOverlappingPlacement(const std::vector<ShipPlacement>& placement) {
  for (int index = 1; index < placement.size(); ++index) {
    for (int other = 0; other < index; ++other) {
      if (placement.at(index).OverlapsWith(placement.at(other))) {
        return true;
      }
    }
  }

  return false;
}
}  // namespace

bool ExistsFeasiblePlacement(const BattleshipConfiguration& conf,
                             std::vector<ShipPlacement>* partial_placement) {
  // Debug-time check of preconditions.
  SPIEL_DCHECK_LE(partial_placement->size(), conf.ships.size());
  SPIEL_DCHECK_FALSE(IsOverlappingPlacement(*partial_placement));
  for (int index = 0; index < partial_placement->size(); ++index) {
    const ShipPlacement& placement = partial_placement->at(index);

    SPIEL_CHECK_EQ(placement.ship.id, conf.ships.at(index).id);
    SPIEL_CHECK_TRUE(
        placement.IsWithinBounds(conf.board_width, conf.board_height));
  }

  if (partial_placement->size() == conf.ships.size()) {
    // All ships have been placed. The placement is valid because of the
    // precondition.
    return true;
  } else {
    // We try to place the next ship in the board. We start by trying to place
    // the ship horizontally.
    //
    // Because of the precondition, partial_placement is a placement of a prefix
    // of the ships in conf.ships. Hence, the next ship that needs to be placed
    // is simply:
    const Ship& ship = conf.ships.at(partial_placement->size());

    // -- Horizontal placement.
    for (int row = 0; row < conf.board_height; ++row) {
      for (int col = 0; col < conf.board_width - ship.length + 1; ++col) {
        // First, we append the placement of the next ship to the partial
        // placement vector.
        partial_placement->push_back({ShipPlacement::Direction::Horizontal,
                                      /* ship = */ ship,
                                      /* tl_corner = */ Cell{row, col}});
        if (!IsOverlappingPlacement(*partial_placement) &&
            ExistsFeasiblePlacement(conf, partial_placement)) {
          // The new partial placement led to a solution. We honor the
          // postcondition and early-return sucess.
          partial_placement->pop_back();
          return true;
        } else {
          // The new partial placement does not lead to a solution. We remove
          // the placement and continue with the next placement.
          partial_placement->pop_back();
        }
      }
    }

    // -- Vertical placement.
    for (int row = 0; row < conf.board_height - ship.length + 1; ++row) {
      for (int col = 0; col < conf.board_width; ++col) {
        // First, we append the placement of the next ship to the partial
        // placement vector.
        partial_placement->push_back({ShipPlacement::Direction::Vertical,
                                      /* ship = */ ship,
                                      /* tl_corner = */ Cell{row, col}});
        if (!IsOverlappingPlacement(*partial_placement) &&
            ExistsFeasiblePlacement(conf, partial_placement)) {
          // The new partial placement led to a solution. We honor the
          // postcondition and early-return sucess.
          partial_placement->pop_back();
          return true;
        } else {
          // The new partial placement does not lead to a solution. We remove
          // the placement and continue with the next placement.
          partial_placement->pop_back();
        }
      }
    }
  }

  return false;
}

CellAndDirection::CellAndDirection(const Direction direction,
                                   const Cell& tl_corner)
    : direction(direction), tl_corner_(tl_corner) {
  SPIEL_CHECK_GE(tl_corner.row, 0);
  SPIEL_CHECK_GE(tl_corner.col, 0);
}

ShipPlacement::ShipPlacement(const Direction dir, const Ship& ship,
                             const Cell& tl_corner)
    : CellAndDirection(dir, tl_corner), ship(ship) {
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

bool ShipPlacement::IsWithinBounds(const int board_width,
                                   const int board_height) const {
  const Cell tl_corner = TopLeftCorner();
  const Cell br_corner = BottomRightCorner();

  return (tl_corner.row >= 0 && tl_corner.row < board_height) &&
         (br_corner.row >= 0 && br_corner.row < board_height) &&
         (tl_corner.col >= 0 && tl_corner.col < board_width) &&
         (br_corner.col >= 0 && br_corner.col < board_width);
}

std::string ShipPlacement::ToString() const {
  const char direction_char = direction == Direction::Horizontal ? 'h' : 'v';

  return absl::StrFormat("%c_%d_%d", direction_char, tl_corner_.row,
                         tl_corner_.col);
}

}  // namespace battleship
}  // namespace open_spiel
