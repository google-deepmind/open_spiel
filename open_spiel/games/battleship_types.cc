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

std::string ShipPlacement::ToString() const {
  const char direction_char = direction == Direction::Horizontal ? 'h' : 'v';

  return absl::StrFormat("%c_%d_%d", direction_char, tl_corner_.row,
                         tl_corner_.col);
}

}  // namespace battleship
}  // namespace open_spiel
