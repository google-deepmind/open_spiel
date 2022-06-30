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

#ifndef OPEN_SPIEL_GAMES_HAVANNAH_H_
#define OPEN_SPIEL_GAMES_HAVANNAH_H_

#include <array>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "open_spiel/spiel.h"

// https://en.wikipedia.org/wiki/Havannah
//
// Parameters:
//   "board_size"        int     radius of the board   (default = 8)
//   "swap"              bool    Whether to allow the swap rule.
//   "ansi_color_output" bool    Whether to color the output for a terminal.

namespace open_spiel {
namespace havannah {

inline constexpr int kNumPlayers = 2;
inline constexpr int kDefaultBoardSize = 8;
inline constexpr int kMaxNeighbors =
    6;  // Maximum number of neighbors for a cell
inline constexpr int kCellStates = 1 + kNumPlayers;

enum HavannahPlayer : uint8_t {
  kPlayer1,
  kPlayer2,
  kPlayerNone,
  kPlayerDraw,
  kPlayerInvalid,
};

enum MoveSpecial {
  kMoveNone = -1,
  kMoveUnknown = -2,
  kMoveOffset = -3,
};

inline int CalcXY(int x, int y, int board_size) {
  int diameter = board_size * 2 - 1;
  if (x >= 0 && y >= 0 && x < diameter && y < diameter &&
      (y - x < board_size) && (x - y < board_size)) {
    return x + y * diameter;
  } else {
    return kMoveUnknown;
  }
}

struct Move {
  int8_t x, y;  // The x,y coordinates
  int16_t xy;  // precomputed x + y * board_diameter as an index into the array.

  inline constexpr Move(MoveSpecial m = kMoveUnknown) : x(-1), y(-1), xy(m) {}
  inline constexpr Move(int x_, int y_, MoveSpecial m) : x(x_), y(y_), xy(m) {}
  Move(int x_, int y_, int board_size)
      : x(x_), y(y_), xy(CalcXY(x_, y_, board_size)) {}

  std::string ToString() const;

  bool operator==(const Move& b) const { return xy == b.xy; }
  bool operator!=(const Move& b) const { return xy != b.xy; }
  bool operator==(const MoveSpecial& b) const { return xy == b; }
  bool operator!=(const MoveSpecial& b) const { return xy != b; }

  // Whether the move is valid and on the board. May be invalid because it is
  // a MoveSpecial, in the cut-off corners, or otherwise off the board.
  bool OnBoard() const { return xy >= 0; }

  // Flags for which edge/corner this move is part of.
  int Edge(int board_size) const;
  int Corner(int board_size) const;
};

// List of neighbors of a cell: [cell][direction]
typedef std::vector<std::array<Move, kMaxNeighbors>> NeighborList;

// State of an in-play game.
class HavannahState : public State {
  // Represents a single cell on the board, as well as the structures needed for
  // groups of cells. Groups of cells are defined by a union-find structure
  // embedded in the array of cells. Following the `parent` indices will lead to
  // the group leader which has the up to date size, corner and edge
  // connectivity of that group. Size, corner and edge are not valid for any
  // cell that is not a group leader.
  struct Cell {
    // Who controls this cell.
    HavannahPlayer player;

    // Whether this cell is marked/visited in a ring search. Should always be
    // false except while running CheckRingDFS.
    bool mark;

    // A parent index to allow finding the group leader. It is the leader of the
    // group if it points to itself. Allows path compression to shorten the path
    // from a direct parent to the leader.
    uint16_t parent;

    // These three are only defined for the group leader's cell.
    uint16_t size;   // Size of this group of cells.
    uint8_t corner;  // A bitset of which corners this group is connected to.
    uint8_t edge;    // A bitset of which edges this group is connected to.

    Cell() {}
    Cell(HavannahPlayer player_, int parent_, int corner_, int edge_)
        : player(player_),
          mark(false),
          parent(parent_),
          size(1),
          corner(corner_),
          edge(edge_) {}

    // How many corner or edges this group of cell is connected to. Only defined
    // if called on the group leader.
    int NumCorners() const;
    int NumEdges() const;
  };

 public:
  HavannahState(std::shared_ptr<const Game> game, int board_size,
                bool ansi_color_output = false, bool allow_swap = false);

  HavannahState(const HavannahState&) = default;

  Player CurrentPlayer() const override {
    return IsTerminal() ? kTerminalPlayerId : static_cast<int>(current_player_);
  }
  std::string ActionToString(Player player, Action action_id) const override;
  std::string ToString() const override;
  bool IsTerminal() const override { return outcome_ != kPlayerNone; }
  std::vector<double> Returns() const override;
  std::string InformationStateString(Player player) const override;
  std::string ObservationString(Player player) const override;

  // A 3d tensor, 3 player-relative one-hot 2d planes. The layers are: the
  // specified player, the other player, and empty.
  void ObservationTensor(Player player,
                         absl::Span<float> values) const override;
  std::unique_ptr<State> Clone() const override;
  std::vector<Action> LegalActions() const override;

 protected:
  void DoApplyAction(Action action) override;

  // Find the leader of the group. Not const due to union-find path compression.
  int FindGroupLeader(int cell);

  // Join the groups of two positions, propagating group size, and edge/corner
  // connections. Returns true if they were already the same group.
  bool JoinGroups(int cell_a, int cell_b);

  // Do a depth first search for a ring starting at `move`.
  // `left` and `right give the direction bounds for the search. A valid ring
  // won't take any sharp turns, only going in one of the 3 forward directions.
  // The only exception is the very beginning where we don't know the direction
  // and it's valid to search in all 6 directions. 4 is enough though, since any
  // valid ring can't start and end in the 2 next to each other while still
  // going through `move.`
  bool CheckRingDFS(const Move& move, int left, int right);

  // Turn an action id into a `Move` with an x,y.
  Move ActionToMove(Action action_id) const;

  bool AllowSwap() const;

 private:
  std::vector<Cell> board_;
  HavannahPlayer current_player_ = kPlayer1;
  HavannahPlayer outcome_ = kPlayerNone;
  const int board_size_;
  const int board_diameter_;
  const int valid_cells_;
  int moves_made_ = 0;
  Move last_move_ = kMoveNone;
  const NeighborList& neighbors_;
  const bool ansi_color_output_;
  const bool allow_swap_;
};

// Game object.
class HavannahGame : public Game {
 public:
  explicit HavannahGame(const GameParameters& params);

  int NumDistinctActions() const override {
    // Really diameter^2 - size*(size-1), but that's harder to represent, so
    // the extra actions in the corners are never legal.
    return Diameter() * Diameter();
  }
  std::unique_ptr<State> NewInitialState() const override {
    return std::unique_ptr<State>(
        new HavannahState(shared_from_this(), board_size_, ansi_color_output_,
                          allow_swap_));
  }
  int NumPlayers() const override { return kNumPlayers; }
  double MinUtility() const override { return -1; }
  double UtilitySum() const override { return 0; }
  double MaxUtility() const override { return 1; }
  std::vector<int> ObservationTensorShape() const override {
    return {kCellStates, Diameter(), Diameter()};
  }
  int MaxGameLength() const override {
    // The true number of playable cells on the board.
    // No stones are removed, and it is possible to draw by filling the board.
    return Diameter() * Diameter() - board_size_ * (board_size_ - 1) +
           allow_swap_;
  }

 private:
  int Diameter() const { return board_size_ * 2 - 1; }
  const int board_size_;
  const bool ansi_color_output_ = false;
  const bool allow_swap_ = false;
};

}  // namespace havannah
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_HAVANNAH_H_
