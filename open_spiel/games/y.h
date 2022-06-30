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

#ifndef OPEN_SPIEL_GAMES_Y_H_
#define OPEN_SPIEL_GAMES_Y_H_

#include <array>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "open_spiel/spiel.h"

// https://en.wikipedia.org/wiki/Y_(game)
// Does not implement pie rule to balance the game
//
// Parameters:
//   "board_size"        int     size of the board   (default = 19)
//   "ansi_color_output" bool    Whether to color the output for a terminal.

namespace open_spiel {
namespace y_game {

inline constexpr int kNumPlayers = 2;
inline constexpr int kDefaultBoardSize = 19;
inline constexpr int kMaxNeighbors =
    6;  // Maximum number of neighbors for a cell
inline constexpr int kCellStates = 1 + kNumPlayers;

enum YPlayer : uint8_t {
  kPlayer1,
  kPlayer2,
  kPlayerNone,
  kPlayerInvalid,
};

enum MoveSpecial {
  kMoveNone = -1,
  kMoveUnknown = -2,
  kMoveOffset = -3,
};

int CalcXY(int x, int y, int board_size) {
  if (x >= 0 && y >= 0 && x < board_size && y < board_size &&
      (x + y < board_size)) {
    return x + y * board_size;
  } else {
    return kMoveUnknown;
  }
}

struct Move {
  int8_t x, y;  // The x,y coordinates
  int16_t xy;   // precomputed x + y * board_size as an index into the array.

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
  // a MoveSpecial, in the cut-off corner, or otherwise off the board.
  bool OnBoard() const { return xy >= 0; }

  // Flags for which edge this move is part of.
  int Edge(int board_size) const;
};

// List of neighbors of a cell: [cell][direction]
typedef std::vector<std::array<Move, kMaxNeighbors>> NeighborList;

// State of an in-play game.
class YState : public State {
  // Represents a single cell on the board, as well as the structures needed for
  // groups of cells. Groups of cells are defined by a union-find structure
  // embedded in the array of cells. Following the `parent` indices will lead to
  // the group leader which has the up to date size and edge
  // connectivity of that group. Size and edge are not valid for any
  // cell that is not a group leader.
  struct Cell {
    // Who controls this cell.
    YPlayer player;

    // A parent index to allow finding the group leader. It is the leader of the
    // group if it points to itself. Allows path compression to shorten the path
    // from a direct parent to the leader.
    uint16_t parent;

    // These three are only defined for the group leader's cell.
    uint16_t size;  // Size of this group of cells.
    uint8_t edge;   // A bitset of which edges this group is connected to.

    Cell() {}
    Cell(YPlayer player_, int parent_, int edge_)
        : player(player_), parent(parent_), size(1), edge(edge_) {}
  };

 public:
  YState(std::shared_ptr<const Game> game, int board_size,
         bool ansi_color_output = false);

  YState(const YState&) = default;

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

  // Join the groups of two positions, propagating group size, and edge
  // connections. Returns true if they were already the same group.
  bool JoinGroups(int cell_a, int cell_b);

  // Turn an action id into a `Move` with an x,y.
  Move ActionToMove(Action action_id) const;

 private:
  std::vector<Cell> board_;
  YPlayer current_player_ = kPlayer1;
  YPlayer outcome_ = kPlayerNone;
  const int board_size_;
  int moves_made_ = 0;
  Move last_move_ = kMoveNone;
  const NeighborList& neighbors;
  const bool ansi_color_output_;
};

// Game object.
class YGame : public Game {
 public:
  explicit YGame(const GameParameters& params);

  int NumDistinctActions() const override {
    // Really size*(size+1)/2, but that's harder to represent, so the extra
    // actions in the corner are never legal.
    return board_size_ * board_size_;
  }
  std::unique_ptr<State> NewInitialState() const override {
    return std::unique_ptr<State>(
        new YState(shared_from_this(), board_size_, ansi_color_output_));
  }
  int NumPlayers() const override { return kNumPlayers; }
  double MinUtility() const override { return -1; }
  double UtilitySum() const override { return 0; }
  double MaxUtility() const override { return 1; }
  std::vector<int> ObservationTensorShape() const override {
    return {kCellStates, board_size_, board_size_};
  }
  int MaxGameLength() const override {
    // The true number of playable cells on the board.
    // No stones are removed, and someone will win by filling the board.
    // Increase this by one if swap is ever implemented.
    return board_size_ * (board_size_ + 1) / 2;
  }

 private:
  const int board_size_;
  const bool ansi_color_output_ = false;
};

}  // namespace y_game
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_Y_H_
