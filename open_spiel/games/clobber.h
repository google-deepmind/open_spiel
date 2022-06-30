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

#ifndef OPEN_SPIEL_GAMES_CLOBBER_H_
#define OPEN_SPIEL_GAMES_CLOBBER_H_

// Implementation of the board game Clobber.
// https://en.wikipedia.org/wiki/Clobber
//
// Some notes about this implementation:
// - The two players:
//     Clobber is a two player game. The two players in this
//     implementation are 'o' (White, 0) and 'x' (Black, 1). In the
//     default board of any size, the bottom left corner is always
//     'o' and continues in a checkerboard pattern from there. 'o'
//     moves first in the default board.
// - Custom boards:
//     A custom board can be used to initialize a state when calling
//     either the ClobberState(rows, columns, board_string) constructer
//     or ClobberGame's method NewInitialString(board_string). Where
//     'rows' and 'columns' are the number of rows and columns on the
//     board respectively, and 'board_string' is a string representing
//     the board. The format of board string is as follows:
//       - The first character is either a '0' or '1', this indicates
//         which player's turn it is (white or black respectively).
//       - The next characters are either 'o', 'x', or '.' which
//         represent white pieces, black pieces, or empty cells
//         respectively. There must be rows * columns number of these
//         characters following the first character.
//     For example, a state initialized from "1x.o.xo.x." on a game with
//     3 rows and 3 columns would have 'x' (Black, 1) play first on a
//     3x3 board with configuration:
//         x.o
//         .xo
//         .x.
// - Observation tensor:
//     This version implements a 3-plane observation tensor. Each plane
//     has equal dimensions as the board. The first plane contains 1's\
//     where the current player's pieces are, and 0's elsewhere. The
//     next plane contains 1's where their opponent's pieces are, and
//     0's elsewhere. Finally, the last plane consists of 1's where the
//     empty cells are, and 0's elsewhere.

#include <memory>
#include <string>
#include <vector>

#include "open_spiel/spiel.h"

namespace open_spiel {
namespace clobber {

inline constexpr int kNumPlayers = 2;

// State of a cell.
enum class CellState {
  kEmpty,  // Represented by ' '.
  kWhite,  // Represented by 'o'.
  kBlack,  // Represented by 'x'.
};

// State of an in-play game.
class ClobberState : public State {
 public:
  explicit ClobberState(std::shared_ptr<const Game> game, int rows,
                        int columns);
  explicit ClobberState(std::shared_ptr<const Game> game, int rows, int columns,
                        const std::string& board_string);
  Player CurrentPlayer() const override {
    return IsTerminal() ? kTerminalPlayerId : current_player_;
  }
  std::string ActionToString(Player player, Action action_id) const override;
  std::string ToString() const override;
  bool IsTerminal() const override;
  std::vector<double> Returns() const override;
  std::string InformationStateString(Player player) const override;
  std::string ObservationString(Player player) const override;
  void ObservationTensor(Player player,
                         absl::Span<float> values) const override;
  std::unique_ptr<State> Clone() const override {
    return std::unique_ptr<State>(new ClobberState(*this));
  }
  void UndoAction(Player player, Action action) override;
  bool InBounds(int row, int column) const;
  void SetBoard(int row, int column, CellState state) {
    board_[row * columns_ + column] = state;
  }
  CellState BoardAt(int row, int column) const {
    return board_[row * columns_ + column];
  }
  std::vector<Action> LegalActions() const override;

 protected:
  void DoApplyAction(Action action) override;

 private:
  // Returns the appropriate plane for the cell's state and current
  // player. If the cell's state is Empty, the plane is 2. Otherwise, the
  // plane depends on both the state and the player. This method ensures
  // that whichever player's turn it is, their pieces will be on plane 0,
  // and their opponents will be on plane 1.
  int ObservationPlane(CellState state, Player player) const;

  // This method takes advantage of the fact that in Clobber, a player
  // has a move if-and-only-if the oppposing player also has that move.
  // Therefore, at each board cell, just check if any adjacent cell has
  // the opponent's piece on it.
  bool MovesRemaining() const;

  Player current_player_ = 0;  // Player zero (White, 'o') goes first.
  Player outcome_ = kInvalidPlayer;
  int num_moves_ = 0;
  int rows_;
  int columns_;
  std::vector<CellState> board_;
};

// Game object.
class ClobberGame : public Game {
 public:
  explicit ClobberGame(const GameParameters& params);
  int NumDistinctActions() const override;
  std::unique_ptr<State> NewInitialState(
      const std::string& board_string) const override {
    return absl::make_unique<ClobberState>(shared_from_this(), rows_, columns_,
                                           board_string);
  }
  std::unique_ptr<State> NewInitialState() const override {
    return absl::make_unique<ClobberState>(shared_from_this(), rows_, columns_);
  }
  int NumPlayers() const override { return kNumPlayers; }
  double MinUtility() const override { return -1; }
  double UtilitySum() const override { return 0; }
  double MaxUtility() const override { return 1; }
  std::vector<int> ObservationTensorShape() const override {
    return {kNumPlayers + 1, rows_, columns_};
  }
  // On every turn, one piece is taken out. The longest game occurs
  // when the last player takes out the only remaining opponenent's
  // piece with their last piece. Therefore, there is still one piece on
  // the board. Hence, the maximum number of moves is # of cells - 1.
  int MaxGameLength() const override { return rows_ * columns_ - 1; }

 private:
  int rows_;
  int columns_;
};

std::ostream& operator<<(std::ostream& stream, const CellState& state);

}  // namespace clobber
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_CLOBBER_H_
