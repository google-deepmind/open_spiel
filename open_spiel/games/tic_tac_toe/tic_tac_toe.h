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

#ifndef OPEN_SPIEL_GAMES_TIC_TAC_TOE_H_
#define OPEN_SPIEL_GAMES_TIC_TAC_TOE_H_

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "open_spiel/spiel.h"

// Simple game of Noughts and Crosses:
// https://en.wikipedia.org/wiki/Tic-tac-toe
//
// Parameters:
//       "columns"    int     number of columns on the board   (default = 3)
//       "rows"       int     number of rows on the board      (default = 3)

namespace open_spiel {

namespace ultimate_tic_tac_toe {
  class UltimateTTTState;
}

namespace tic_tac_toe {

// Constants.
inline constexpr int kNumPlayers = 2;
inline constexpr int kDefaultRows = 3;
inline constexpr int kDefaultCols = 3;
inline constexpr int kCellStates = 1 + kNumPlayers;  // empty, 'x', and 'o'.

// https://math.stackexchange.com/questions/485752/tictactoe-state-space-choose-calculation/485852
inline constexpr int kNumberStates = 5478;

// State of a cell.
enum class CellState {
  kEmpty,
  kNought,  // O
  kCross,   // X
};

// A component is a game piece
class Component {
 public:
  Component();

  CellState state_;
};

// The game board is composed of a grid of cells
class GridBoard {
 public:
  // A tile in a board is a position that can hold components
  class Tile {
   public:
    Tile() = default;

    // The component in this tile, if any
    Component component_;
  };

  // Constructs an empty board of the given dimensions
  GridBoard(size_t num_rows, size_t num_cols);

  // Get the contents of the board at a given index
  const CellState& At(size_t index) const;
  CellState& At(size_t index);

  // Get the contents of the board at a given 2D position
  const CellState& At(size_t row, size_t col) const;
  CellState& At(size_t row, size_t col);

  // Returns the total number of rows of the board
  size_t Rows() const;

  // Returns the total number of columns of the board
  size_t Cols() const;

  // Returns the total number of cells of the board
  size_t Size() const;

  // Get a visual representation of the board as text.
  std::string ToString() const;

 private:
  // The underlying container - i.e., the actual board. Should be replaced
  // by inplace_vector when supported, since it should not be resizable
  std::vector<Tile> board_;

  // The number of rows of the board
  const size_t num_rows_;

  // The number of columns of the board
  const size_t num_cols_;
};

// State of an in-play game.
class TicTacToeState : public State {
  // Since Ultimate TTT is a TTT, make sure it can access the protected
  // members too
  friend class ultimate_tic_tac_toe::UltimateTTTState;

 public:
  TicTacToeState(std::shared_ptr<const Game> game, size_t rows, size_t cols);

  TicTacToeState(const TicTacToeState&) = default;
  TicTacToeState& operator=(const TicTacToeState&) = default;

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
  std::unique_ptr<State> Clone() const override;
  void UndoAction(Player player, Action move) override;
  std::vector<Action> LegalActions() const override;

  // Get the possible legal actions for a given board state
  std::vector<Action> LegalActions(const GridBoard &board) const;
  CellState BoardAt(int cell) const { return board_.At(cell); }
  CellState BoardAt(int row, int column) const {
    return board_.At(row, column);
  }
  Player outcome() const { return outcome_; }

  // Only used by Ultimate Tic-Tac-Toe.
  void SetCurrentPlayer(Player player) { current_player_ = player; }

 protected:
  GridBoard board_;
  void DoApplyAction(Action move) override;

 private:
  bool HasLine(Player player) const;  // Does this player have a line?
  bool IsFull() const;                // Is the board full?
  Player current_player_ = 0;         // Player zero goes first
  Player outcome_ = kInvalidPlayer;
  int num_moves_ = 0;
};

// Game object.
class TicTacToeGame : public Game {
 public:
  explicit TicTacToeGame(const GameParameters& params);
  int NumDistinctActions() const override { return rows_ * cols_; }
  std::unique_ptr<State> NewInitialState() const override {
    return std::unique_ptr<State>(
      new TicTacToeState(shared_from_this(), rows_, cols_));
  }
  int NumPlayers() const override { return kNumPlayers; }
  double MinUtility() const override { return -1; }
  absl::optional<double> UtilitySum() const override { return 0; }
  double MaxUtility() const override { return 1; }
  std::vector<int> ObservationTensorShape() const override {
    return {kCellStates, rows_, cols_};
  }
  int MaxGameLength() const override { return rows_ * cols_; }
  std::string ActionToString(Player player, Action action_id) const override;

  // Returns the total number of rows of the board
  size_t Rows() const;

  // Returns the total number of columns of the board
  size_t Cols() const;

 private:
  // The number of rows in the grid
  int rows_ = -1;

  // The number of columns in the grid
  int cols_ = -1;
};

Component PlayerToComponent(Player player);
std::string StateToString(CellState state);

// Does this player have a line?
bool BoardHasLine(const GridBoard& board, const Player player);

inline std::ostream& operator<<(std::ostream& stream, const CellState& state) {
  return stream << StateToString(state);
}

}  // namespace tic_tac_toe
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_TIC_TAC_TOE_H_
