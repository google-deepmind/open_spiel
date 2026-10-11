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

#ifndef OPEN_SPIEL_GAMES_SNAKE_H_
#define OPEN_SPIEL_GAMES_SNAKE_H_

#include <deque>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "open_spiel/simultaneous_move_game.h"
#include "open_spiel/spiel.h"

// A multi-player Snake game played on a rectangular grid.
//
// Each player controls a snake. At every step, all alive players simultaneously
// choose a direction (north, east, south, or west). All snakes then move one
// cell in their chosen direction. A snake dies if its new head:
//   - leaves the board,
//   - collides with another snake's new head (both snakes die), or
//   - collides with any (post-move) body cell of a snake (including its own).
// A snake that moves onto the (single) fruit cell grows by one and increments
// its score. After any snake eats the fruit, a chance node places a new fruit
// uniformly at random on an empty cell.
//
// The episode ends when fewer than two snakes are alive or after a configurable
// horizon. Each player's return is the number of fruits they ate.
//
// Parameters:
//   "rows"     int    rows of the board                 (default = 10)
//   "columns"  int    columns of the board              (default = 10)
//   "players"  int    number of players (must be 2 or 4) (default = 2)
//   "horizon"  int    max number of steps               (default = 100)

namespace open_spiel {
namespace snake {

inline constexpr int kDefaultRows = 10;
inline constexpr int kDefaultColumns = 10;
inline constexpr int kDefaultPlayers = 2;
inline constexpr int kDefaultHorizon = 100;
inline constexpr int kMaxPlayers = 4;
inline constexpr int kMinPlayers = 2;

// Movement actions.
inline constexpr int kNumMovementActions = 4;
inline constexpr Action kNorth = 0;
inline constexpr Action kEast = 1;
inline constexpr Action kSouth = 2;
inline constexpr Action kWest = 3;

// A cell on the board, indexed by (row, column).
struct Cell {
  int row;
  int col;
  bool operator==(const Cell& o) const { return row == o.row && col == o.col; }
  bool operator!=(const Cell& o) const { return !(*this == o); }
};

struct Snake {
  // Body cells; front() is the head, back() is the tail.
  std::deque<Cell> body;
  bool alive = true;
  int score = 0;
};

class SnakeGame;

class SnakeState : public SimMoveState {
 public:
  SnakeState(std::shared_ptr<const Game> game, int rows, int cols,
             int num_players, int horizon);
  SnakeState(const SnakeState&) = default;

  Player CurrentPlayer() const override;
  std::string ActionToString(Player player, Action action_id) const override;
  std::string ToString() const override;
  bool IsTerminal() const override;
  std::vector<double> Returns() const override;
  std::string ObservationString(Player player) const override;
  void ObservationTensor(Player player,
                         absl::Span<float> values) const override;
  std::unique_ptr<State> Clone() const override;
  std::vector<Action> LegalActions(Player player) const override;
  ActionsAndProbs ChanceOutcomes() const override;

  int NumRows() const { return num_rows_; }
  int NumCols() const { return num_cols_; }
  const Snake& GetSnake(Player p) const { return snakes_[p]; }
  bool HasFruit() const { return has_fruit_; }
  Cell Fruit() const { return fruit_; }

 protected:
  void DoApplyAction(Action action_id) override;
  void DoApplyActions(const std::vector<Action>& actions) override;

 private:
  void PlaceInitialSnakes();
  void PlaceFruit(int empty_cell_index);
  // Returns the list of empty cells (no snake body, no fruit).
  std::vector<Cell> EmptyCells() const;
  bool InBounds(const Cell& c) const;
  int NumAlive() const;

  const int num_rows_;
  const int num_cols_;
  const int num_players_total_;
  const int horizon_;

  std::vector<Snake> snakes_;
  bool has_fruit_ = false;
  Cell fruit_ = {-1, -1};
  // True when the next decision is a chance node placing a new fruit.
  bool chance_pending_ = true;
  int total_moves_ = 0;
};

class SnakeGame : public SimMoveGame {
 public:
  explicit SnakeGame(const GameParameters& params);
  std::unique_ptr<State> NewInitialState() const override {
    return std::unique_ptr<State>(new SnakeState(
        shared_from_this(), num_rows_, num_cols_, num_players_, horizon_));
  }
  int NumDistinctActions() const override { return kNumMovementActions; }
  // Maximum number of empty cells where the fruit could be placed: any cell
  // that is not occupied by a snake. Initially the snakes occupy num_players_
  // cells, so the maximum is rows * cols - num_players_. We use rows * cols as
  // a safe upper bound to keep the bookkeeping simple.
  int MaxChanceOutcomes() const override { return num_rows_ * num_cols_; }
  int NumPlayers() const override { return num_players_; }
  double MinUtility() const override { return 0; }
  double MaxUtility() const override {
    return static_cast<double>(num_rows_ * num_cols_ - num_players_);
  }
  std::vector<int> ObservationTensorShape() const override {
    // Per player: head plane + body plane. Plus a fruit plane.
    return {2 * num_players_ + 1, num_rows_, num_cols_};
  }
  int MaxGameLength() const override { return horizon_; }
  // One chance node before the first move, plus potentially one chance node
  // per step (if a fruit is eaten).
  int MaxChanceNodesInHistory() const override { return horizon_ + 1; }

  int NumRows() const { return num_rows_; }
  int NumCols() const { return num_cols_; }
  int Horizon() const { return horizon_; }

 private:
  const int num_rows_;
  const int num_cols_;
  const int num_players_;
  const int horizon_;
};

}  // namespace snake
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_SNAKE_H_
