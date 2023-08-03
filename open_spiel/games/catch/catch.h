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

#ifndef OPEN_SPIEL_GAMES_CATCH_H_
#define OPEN_SPIEL_GAMES_CATCH_H_

#include <memory>
#include <string>
#include <vector>

#include "open_spiel/spiel.h"

// Catch is a single player game, often used for unit testing RL algorithms.
//
// The player must move a paddle to intercept a falling ball. The initial
// column of the ball is decided by chance. Each turn, the ball moves downwards
// while remaining in the initial column.
//
// Please note: In each turn, all actions (left, stay, right) are legal. This
// is different to the Python implementation of the game.
//
// References:
// a) Recurrent models of visual attention, 2014, Minh et al.
//    (Advances in Neural Information Processing Systems 27, pages 2204–2212.)
// b) Behaviour Suite for Reinforcement Learning, 2019, Osband et al.
//    (https://arxiv.org/abs/1908.03568)
//
// Parameters:
//  "rows"       int    rows of the board        (default = 10)
//  "columns"    int    columns of the board     (default = 5)

namespace open_spiel {
namespace catch_ {

// Constants.
inline constexpr int kNumPlayers = 1;
inline constexpr int kNumActions = 3;
inline constexpr int kDefaultRows = 10;
inline constexpr int kDefaultColumns = 5;

// State of a cell.
enum class CellState {
  kEmpty,
  kBall,
  kPaddle,
};

class CatchGame;

// State of an in-play game.
class CatchState : public State {
 public:
  CatchState(std::shared_ptr<const Game> game);
  CatchState(const CatchState&) = default;

  Player CurrentPlayer() const override;
  std::string ActionToString(Player player, Action action_id) const override;
  std::string ToString() const override;
  bool IsTerminal() const override;
  std::vector<double> Returns() const override;
  std::string ObservationString(Player player) const override;
  void ObservationTensor(Player player,
                         absl::Span<float> values) const override;
  std::unique_ptr<State> Clone() const override;
  std::vector<Action> LegalActions() const override;
  ActionsAndProbs ChanceOutcomes() const override;
  CellState BoardAt(int row, int column) const;

 protected:
  void DoApplyAction(Action move) override;

 private:
  int num_rows_ = -1;
  int num_columns_ = -1;
  bool initialized_ = false;
  int ball_row_ = -1;
  int ball_col_ = -1;
  int paddle_col_ = -1;
};

// Game object.
class CatchGame : public Game {
 public:
  explicit CatchGame(const GameParameters& params);
  std::unique_ptr<State> NewInitialState() const override {
    return std::unique_ptr<State>(new CatchState(shared_from_this()));
  }
  std::vector<int> ObservationTensorShape() const override {
    return {num_rows_, num_columns_};
  }

  int NumDistinctActions() const override { return kNumActions; }
  int MaxChanceOutcomes() const override { return num_columns_; }
  int NumPlayers() const override { return kNumPlayers; }
  double MaxUtility() const override { return 1; }
  double MinUtility() const override { return -1; }
  int MaxGameLength() const override { return num_rows_; }
  // There is only initial chance.
  int MaxChanceNodesInHistory() const override { return 1; }
  int NumRows() const { return num_rows_; }
  int NumColumns() const { return num_columns_; }

 private:
  const int num_rows_;
  const int num_columns_;
};

}  // namespace catch_
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_CATCH_H_
