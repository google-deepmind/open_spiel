// Copyright 2023 DeepMind Technologies Limited
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

#ifndef OPEN_SPIEL_GAMES_ABALONE_ABALONE_H_
#define OPEN_SPIEL_GAMES_ABALONE_ABALONE_H_

#include <array>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "open_spiel/spiel.h"
#include "open_spiel/algorithms/mcts.h"
#include "open_spiel/games/abalone/abalone_core.h"

// Game of Abalone where players push marbles off the board to eliminate
// the opponent's marbles.
// Parameters:
//  "marbles_to_win"   int    marble count to remove from board (default = 6)
//  "board"            string initial board setup (default = "classic")
//  "invert"           bool   invert player positions (default = false)
//  "marble_advantage" bool   if the game reaches the move limit without a
//                            winner, the winner is the player who has
//                            removed the most marbles (default = false)
//  "draw_penalty"     double penalty applied on a draw (game reaches the
//                            move limit with no winner): the player with
//                            fewer marbles receives -draw_penalty and the
//                            other +draw_penalty. No penalty when marbles
//                            are equal. Zero-sum (default = 0.0)

namespace open_spiel {

namespace abalone {

// State of an in-play game.
class AbaloneState : public State {
 public:
  explicit AbaloneState(std::shared_ptr<const Game> game);

  AbaloneState(const AbaloneState &) = default;
  AbaloneState &operator=(const AbaloneState &) = default;

  Player CurrentPlayer() const override {
    return IsTerminal() ? kTerminalPlayerId : core_state_.ToPlay();
  }
  std::string ActionToString(Player player, Action action_id) const override;
  Action StringToAction(Player player,
                        const std::string& action_str) const override;

  std::string ToString() const override;
  bool IsTerminal() const override;
  std::vector<double> Rewards() const override;
  std::vector<double> Returns() const override;
  std::string InformationStateString(Player player) const override;
  std::string ObservationString(Player player) const override;
  void ObservationTensor(Player player,
                          absl::Span<float> values) const override;
  std::unique_ptr<State> Clone() const override;
  std::vector<Action> LegalActions() const override;
  virtual void UndoAction(Player player, Action action);

  abalone_core::core_state core_state_;

 protected:
  void DoApplyAction(Action action) override;
  void ResetBoard();

  // Per-step reward, kept consistent with Returns() (Rewards() must sum to
  // Returns() at every step): Rewards() = Returns() - prev_returns_, updated
  // once per DoApplyAction.
  std::vector<double> rewards_ = {0.0, 0.0};
  std::vector<double> prev_returns_ = {0.0, 0.0};

  friend struct Move;
};

// Game object.
class AbaloneGame : public Game {
 public:
  explicit AbaloneGame(const GameParameters &params);
  int NumDistinctActions() const override {
    return abalone_core::kNumCells * abalone_core::kNumActionsPerCell;
  }
  std::unique_ptr<State> NewInitialState() const override;
  int NumPlayers() const override { return abalone_core::kNumPlayers; }
  double MinUtility() const override { return -1; }
  absl::optional<double> UtilitySum() const override { return 0; }
  double MaxUtility() const override { return 1; }
  std::vector<int> ObservationTensorShape() const override {
    // status: empty, player1, player2
    return {abalone_core::kNumPlayers + 1, abalone_core::kNumRows,
            abalone_core::kNumCols};
  }
  int MaxGameLength() const override { return abalone_core::kHistoryMax; }
  std::string ActionToString(Player player, Action action_id) const override;

 protected:
  friend class AbaloneState;

  // config
  int m_marbles_to_win;
  bool m_marble_advantage;
  double m_marble_reward;
  double m_draw_penalty;
  std::string m_init_board;
  bool m_init_invert;  // invert the positions of player 1 and player 2
};

inline std::ostream& operator<<(std::ostream& stream,
                                const abalone_core::CellState& state) {
  return stream << abalone_core::StateToString(state);
}

class AbaloneEvaluator : public open_spiel::algorithms::Evaluator {
 public:
  explicit AbaloneEvaluator() {}

  // Runs random games, returning the average returns.
  std::vector<double> Evaluate(const State& state) override;

  // Returns equal probability for each action.
  ActionsAndProbs Prior(const State& state) override;
};

// other functions
std::pair<open_spiel::Action, float> AllAbaloneMoves_ABSpiel(
    const std::unique_ptr<State>& _state, int _depth,
    std::vector<std::pair<open_spiel::Action, float>>* all_moves = nullptr);

std::pair<Action, std::vector<std::pair<Action, float>>> AbaloneAB(
    const State& state, int depth);

}  // namespace abalone
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_ABALONE_ABALONE_H_
