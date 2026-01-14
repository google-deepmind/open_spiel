// Copyright 2025 DeepMind Technologies Limited
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

#ifndef OPEN_SPIEL_GAMES_LATENT_TTT_H_
#define OPEN_SPIEL_GAMES_LATENT_TTT_H_

#include <array>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "open_spiel/abseil-cpp/absl/types/span.h"
#include "open_spiel/game_parameters.h"
#include "open_spiel/games/tic_tac_toe/tic_tac_toe.h"
#include "open_spiel/spiel.h"

// Latent Tic-Tac-Toe.
// Implements the game from "Monte Carlo Sampling for Regret Minimization in
// Extensive Games" (http://mlanctot.info/files/papers/nips09mccfr.pdf).
// The rules are described in the paper as follows: "Latent Tic-Tac-Toe
// is a twist on the classic game where moves are not disclosed until after
// the opponent’s next move, and lost if invalid at the time they are revealed."

namespace open_spiel {
namespace latent_ttt {

class LatentTTTState : public State {
 public:
  explicit LatentTTTState(std::shared_ptr<const Game> game);

  LatentTTTState(const LatentTTTState&) = default;
  LatentTTTState& operator=(const LatentTTTState&) = default;

  Player CurrentPlayer() const override { return state_.CurrentPlayer(); }
  std::string ActionToString(Player player, Action action_id) const override {
    return state_.ActionToString(player, action_id);
  }
  std::string ToString() const override { return state_.ToString(); }
  bool IsTerminal() const override { return state_.IsTerminal(); }
  std::vector<double> Returns() const override { return state_.Returns(); }
  std::string InformationStateString(Player player) const override;
  std::string ObservationString(Player player) const override;
  void ObservationTensor(Player player,
                         absl::Span<float> values) const override;
  std::unique_ptr<State> Clone() const override;
  void UndoAction(Player player, Action move) override;
  std::vector<Action> LegalActions() const override;

 private:
  void DoApplyAction(Action move) override;
  std::string ViewToString(Player player) const;

  tic_tac_toe::TicTacToeState state_;
  std::vector<std::pair<int, Action>> action_sequence_;
  std::array<tic_tac_toe::CellState, tic_tac_toe::kNumCells> x_view_;
  std::array<tic_tac_toe::CellState, tic_tac_toe::kNumCells> o_view_;
};

// Game object.
class LatentTTTGame : public Game {
 public:
  explicit LatentTTTGame(const GameParameters& params);
  int NumDistinctActions() const override { return tic_tac_toe::kNumCells; }
  std::unique_ptr<State> NewInitialState() const override {
    return std::unique_ptr<State>(new LatentTTTState(shared_from_this()));
  }
  int NumPlayers() const override { return tic_tac_toe::kNumPlayers; }
  double MinUtility() const override { return -1; }
  absl::optional<double> UtilitySum() const override { return 0; }
  double MaxUtility() const override { return 1; }
  std::vector<int> ObservationTensorShape() const override {
    return {tic_tac_toe::kCellStates, tic_tac_toe::kNumRows,
            tic_tac_toe::kNumCols};
  }
  // Each move can be done twice, once by the player and once by the opponent.
  int MaxGameLength() const override { return tic_tac_toe::kNumCells * 2; }
  std::string ActionToString(Player player, Action action_id) const override;
};

}  // namespace latent_ttt
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_LATENT_TTT_H_
