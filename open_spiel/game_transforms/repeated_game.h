// Copyright 2021 DeepMind Technologies Limited
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

#ifndef OPEN_SPIEL_GAME_TRANSFORMS_REPEATED_GAME_H_
#define OPEN_SPIEL_GAME_TRANSFORMS_REPEATED_GAME_H_

#include <memory>
#include <string>
#include <vector>

#include "open_spiel/simultaneous_move_game.h"
#include "open_spiel/spiel.h"

// Transform for creating a repeated game from a normal-form game.
// https://en.wikipedia.org/wiki/Repeated_game.
//
// Parameters:
//   "enable_infostate"   bool     Enable the sequence of round outcomes as the
//                                 information state tensor (default: false).
//   "stage_game"         game     The game that will be repeated.
//   "num_repititions"    int      Number of times that the game is repeated.


namespace open_spiel {

class RepeatedState : public SimMoveState {
 public:
  RepeatedState(std::shared_ptr<const Game> game,
                std::shared_ptr<const Game> stage_game, int num_repetitions);

  Player CurrentPlayer() const override {
    return IsTerminal() ? kTerminalPlayerId : kSimultaneousPlayerId;
  }
  std::string ActionToString(Player player, Action action_id) const override;
  std::string ToString() const override;
  bool IsTerminal() const override;
  std::vector<double> Rewards() const override;
  std::vector<double> Returns() const override;
  std::string ObservationString(Player player) const override;
  void InformationStateTensor(Player player,
                              absl::Span<float> values) const override;
  void ObservationTensor(Player player,
                         absl::Span<float> values) const override;
  std::unique_ptr<State> Clone() const override;
  std::vector<Action> LegalActions(Player player) const override;

 protected:
  void DoApplyActions(const std::vector<Action>& actions) override;

 private:
  void ObliviousObservationTensor(Player player,
                                  absl::Span<float> values) const;

  std::shared_ptr<const Game> stage_game_;
  // Store a reference initial state of the stage game for efficient calls
  // to state functions (e.g. LegalActions()).
  std::shared_ptr<const State> stage_game_state_;
  int num_repetitions_;
  std::vector<std::vector<Action>> actions_history_{};
  std::vector<std::vector<double>> rewards_history_{};
};

class RepeatedGame : public SimMoveGame {
 public:
  RepeatedGame(std::shared_ptr<const Game> stage_game,
               const GameParameters& params);
  std::unique_ptr<State> NewInitialState() const override;
  int MaxGameLength() const override { return num_repetitions_; }
  int NumPlayers() const override { return stage_game_->NumPlayers(); }
  int NumDistinctActions() const override {
    return stage_game_->NumDistinctActions();
  }
  double MinUtility() const override {
    return stage_game_->MinUtility() * num_repetitions_;
  }
  double MaxUtility() const override {
    return stage_game_->MaxUtility() * num_repetitions_;
  }
  std::vector<int> InformationStateTensorShape() const override;
  std::vector<int> ObservationTensorShape() const override;

  const Game* StageGame() const { return stage_game_.get(); }

 private:
  std::shared_ptr<const Game> stage_game_;
  const int num_repetitions_;
};

// Creates a repeated game based on the stage game.
std::shared_ptr<const Game> CreateRepeatedGame(const Game& stage_game,
                                               const GameParameters& params);
std::shared_ptr<const Game> CreateRepeatedGame(
    const std::string& stage_game_name, const GameParameters& params);

}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAME_TRANSFORMS_REPEATED_GAME_H_
