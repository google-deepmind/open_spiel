// Copyright 2019 DeepMind Technologies Ltd. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef OPEN_SPIEL_GAME_TRANSFORMS_WITH_OBSERVATION_HISTORY_H_
#define OPEN_SPIEL_GAME_TRANSFORMS_WITH_OBSERVATION_HISTORY_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "open_spiel/spiel.h"
#include "open_spiel/game_transforms/game_wrapper.h"

// This wrapper takes a game and adds tracking of observations for each player.
//
// The underlying game must provide ObservationString and
// PublicObservationString for the tracking to work.
//

namespace open_spiel {


class WithObservationHistoryState: public WrappedState {
 public:
  WithObservationHistoryState(std::shared_ptr<const Game> game,
                              std::unique_ptr<State> state);
  WithObservationHistoryState(const WithObservationHistoryState& other);

  // Methods that override the behaviour of the underlying state.
  const std::vector<std::string>& PublicObservationHistory() const override;
  const AOHistory& ActionObservationHistory(Player) const override;

  void ApplyAction(Action action_id) override;
  std::unique_ptr<State> Clone() const override;
  void UndoAction(Player player, Action action) override;
  std::unique_ptr<State> ResampleFromInfostate(
      int player_id, std::function<double()> rng) const override;
  std::unique_ptr<
      std::pair<std::vector<std::unique_ptr<State>>, std::vector<double>>>
  GetHistoriesConsistentWithInfostate(int player_id) const override;

 private:
  void InitializeRootState();
  // Make a rollout to collect the observation histories for given actions.
  void RolloutUpdate(const std::vector<Action>& actions);
  void UpdatePublicObservation(const State&);
  void UpdateObservation(Player, const State&);
  void UpdateAction(Player, Action);

  std::vector<std::string> public_observation_history_;
  // Action-Observation history per each player.
  std::vector<AOHistory> action_observation_history_;
};

class WithObservationHistoryGame: public WrappedGame {
 public:
  explicit WithObservationHistoryGame(std::shared_ptr<const Game> game);

  // Methods that override the behaviour of the underlying game.
  std::unique_ptr<State> NewInitialState() const override {
    return std::unique_ptr<State>(new WithObservationHistoryState(
        shared_from_this(), game_->NewInitialState()));
  }
  std::shared_ptr<const Game> Clone() const override {
    return std::shared_ptr<const Game>(
        new WithObservationHistoryGame(*this));
  }
};

// Return back a transformed clone of the game.
std::shared_ptr<const Game> ConvertToWithObservationHistory(const Game& game);

// These are equivalent to LoadGame but converts the game to turn-based if it is
// not already one. They are simple wrappers provided for the Python API.
std::shared_ptr<const Game> LoadGameWithObservationHistory(
    const std::string& name);
std::shared_ptr<const Game> LoadGameWithObservationHistory(
    const std::string& name, const GameParameters& params);

}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAME_TRANSFORMS_WITH_OBSERVATION_HISTORY_H_
