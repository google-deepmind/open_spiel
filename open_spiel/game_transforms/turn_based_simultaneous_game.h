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

#ifndef OPEN_SPIEL_GAME_TRANSFORMS_TURN_BASED_SIMULTANEOUS_GAME_H_
#define OPEN_SPIEL_GAME_TRANSFORMS_TURN_BASED_SIMULTANEOUS_GAME_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "open_spiel/spiel.h"

// This wrapper turns any n-player simultaneous move game into an equivalent
// turn-based game where simultaneous move nodes are encoded as n turns.
//
// The underlying game must provide InformationStateString and
// InformationStateTensor for the wrapped functions to work.
//
// TODO:
//   - implement UndoAction for these games.

namespace open_spiel {

class TurnBasedSimultaneousObserver;

class TurnBasedSimultaneousState : public State {
 public:
  TurnBasedSimultaneousState(std::shared_ptr<const Game> game,
                             std::unique_ptr<State> state);
  TurnBasedSimultaneousState(const TurnBasedSimultaneousState& other);

  Player CurrentPlayer() const override;
  std::string ActionToString(Player player, Action action_id) const override;
  std::string ToString() const override;
  bool IsTerminal() const override;
  std::vector<double> Returns() const override;
  std::string InformationStateString(Player player) const override;
  std::string ObservationString(Player player) const override;
  void InformationStateTensor(Player player,
                              absl::Span<float> values) const override;
  void ObservationTensor(Player player,
                         absl::Span<float> values) const override;
  std::unique_ptr<State> Clone() const override;
  std::vector<std::pair<Action, double>> ChanceOutcomes() const override;

  // Access to the wrapped state, used for debugging and in the tests.
  const State* SimultaneousGameState() const { return state_.get(); }
  std::vector<Action> LegalActions() const override;

 protected:
  void DoApplyAction(Action action_id) override;

 private:
  void DetermineWhoseTurn();
  void RolloutModeIncrementCurrentPlayer();
  friend TurnBasedSimultaneousObserver;
  std::unique_ptr<State> state_;

  // A vector of actions that is used primarily to store the intermediate
  // actions taken by the players when extending the simultaneous move nodes
  // to be turn-based.
  std::vector<Action> action_vector_;

  // The current player (which will never be kSimultaneousPlayerId).
  Player current_player_;

  // Are we currently rolling out a simultaneous move node?
  bool rollout_mode_;
};

class TurnBasedSimultaneousGame : public Game {
 public:
  explicit TurnBasedSimultaneousGame(std::shared_ptr<const Game> game);

  std::unique_ptr<State> NewInitialState() const override {
    return std::unique_ptr<State>(new TurnBasedSimultaneousState(
        shared_from_this(), game_->NewInitialState()));
  }

  int NumDistinctActions() const override {
    return game_->NumDistinctActions();
  }
  int MaxChanceOutcomes() const override { return game_->MaxChanceOutcomes(); }
  int NumPlayers() const override { return game_->NumPlayers(); }
  double MinUtility() const override { return game_->MinUtility(); }
  double MaxUtility() const override { return game_->MaxUtility(); }
  double UtilitySum() const override { return game_->UtilitySum(); }
  int MaxGameLength() const override {
    return game_->MaxGameLength() * NumPlayers();
  }
  int MaxChanceNodesInHistory() const override {
    return game_->MaxChanceNodesInHistory();
  }
  std::vector<int> InformationStateTensorShape() const override {
    return {NumDistinctActions() + NumPlayers()
            + game_->InformationStateTensorSize()};
  }
  std::vector<int> ObservationTensorShape() const override {
    return {NumDistinctActions() + NumPlayers()
            + game_->ObservationTensorSize()};
  }
  std::shared_ptr<Observer> MakeObserver(
      absl::optional<IIGObservationType> iig_obs_type,
      const GameParameters& params) const override;
 private:
  friend TurnBasedSimultaneousState;
  std::shared_ptr<const Game> game_;
  // Used to implement the old observation API.
  std::shared_ptr<TurnBasedSimultaneousObserver> default_observer_;
  std::shared_ptr<TurnBasedSimultaneousObserver> info_state_observer_;
};

// Return back a transformed clone of the game.
std::shared_ptr<const Game> ConvertToTurnBased(const Game& game);

// These are equivalent to LoadGame but converts the game to turn-based if it is
// not already one. They are simple wrappers provided for the Python API.
std::shared_ptr<const Game> LoadGameAsTurnBased(const std::string& name);
std::shared_ptr<const Game> LoadGameAsTurnBased(const std::string& name,
                                                const GameParameters& params);

}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAME_TRANSFORMS_TURN_BASED_SIMULTANEOUS_GAME_H_
