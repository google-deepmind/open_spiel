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

#ifndef OPEN_SPIEL_GAME_TRANSFORMS_GAME_WRAPPER_H_
#define OPEN_SPIEL_GAME_TRANSFORMS_GAME_WRAPPER_H_

#include "open_spiel/spiel.h"

// Wraps a game, forwarding everything to the original implementation.
// Transforms can inherit from this, overriding only what they need.

namespace open_spiel {

class WrappedState : public State {
 public:
  WrappedState(std::shared_ptr<const Game> game, std::unique_ptr<State> state)
      : State(game), state_(std::move(state)) {}
  WrappedState(const WrappedState& other)
      : State(other), state_(other.state_->Clone()) {}

  Player CurrentPlayer() const override { return state_->CurrentPlayer(); }

  std::vector<Action> LegalActions(Player player) const override {
    return state_->LegalActions(player);
  }

  std::vector<Action> LegalActions() const override {
    return state_->LegalActions();
  }

  std::string ActionToString(Player player, Action action_id) const override {
    return state_->ActionToString(player, action_id);
  }

  std::string ToString() const override { return state_->ToString(); }

  bool IsTerminal() const override { return state_->IsTerminal(); }

  std::vector<double> Rewards() const override { return state_->Rewards(); }

  std::vector<double> Returns() const override { return state_->Returns(); }

  std::string InformationStateString(Player player) const override {
    return state_->InformationStateString(player);
  }

  void InformationStateTensor(Player player,
                              absl::Span<float> values) const override {
    state_->InformationStateTensor(player, values);
  }

  std::string ObservationString(Player player) const override {
    return state_->ObservationString(player);
  }

  void ObservationTensor(Player player,
                         absl::Span<float> values) const override {
    state_->ObservationTensor(player, values);
  }

  std::unique_ptr<State> Clone() const override = 0;

  void UndoAction(Player player, Action action) override {
    state_->UndoAction(player, action);
    history_.pop_back();
  }

  ActionsAndProbs ChanceOutcomes() const override {
    return state_->ChanceOutcomes();
  }

  std::vector<Action> LegalChanceOutcomes() const override {
    return state_->LegalChanceOutcomes();
  }

  const State& GetWrappedState() const { return *state_; }

  std::vector<Action> ActionsConsistentWithInformationFrom(
      Action action) const override {
    return state_->ActionsConsistentWithInformationFrom(action);
  }

 protected:
  void DoApplyAction(Action action_id) override {
    state_->ApplyAction(action_id);
  }

  void DoApplyActions(const std::vector<Action>& actions) override {
    state_->ApplyActions(actions);
  }

  std::unique_ptr<State> state_;
};

class WrappedGame : public Game {
 public:
  WrappedGame(std::shared_ptr<const Game> game, GameType game_type,
              GameParameters game_parameters)
      : Game(game_type, game_parameters), game_(game) {}

  int NumDistinctActions() const override {
    return game_->NumDistinctActions();
  }

  std::unique_ptr<State> NewInitialState() const override = 0;

  int MaxChanceOutcomes() const override { return game_->MaxChanceOutcomes(); }
  int NumPlayers() const override { return game_->NumPlayers(); }
  double MinUtility() const override { return game_->MinUtility(); }
  double MaxUtility() const override { return game_->MaxUtility(); }
  double UtilitySum() const override { return game_->UtilitySum(); }

  std::vector<int> InformationStateTensorShape() const override {
    return game_->InformationStateTensorShape();
  }

  std::vector<int> ObservationTensorShape() const override {
    return game_->ObservationTensorShape();
  }

  int MaxGameLength() const override { return game_->MaxGameLength(); }
  int MaxChanceNodesInHistory() const override {
    return game_->MaxChanceNodesInHistory();
  }

 protected:
  std::shared_ptr<const Game> game_;
};

}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAME_TRANSFORMS_GAME_WRAPPER_H_
