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

#ifndef THIRD_PARTY_OPEN_SPIEL_GAME_TRANSFORMS_GAME_WRAPPER_H_
#define THIRD_PARTY_OPEN_SPIEL_GAME_TRANSFORMS_GAME_WRAPPER_H_

#include "open_spiel/spiel.h"

// Wraps a game, forwarding everything to the original implementation.
// Transforms can inherit from this, overriding only what they need.

namespace open_spiel {

class WrappedState : public State {
 public:
  WrappedState(std::unique_ptr<State> state)
      : State(state->NumDistinctActions(), state->NumPlayers()),
        state_(std::move(state)) {}
  WrappedState(const WrappedState& other)
      : State(other), state_(other.state_->Clone()) {}

  int CurrentPlayer() const override { return state_->CurrentPlayer(); }

  virtual std::vector<Action> LegalActions(int player) const {
    return state_->LegalActions(player);
  }

  std::vector<Action> LegalActions() const override {
    return state_->LegalActions();
  }

  std::string ActionToString(int player, Action action_id) const override {
    return state_->ActionToString(player, action_id);
  }

  std::string ToString() const override { return state_->ToString(); }

  bool IsTerminal() const override { return state_->IsTerminal(); }

  std::vector<double> Rewards() const override { return state_->Rewards(); }

  std::vector<double> Returns() const override { return state_->Returns(); }

  std::string InformationState(int player) const override {
    return state_->InformationState(player);
  }

  void InformationStateAsNormalizedVector(
      int player, std::vector<double>* values) const override {
    state_->InformationStateAsNormalizedVector(player, values);
  }

  virtual std::string Observation(int player) const {
    return state_->Observation(player);
  }

  virtual void ObservationAsNormalizedVector(
      int player, std::vector<double>* values) const {
    state_->ObservationAsNormalizedVector(player, values);
  }

  virtual std::unique_ptr<State> Clone() const = 0;

  virtual void UndoAction(int player, Action action) {
    state_->UndoAction(player, action);
    history_.pop_back();
  }

  ActionsAndProbs ChanceOutcomes() const override {
    return state_->ChanceOutcomes();
  }

  std::vector<Action> LegalChanceOutcomes() const override {
    return state_->LegalChanceOutcomes();
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
  WrappedGame(std::unique_ptr<Game> game, GameType game_type,
              GameParameters game_parameters)
      : Game(game_type, game_parameters), game_(std::move(game)) {}
  WrappedGame(const WrappedGame& other)
      : Game(other), game_(other.game_->Clone()) {}

  int NumDistinctActions() const override {
    return game_->NumDistinctActions();
  }

  std::unique_ptr<State> NewInitialState() const override = 0;
  std::unique_ptr<Game> Clone() const override = 0;

  int MaxChanceOutcomes() const override { return game_->MaxChanceOutcomes(); }
  int NumPlayers() const override { return game_->NumPlayers(); }
  double MinUtility() const override { return game_->MaxUtility(); }
  double MaxUtility() const override { return game_->MinUtility(); }
  double UtilitySum() const override { return game_->UtilitySum(); }

  std::vector<int> InformationStateNormalizedVectorShape() const override {
    return game_->InformationStateNormalizedVectorShape();
  }

  std::vector<int> ObservationNormalizedVectorShape() const override {
    return game_->ObservationNormalizedVectorShape();
  }

  int MaxGameLength() const override { return game_->MaxGameLength(); }

 protected:
  std::unique_ptr<Game> game_;
};

}  // namespace open_spiel

#endif  // THIRD_PARTY_OPEN_SPIEL_GAME_TRANSFORMS_GAME_WRAPPER_H_
