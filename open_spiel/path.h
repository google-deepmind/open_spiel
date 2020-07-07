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

#ifndef OPEN_SPIEL_PATH_H_
#define OPEN_SPIEL_PATH_H_

#include <string>
#include <utility>
#include <vector>

#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/fog/observation_history.h"

namespace open_spiel {

// This class represents a path in the world tree: a valid sequence of states
// from the root to the current state.
// This is somewhat similar to BatchedTrajectory, but it stores the underlying
// list of States. This allows to call all the normal State methods on them,
// except ones that modify them (you need to clone these states first).
//
// Additionally, Path derives from State and forwards all the method calls to
// the last State on the path, except for the ones that mutate the State.
// Methods that perform mutation, like ApplyAction and UndoAction, simply
// update the path -- push or pop the state.
//
// This class is useful if you need to store the states when you are doing
// a rollout in the game tree, or if you want to make queries for observation
// histories in the game.
class Path: public State {
 private:
  std::vector<std::unique_ptr<State>> states_;

 public:
  explicit Path(std::shared_ptr<const Game> game)
      : State(game) {
    states_.push_back(game->NewInitialState());
  }

  explicit Path(std::unique_ptr<State> current_state);

  Path(const Path& other) : State(other.GetGame()) {
    states_.reserve(other.states_.size());
    for (const auto& state : other.states_) {
      states_.push_back(state->Clone());
    }
  }

  ~Path() = default;

  // ---------------------------------------------------------------------------
  // The only path mutations
  // ---------------------------------------------------------------------------

  void ApplyAction(Action action_id) override {
    states_.push_back(states_.back()->Child(action_id));
  }

  void UndoAction(Player player, Action action) override {
    // We have to make sure that we have at least
    // one state remaining on the path.
    SPIEL_CHECK_GE(states_.size(), 2);
    SPIEL_CHECK_EQ(states_.back()->CurrentPlayer(), player);
    SPIEL_CHECK_EQ(states_.back()->History().back(), action);
    states_.pop_back();
  }

  // ---------------------------------------------------------------------------
  // All remaining methods are const.
  // ---------------------------------------------------------------------------

  // Provide only const methods for accessing states on the path
  // to prevent mutation.
  const State& operator[](int idx) const { return *states_[idx]; }
  const State& back() const { return *states_.back(); }
  const State& front() const { return *states_[0]; }

  std::unique_ptr<Path> ClonePath() const {
    return std::unique_ptr<Path>(new Path(*this));
  }

  // Return a history of public observations.
  // This method can be called only if the game provides factored observations
  // strings.
  //
  // Public observation history identifies the current public state, and is
  // useful for integration with public state API -- you can construct a
  // PublicState by using the public observation history.
  POHistory PublicObservationHistory() const;

  // Return a history of actions and observations for a given player.
  // This method can be called only if the game provides observations strings.
  //
  // Action-Observation histories partition the game tree in the same way
  // as information states, but they contain more structured information.
  // Algorithms can use this structured information for targeted traversal
  // of imperfect information games.
  const AOHistory& ActionObservationHistory(Player) const;

  // Return Action-Observation history for the current player.
  const AOHistory& ActionObservationHistory() const {
    return ActionObservationHistory(CurrentPlayer());
  }

  // ---------------------------------------------------------------------------
  // Wrap all the remaining State methods.
  // ---------------------------------------------------------------------------

  Player CurrentPlayer() const override {
    return states_.back()->CurrentPlayer();
  }

  std::vector<Action> LegalActions(Player player) const override {
    return states_.back()->LegalActions(player);
  }

  std::vector<Action> LegalActions() const override {
    return states_.back()->LegalActions();
  }

  std::string ActionToString(Player player, Action action_id) const override {
    return states_.back()->ActionToString(player, action_id);
  }

  std::string ToString() const override {
    return states_.back()->ToString();
  }

  bool IsTerminal() const override {
    return states_.back()->IsTerminal();
  }

  std::vector<double> Rewards() const override {
    return states_.back()->Rewards();
  }

  std::vector<double> Returns() const override {
    return states_.back()->Returns();
  }

  std::string InformationStateString(Player player) const override {
    return states_.back()->InformationStateString(player);
  }

  void InformationStateTensor(
      Player player, std::vector<double>* values) const override {
    states_.back()->InformationStateTensor(player, values);
  }

  std::string ObservationString(Player player) const override {
    return states_.back()->ObservationString(player);
  }

  std::string PublicObservationString() const override {
    return states_.back()->PublicObservationString();
  }

  std::string PrivateObservationString(Player player) const override {
    return states_.back()->PrivateObservationString(player);
  }

  void ObservationTensor(Player player,
                         std::vector<double>* values) const override {
    states_.back()->ObservationTensor(player, values);
  }

  ActionsAndProbs ChanceOutcomes() const override {
    return states_.back()->ChanceOutcomes();
  }

  std::vector<Action> LegalChanceOutcomes() const override {
    return states_.back()->LegalChanceOutcomes();
  }

  std::unique_ptr<State> Clone() const {
    return states_.back()->Clone();
  }

  std::vector<Action> History() const override {
    return states_.back()->History();
  }

  std::vector<PlayerAction> FullHistory() const override {
    return states_.back()->FullHistory();
  }

  std::unique_ptr<State> ResampleFromInfostate(
      int player_id, std::function<double()> rng) const override {
    return states_.back()->ResampleFromInfostate(player_id, rng);
  }

  std::unique_ptr<
      std::pair<std::vector<std::unique_ptr<State>>,
                std::vector<double>>>
  GetHistoriesConsistentWithInfostate(int player_id) const override {
    return states_.back()->GetHistoriesConsistentWithInfostate(player_id);
  }

  std::unique_ptr<
      std::pair<std::vector<std::unique_ptr<State>>,
                std::vector<double>>>
  GetHistoriesConsistentWithInfostate() const override {
    return states_.back()->GetHistoriesConsistentWithInfostate();
  }
};

}  // namespace open_spiel

#endif  // OPEN_SPIEL_PATH_H_
