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

#ifndef OPEN_SPIEL_GAME_TRANSFORMS_COOP_TO_1P_H_
#define OPEN_SPIEL_GAME_TRANSFORMS_COOP_TO_1P_H_

#include <memory>

#include "open_spiel/game_transforms/game_wrapper.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

// Transforms a co-operative game into a 1-player environment, in which
// actions build a policy in the underlying game.
//
// We make very strong assumptions about the underlying game:
// - The initial num_players actions must be chance actions, one per player.
//   These are assumed to map 1:1 to the private state for that player.
// - The public state of the game is determined solely by the last non-chance
//   action.
//
// These assumptions hold for tiny_hanabi and tiny_bridge_2p, but are unlikely
// to hold in other games.

namespace open_spiel {
namespace coop_to_1p {

// Information we have about each player's private state.
struct PlayerPrivate {
  // Each private state may either - have a valid action assigned, be waiting
  // for an action assignment, or have been ruled out by prior play.
  static inline constexpr Action kImpossible = -100;
  static inline constexpr Action kUnassigned = -99;
  std::vector<Action> assignments;
  int next_unassigned;  // index into assignments

  // Name of each private state.
  std::vector<std::string> names;

  PlayerPrivate(int num_privates)
      : assignments(num_privates, kUnassigned),
        next_unassigned(0),
        names(num_privates) {}

  // Assign the next unassigned private.
  void Assign(Action action) {
    assignments[next_unassigned++] = action;
    while (next_unassigned < assignments.size() &&
           assignments[next_unassigned] != kUnassigned)
      ++next_unassigned;
  }

  // Have all assignments been made?
  bool AssignmentsComplete() const {
    return next_unassigned == assignments.size();
  }

  // Reset assignments for the next action choice.
  void Reset(Action action) {
    next_unassigned = assignments.size();
    for (int i = 0; i < assignments.size(); ++i) {
      if (assignments[i] == action) {
        if (next_unassigned == assignments.size()) next_unassigned = i;
        assignments[i] = kUnassigned;
      } else {
        assignments[i] = kImpossible;
      }
    }
  }
};

// This is a single player game.
inline constexpr Player kPlayerId = 0;

// The state is mostly a wrapper over the imperfect information state.
class CoopTo1pState : public State {
 public:
  CoopTo1pState(std::shared_ptr<const Game> game, int num_privates,
                std::unique_ptr<State> state)
      : State(game),
        state_(std::move(state)),
        num_privates_(num_privates),
        prev_player_(kInvalidPlayer),
        prev_action_(kInvalidAction) {}
  CoopTo1pState(const CoopTo1pState& other)
      : State(other),
        state_(other.state_->Clone()),
        num_privates_(other.num_privates_),
        privates_(other.privates_),
        actual_private_(other.actual_private_),
        prev_player_(other.prev_player_),
        prev_action_(other.prev_action_) {}
  Player CurrentPlayer() const override {
    Player underlying_player = state_->CurrentPlayer();
    return underlying_player < 0 ? underlying_player : kPlayerId;
  }
  std::vector<Action> LegalActions(Player player) const override {
    if (player == CurrentPlayer())
      return LegalActions();
    else
      return {};
  }
  std::vector<Action> LegalActions() const override {
    return state_->LegalActions(state_->CurrentPlayer());
  }
  std::vector<int> LegalActionsMask() const {
    return state_->LegalActionsMask(state_->CurrentPlayer());
  }
  bool IsTerminal() const override { return state_->IsTerminal(); }
  std::vector<double> Rewards() const override {
    return {state_->Rewards().front()};
  }
  std::vector<double> Returns() const override {
    return {state_->Returns().front()};
  }
  std::unique_ptr<State> Clone() const override;
  std::string ActionToString(Player player, Action action_id) const override;
  std::string ToString() const override;
  std::string ObservationString(Player player) const override;
  void ObservationTensor(Player player,
                         absl::Span<float> values) const override;
  ActionsAndProbs ChanceOutcomes() const override {
    return state_->ChanceOutcomes();
  }
  std::vector<Action> LegalChanceOutcomes() const override {
    return state_->LegalChanceOutcomes();
  }

 protected:
  void DoApplyAction(Action action_id) override;

 private:
  std::unique_ptr<State> state_;
  int num_privates_;
  std::vector<PlayerPrivate> privates_;
  std::vector<Action> actual_private_;
  Player prev_player_;
  Action prev_action_;

  std::string Assignments() const;
  std::string PublicStateString() const;
  std::string AssignmentToString(Player player, Action assignment) const;
};

class CoopTo1pGame : public Game {
 public:
  CoopTo1pGame(std::shared_ptr<const Game> game, GameType game_type,
               GameParameters game_parameters);
  std::unique_ptr<State> NewInitialState() const override;
  int NumPlayers() const override { return 1; }
  std::vector<int> ObservationTensorShape() const override;
  int MaxGameLength() const override;
  int MaxChanceNodesInHistory() const override {
    return game_->MaxGameLength();
  }

  int NumDistinctActions() const override {
    return game_->NumDistinctActions();
  }
  int MaxChanceOutcomes() const override { return game_->MaxChanceOutcomes(); }
  double MinUtility() const override { return game_->MinUtility(); }
  double MaxUtility() const override { return game_->MaxUtility(); }
  double UtilitySum() const override { return game_->UtilitySum(); }

 private:
  std::shared_ptr<const Game> game_;
  int NumPrivates() const { return game_->MaxChanceOutcomes(); }
};

}  // namespace coop_to_1p
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAME_TRANSFORMS_COOP_TO_1P_H_
