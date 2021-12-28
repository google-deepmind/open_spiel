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

#ifndef OPEN_SPIEL_GAME_TRANSFORMS_RESTRICTED_NASH_RESPONSE_H_
#define OPEN_SPIEL_GAME_TRANSFORMS_RESTRICTED_NASH_RESPONSE_H_

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "open_spiel/game_transforms/game_wrapper.h"
#include "open_spiel/spiel.h"
#include "open_spiel/policy.h"

// An implementation of Restricted Nash Response by Johanson et al. '08:
// http://www.johanson.ca/publications/poker/2007-nips-rnash/2007-nips-rnash.html

namespace open_spiel {

constexpr Player kDefaultFixedPlayer = 0;
constexpr double kDefaultP = 0.5;

enum { kFixedAction = 0, kFreeAction = 1 };

class RestrictedNashResponseObserver;

class RestrictedNashResponseState : public State {
 public:
  RestrictedNashResponseState(std::shared_ptr<const Game> game,
                              std::unique_ptr<State> state, bool fixed,
                              Player fixed_player, bool initial_state, double p,
                              std::shared_ptr<Policy> fixed_policy);

  RestrictedNashResponseState(const RestrictedNashResponseState &other);

  Player CurrentPlayer() const override;

  std::string ActionToString(Player player, Action action_id) const override;

  std::string ToString() const override;

  bool IsTerminal() const override;

  std::vector<double> Returns() const override;

  std::string InformationStateString(Player player) const override;

  void InformationStateTensor(Player player,
                              absl::Span<float> values) const override;

  std::string ObservationString(Player player) const override;

  void ObservationTensor(Player player,
                         absl::Span<float> values) const override;

  std::unique_ptr<State> Clone() const override;

  std::vector<std::pair<Action, double>> ChanceOutcomes() const override;

  std::vector<Action> LegalActions(Player player) const override;

  std::vector<Action> LegalActions() const override;

  std::shared_ptr<const Game> GetOriginalGame() const {
    return state_->GetGame();
  }

  bool IsPlayerFixed(Player player) const { return player == fixed_player_; }

  bool IsStateFixed() const { return fixed_; }

  std::shared_ptr<State> GetOriginalState() const { return state_; }

  bool IsRestrictedNashResponseInitialState() const { return is_initial_; }

 protected:
  void DoApplyAction(Action action_id) override;
  void DoApplyActions(const std::vector<Action> &actions) override;

 private:
  // underlying state
  std::shared_ptr<State> state_;

  // Variables showing if we are in the initial state and if not whether this
  // part is fixed or not.
  bool is_initial_;
  bool fixed_;
  // Constants representing p value and the player who is fixed.
  const double p_;
  const Player fixed_player_;
  // Constants for the fixed strategy and if we use explicit fixed strategy
  std::shared_ptr<Policy> fixed_policy_;
  const bool use_fixed_policy_;
};

class RestrictedNashResponseGame : public WrappedGame {
 public:
  explicit RestrictedNashResponseGame(
      std::shared_ptr<const Game> game, Player fixed_player, double p,
      std::shared_ptr<Policy> fixed_policy = nullptr);
  std::shared_ptr<Observer> MakeObserver(
      absl::optional<IIGObservationType> iig_obs_type,
      const GameParameters& params) const;

  std::unique_ptr<State> NewInitialState() const override {
    return std::unique_ptr<State>(new RestrictedNashResponseState(
        shared_from_this(), game_->NewInitialState(), false, fixed_player_,
        true, p_, fixed_policy_));
  }

  int NumDistinctActions() const override {
    return game_->NumDistinctActions();
  }

  int MaxChanceOutcomes() const override {
    if (fixed_policy_) {
      // If a fixed policy is set, then we have a real RNR game, which means
      // there is at least one chance node with 2 outcomes. But also, the
      // fixed player actions are also treated as chance nodes, so the number
      // of distinct actions can also determine the maximum number of chance
      // outcomes.
      std::vector<int> candidates = {
        game_->MaxChanceOutcomes(), 2, game_->NumDistinctActions()
      };
      return *std::max_element(candidates.begin(), candidates.end());
    } else {
      // Otherwise, it's the normal game.
      return game_->MaxChanceOutcomes();
    }
  }

  int NumPlayers() const override { return game_->NumPlayers(); }

  double MinUtility() const override { return game_->MinUtility(); }

  double MaxUtility() const override { return game_->MaxUtility(); }

  double UtilitySum() const override { return game_->UtilitySum(); }

  std::vector<int> InformationStateTensorShape() const override {
    // Underlying game plus
    return {2 + game_->InformationStateTensorSize()};
  }

  std::vector<int> ObservationTensorShape() const override {
    // We flatten the representation of the underlying game and add one-hot
    // indications of the to-play player and the observing player.
    return {2 + game_->ObservationTensorSize()};
  }

  int MaxGameLength() const override { return game_->MaxGameLength() + 1; }

  int MaxChanceNodesInHistory() const override {
    if (fixed_policy_) {
      // If a fixed policy is set, then we have a real RNR game, which has an
      // extra chance node.
      return game_->MaxChanceNodesInHistory() + 1;
    } else {
      // Otherwise, it's just the normal game.
      return game_->MaxChanceNodesInHistory();
    }
  }
  // old observation API
  std::shared_ptr<RestrictedNashResponseObserver> default_observer_;
  std::shared_ptr<RestrictedNashResponseObserver> info_state_observer_;

 private:
  // Fixed player and p constants to be passed to the initial state
  const Player fixed_player_;
  const double p_;
  // Constants for the fixed strategy and if we use explicit fixed strategy
  std::shared_ptr<Policy> fixed_policy_;
};

// Return back a transformed clone of the game.
std::shared_ptr<const Game> ConvertToRNR(
    const Game& game, Player fixed_player, double p,
    std::shared_ptr<Policy> fixed_policy = nullptr);
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAME_TRANSFORMS_RESTRICTED_NASH_RESPONSE_H_
