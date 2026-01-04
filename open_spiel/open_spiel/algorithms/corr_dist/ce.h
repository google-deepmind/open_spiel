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

#ifndef OPEN_SPIEL_ALGORITHMS_CORR_DIST_CE_H_
#define OPEN_SPIEL_ALGORITHMS_CORR_DIST_CE_H_

#include <vector>

#include "open_spiel/abseil-cpp/absl/container/flat_hash_map.h"
#include "open_spiel/abseil-cpp/absl/types/optional.h"
#include "open_spiel/algorithms/corr_dist.h"
#include "open_spiel/game_transforms/game_wrapper.h"
#include "open_spiel/policy.h"
#include "open_spiel/spiel.h"

namespace open_spiel {
namespace algorithms {

class CEState : public WrappedState {
 public:
  CEState(std::shared_ptr<const Game> game, std::unique_ptr<State> state,
          CorrDistConfig config, const CorrelationDevice& mu);

  std::unique_ptr<State> Clone() const override {
    return std::make_unique<CEState>(*this);
  }

  // Need to override this because otherwise WrappedState forwards the
  // implementation to the underlying state, which calls the wrong
  // ChanceOutcomes
  std::vector<Action> LegalChanceOutcomes() const override {
    return State::LegalChanceOutcomes();
  }

  Player CurrentPlayer() const override;
  ActionsAndProbs ChanceOutcomes() const override;
  std::vector<Action> LegalActions() const override;
  std::string InformationStateString(Player player) const override;
  std::string ToString() const override;

  ActionsAndProbs RecommendedStatePolicy(const std::string& info_state) const;

 protected:
  void DoApplyAction(Action action_id) override;

 private:
  const CorrDistConfig config_;
  const CorrelationDevice& mu_;

  // Which joint policy was chosen?
  int rec_index_;
};

class CEGame : public WrappedGame {
 public:
  CEGame(std::shared_ptr<const Game> game, CorrDistConfig config,
         const CorrelationDevice& mu);

  std::unique_ptr<State> NewInitialState() const override {
    return std::make_unique<CEState>(shared_from_this(),
                                     game_->NewInitialState(), config_, mu_);
  }

  // Returns a signal id, which corresponds to a specific policy used by the
  // specified player in the joint policy at the specified recommendation index
  // in the correlation device. This method makes use of a table that maintains
  // these mappings initialized at construction time.
  int GetSignalId(int rec_index, Player player) const {
    const auto iter = recidx_player_to_signal_id_.find({rec_index, player});
    SPIEL_CHECK_TRUE(iter != recidx_player_to_signal_id_.end());
    return iter->second;
  }

  int NumDistinctActions() const override { return orig_num_distinct_actions_; }

 private:
  const CorrDistConfig config_;
  const CorrelationDevice& mu_;

  // Number of distinct actions in the original game.
  int orig_num_distinct_actions_;

  // To compute a correlated equilibria, we need to map individual player
  // strategies to signal ids that are handed out by the recommender at the
  // start of the game. These signal ids get tacked onto the information state
  // strings so that the best response is computed conditionally on the signal.
  //
  // Keeps track of the number of signal id's per player.
  std::vector<int> signal_ids_;

  // Information state identifiers in this game.
  absl::flat_hash_map<std::string, Player> info_state_to_player_;

  // A (sorted Tabular policy string, player id) -> signal id map.
  absl::flat_hash_map<std::pair<std::string, Player>, int>
      policy_player_to_signal_id_;

  // A (recommendation index, player id) -> signal id map.
  absl::flat_hash_map<std::pair<int, Player>, int> recidx_player_to_signal_id_;
};

class CETabularPolicy : public TabularPolicy {
 public:
  CETabularPolicy(CorrDistConfig config) : config_(config) {}

  ActionsAndProbs GetStatePolicy(const std::string& info_state) const override {
    SpielFatalError("GetStatePolicy(const std::string&) should not be called.");
    return TabularPolicy::GetStatePolicy(info_state);
  }
  ActionsAndProbs GetStatePolicy(const State& state, Player pl) const override {
    SPIEL_CHECK_EQ(state.CurrentPlayer(), pl);
    return GetStatePolicy(state);
  }
  ActionsAndProbs GetStatePolicy(const State& state) const override;

 private:
  const CorrDistConfig config_;
};

}  // namespace algorithms
}  // namespace open_spiel

#endif  // OPEN_SPIEL_ALGORITHMS_CORR_DIST_CE_H_
