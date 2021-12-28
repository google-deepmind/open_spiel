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

#ifndef OPEN_SPIEL_ALGORITHMS_CORR_DIST_EFCE_H_
#define OPEN_SPIEL_ALGORITHMS_CORR_DIST_EFCE_H_

#include <vector>

#include "open_spiel/abseil-cpp/absl/types/optional.h"
#include "open_spiel/algorithms/corr_dist.h"
#include "open_spiel/game_transforms/game_wrapper.h"
#include "open_spiel/policy.h"
#include "open_spiel/spiel.h"

// The implementations of the metrics assemble the extended game described in
// von Stengel and Forges 2008, Definition 2.2. Then, the incentive to deviate
// to a best response is computed by running NashConv on this auxiliary game.
//
// The EFCE extended game modifies the original game in the following ways:
//   - An internal variable is kept to determine if the player has deviated.
//     If so, recommendations are no longer given.
//   - If config.deterministic and config.convert are both false, the game
//     starts with a "Monte Carlo" chance node that corresponds to different
//     samplings of deterministic joint policies. The number of samples (and
//     corresponding accuracy) is determined by config.num_samples. If either
//     config.deterministic is true or config.convert is true, this chance node
//     will not exist.
//   - A "joint policy" chance node that corresponds to choosing a joint policy
//     from the correlation device.
//   - Information state keys are modified to include the recommendations
//     received at the current information state and ones received up to this
//     information state, i.e. the sequence of recommendations. New
//     recommendations stop getting appended once the player chooses an action
//     that does not match the recommendation.
//
// In addition, a specific tabular policy is made so as to map the policies
// in this new game back to the original game.

namespace open_spiel {
namespace algorithms {

class EFCEState : public WrappedState {
 public:
  EFCEState(std::shared_ptr<const Game> game, std::unique_ptr<State> state,
            CorrDistConfig config, const CorrelationDevice& mu);

  std::unique_ptr<State> Clone() const override {
    return std::make_unique<EFCEState>(*this);
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
  std::string ToString() const;

  bool HasDefected(Player player) const;

 protected:
  Action CurRecommendation() const;
  void DoApplyAction(Action action_id) override;

 private:
  CorrDistConfig config_;
  const CorrelationDevice& mu_;

  // Which joint policy was chosen?
  int rec_index_;

  // Has the player defected?
  std::vector<int> defected_;

  // The sequence of recommendations, indexed by player
  std::vector<std::vector<Action>> recommendation_seq_;
};

class EFCEGame : public WrappedGame {
 public:
  EFCEGame(std::shared_ptr<const Game> game, CorrDistConfig config,
           const CorrelationDevice& mu)
      : WrappedGame(game, game->GetType(), game->GetParameters()),
        config_(config),
        mu_(mu) {}

  std::unique_ptr<State> NewInitialState() const override {
    return std::make_unique<EFCEState>(shared_from_this(),
                                       game_->NewInitialState(), config_, mu_);
  }

 protected:
  const CorrDistConfig config_;
  const CorrelationDevice& mu_;
};

class EFCETabularPolicy : public TabularPolicy {
 public:
  EFCETabularPolicy(const CorrDistConfig& config) : config_(config) {}

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

#endif  // OPEN_SPIEL_ALGORITHMS_CORR_DIST_EFCE_H_
