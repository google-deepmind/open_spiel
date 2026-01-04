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

#ifndef OPEN_SPIEL_ALGORITHMS_CORR_DIST_AFCCE_H_
#define OPEN_SPIEL_ALGORITHMS_CORR_DIST_AFCCE_H_

#include <vector>

#include "open_spiel/abseil-cpp/absl/types/optional.h"
#include "open_spiel/algorithms/corr_dist.h"
#include "open_spiel/game_transforms/game_wrapper.h"
#include "open_spiel/policy.h"
#include "open_spiel/spiel.h"

namespace open_spiel {
namespace algorithms {

// The AFCCE auxiliary game is very similar to the EFCCE auxiliary game. The
// only difference is that the number of deviations is limited to 1. Once the
// player has deviated, they can no longer deviate and must follow for the
// rest of the game.
class AFCCEState : public WrappedState {
 public:
  AFCCEState(std::shared_ptr<const Game> game, std::unique_ptr<State> state,
             CorrDistConfig config, const CorrelationDevice& mu,
             Action follow_action, Action defect_action);

  std::unique_ptr<State> Clone() const override {
    return std::make_unique<AFCCEState>(*this);
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

  bool HasDefected(Player player) const;

 protected:
  Action CurRecommendation() const;
  void DoApplyAction(Action action_id) override;

 private:
  const CorrDistConfig config_;
  const CorrelationDevice& mu_;

  Action follow_action_;
  Action defect_action_;

  // Which joint policy was chosen?
  int rec_index_;

  // Has the player defected?
  std::vector<int> defected_;

  // Where did the player defect? This is the information set of the original
  // game. Indexed by player.
  std::vector<absl::optional<std::string>> defection_infoset_;

  // The sequence of recommendations, indexed by player
  std::vector<std::vector<Action>> recommendation_seq_;
};

class AFCCEGame : public WrappedGame {
 public:
  AFCCEGame(std::shared_ptr<const Game> game, CorrDistConfig config,
            const CorrelationDevice& mu)
      : WrappedGame(game, game->GetType(), game->GetParameters()),
        config_(config),
        mu_(mu),
        orig_num_distinct_actions_(game->NumDistinctActions()) {}

  std::unique_ptr<State> NewInitialState() const override {
    return std::make_unique<AFCCEState>(shared_from_this(),
                                        game_->NewInitialState(), config_, mu_,
                                        FollowAction(), DefectAction());
  }

  int NumDistinctActions() const override {
    // 2 extra actions: cooperate/follow or defect
    return orig_num_distinct_actions_ + 2;
  }

  int FollowAction() const { return orig_num_distinct_actions_; }
  int DefectAction() const { return orig_num_distinct_actions_ + 1; }

 private:
  const CorrDistConfig config_;
  const CorrelationDevice& mu_;

  // Number of distinct actions in the original game.
  int orig_num_distinct_actions_;
};

class AFCCETabularPolicy : public TabularPolicy {
 public:
  AFCCETabularPolicy(Action follow_action, Action defect_action)
      : follow_action_(follow_action), defect_action_(defect_action) {}

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
  const Action follow_action_;
  const Action defect_action_;
};

}  // namespace algorithms
}  // namespace open_spiel

#endif  // OPEN_SPIEL_ALGORITHMS_CORR_DIST_AFCCE_H_
