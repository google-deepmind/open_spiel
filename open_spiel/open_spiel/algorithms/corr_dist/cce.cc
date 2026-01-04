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

#include "open_spiel/algorithms/corr_dist/cce.h"

#include "open_spiel/abseil-cpp/absl/strings/str_format.h"
#include "open_spiel/algorithms/corr_dist/efcce.h"
#include "open_spiel/spiel_globals.h"

namespace open_spiel {
namespace algorithms {

CCEState::CCEState(std::shared_ptr<const Game> game,
                   std::unique_ptr<State> state, CorrDistConfig config,
                   const CorrelationDevice& mu)
    : WrappedState(game, std::move(state)),
      config_(config),
      mu_(mu),
      rec_index_(-1) {}

Player CCEState::CurrentPlayer() const {
  // Only override this in the first chance actions.
  if (rec_index_ < 0) {
    return kChancePlayerId;
  } else {
    return state_->CurrentPlayer();
  }
}

ActionsAndProbs CCEState::ChanceOutcomes() const {
  if (rec_index_ < 0) {
    ActionsAndProbs outcomes;
    for (int i = 0; i < mu_.size(); ++i) {
      outcomes.push_back({i, mu_[i].first});
    }
    return outcomes;
  } else {
    return state_->ChanceOutcomes();
  }
}

std::vector<Action> CCEState::LegalActions() const {
  SPIEL_CHECK_FALSE(IsSimultaneousNode());

  if (IsTerminal()) {
    return {};
  } else if (IsChanceNode()) {
    return LegalChanceOutcomes();
  } else {
    return state_->LegalActions();
  }
}

std::string CCEState::InformationStateString(Player player) const {
  return state_->InformationStateString(player);
}

std::string CCEState::ToString() const {
  std::string state_str =
      absl::StrFormat("%s\nCur player: %i\nRec index %i", state_->ToString(),
                      CurrentPlayer(), rec_index_);
  return state_str;
}

void CCEState::DoApplyAction(Action action_id) {
  if (rec_index_ < 0) {
    // Pick the joint policy which will provide recommendations.
    rec_index_ = action_id;
    SPIEL_CHECK_LT(rec_index_, mu_.size());
  } else if (state_->IsChanceNode()) {
    // Regular chance node
    state_->ApplyAction(action_id);
  } else {
    // Regular decision node
    state_->ApplyAction(action_id);
  }
}

ActionsAndProbs CCEState::CurrentRecommendedStatePolicy() const {
  SPIEL_CHECK_GE(rec_index_, 0);
  return mu_[rec_index_].second.GetStatePolicy(
      InformationStateString(CurrentPlayer()));
}

ActionsAndProbs CCETabularPolicy::GetStatePolicy(const State& state) const {
  const auto* cce_state = dynamic_cast<const CCEState*>(&state);
  SPIEL_CHECK_TRUE(cce_state != nullptr);
  return cce_state->CurrentRecommendedStatePolicy();
}

}  // namespace algorithms
}  // namespace open_spiel
