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

#include "open_spiel/algorithms/corr_dist/efcce.h"

#include "open_spiel/abseil-cpp/absl/strings/str_format.h"
#include "open_spiel/spiel_globals.h"

namespace open_spiel {
namespace algorithms {

EFCCEState::EFCCEState(std::shared_ptr<const Game> game,
                       std::unique_ptr<State> state, CorrDistConfig config,
                       const CorrelationDevice& mu, Action follow_action,
                       Action defect_action)
    : WrappedState(game, std::move(state)),
      config_(config),
      mu_(mu),
      follow_action_(follow_action),
      defect_action_(defect_action),
      rec_index_(-1),
      defected_(game->NumPlayers(), 0),
      recommendation_seq_(game->NumPlayers(), std::vector<Action>({})) {}

Player EFCCEState::CurrentPlayer() const {
  // Only override this in the first chance actions.
  if (rec_index_ < 0) {
    return kChancePlayerId;
  } else {
    return state_->CurrentPlayer();
  }
}

ActionsAndProbs EFCCEState::ChanceOutcomes() const {
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

std::vector<Action> EFCCEState::LegalActions() const {
  SPIEL_CHECK_FALSE(IsSimultaneousNode());

  if (IsTerminal()) {
    return {};
  } else if (IsChanceNode()) {
    return LegalChanceOutcomes();
  }

  if (!HasDefected(CurrentPlayer())) {
    // If the player has not defected then they have exactly two choices:
    // follow or defect.
    return {follow_action_, defect_action_};
  } else {
    // Player has defected.. they are on their own.
    return state_->LegalActions();
  }
}

std::string EFCCEState::InformationStateString(Player player) const {
  // should look like <infoset string> <delimiter> <defected? true | false>
  // <recommendation sequence, (excluding current recommendation)>
  std::string rec_str = absl::StrJoin(recommendation_seq_[player], ",");
  std::string infoset_str = state_->InformationStateString(player);
  SPIEL_CHECK_EQ(infoset_str.find(config_.recommendation_delimiter),
                 std::string::npos);
  return absl::StrCat(infoset_str, config_.recommendation_delimiter,
                      HasDefected(player) ? "true " : "false ", rec_str);
}

std::string EFCCEState::ToString() const {
  std::string state_str = absl::StrFormat(
      "%s\nCur player: %i\nRec index %i\nDefected %s", state_->ToString(),
      CurrentPlayer(), rec_index_, absl::StrJoin(defected_, " "));
  for (Player p = 0; p < state_->NumPlayers(); ++p) {
    absl::StrAppend(&state_str, "\nPlayer ", p, " recommendation seq: ",
                    absl::StrJoin(recommendation_seq_[p], ","));
  }
  return state_str;
}

bool EFCCEState::HasDefected(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, game_->NumPlayers());
  return defected_[player] == 1;
}

Action EFCCEState::CurRecommendation() const {
  ActionsAndProbs actions_and_probs =
      mu_[rec_index_].second.GetStatePolicy(state_->InformationStateString());
  Action rec_action = GetAction(actions_and_probs);
  SPIEL_CHECK_TRUE(rec_action != kInvalidAction);
  return rec_action;
}

void EFCCEState::DoApplyAction(Action action_id) {
  if (rec_index_ < 0) {
    // Pick the joint policy which will provide recommendations.
    rec_index_ = action_id;
    SPIEL_CHECK_LT(rec_index_, mu_.size());
  } else if (state_->IsChanceNode()) {
    // Regular chance node
    state_->ApplyAction(action_id);
  } else {
    Player cur_player = CurrentPlayer();
    SPIEL_CHECK_GE(cur_player, 0);
    SPIEL_CHECK_LT(cur_player, game_->NumPlayers());

    if (!HasDefected(cur_player)) {
      // Can only submit these two actions.
      SPIEL_CHECK_TRUE(action_id == follow_action_ ||
                       action_id == defect_action_);

      // Check for defection at this point. This is because the
      // recommendations
      Action recommendation = CurRecommendation();

      if (action_id == follow_action_) {
        // Follow recommendation.
        std::vector<Action> legal_actions = state_->LegalActions();
        SPIEL_CHECK_TRUE(absl::c_find(legal_actions, recommendation) !=
                         legal_actions.end());
        state_->ApplyAction(recommendation);
        recommendation_seq_[cur_player].push_back(recommendation);
      } else {
        // Defect.
        defected_[cur_player] = 1;
      }

    } else {
      // Regular game from here on.
      state_->ApplyAction(action_id);
    }
  }
}

ActionsAndProbs EFCCETabularPolicy::GetStatePolicy(const State& state) const {
  // The best response code has to have a policy defined everywhere when it
  // builds its initial tree. For the fixed policies, the players will not
  // defect, so we define a uniform policy in the regions where players have
  // defected (which will not affect the best responding player, since the
  // opponents will never reach these regions).
  const auto* efcce_state = dynamic_cast<const EFCCEState*>(&state);
  SPIEL_CHECK_TRUE(efcce_state != nullptr);
  if (efcce_state->HasDefected(state.CurrentPlayer())) {
    return UniformStatePolicy(state);
  }

  // Simply returns a fixed policy with prob 1 on the follow action
  return {{follow_action_, 1.0}, {defect_action_, 0.0}};
}

}  // namespace algorithms
}  // namespace open_spiel
