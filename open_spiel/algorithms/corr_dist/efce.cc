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

#include "open_spiel/algorithms/corr_dist/efce.h"

#include "open_spiel/abseil-cpp/absl/strings/str_format.h"

namespace open_spiel {
namespace algorithms {

EFCEState::EFCEState(std::shared_ptr<const Game> game,
                     std::unique_ptr<State> state, CorrDistConfig config,
                     const CorrelationDevice& mu)
    : WrappedState(game, std::move(state)),
      config_(config),
      mu_(mu),
      rec_index_(-1),
      defected_(game->NumPlayers(), 0),
      recommendation_seq_(game->NumPlayers(), std::vector<Action>({})) {}

Player EFCEState::CurrentPlayer() const {
  // Only override this in the first chance actions.
  if (rec_index_ < 0) {
    return kChancePlayerId;
  } else {
    return state_->CurrentPlayer();
  }
}

ActionsAndProbs EFCEState::ChanceOutcomes() const {
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

std::vector<Action> EFCEState::LegalActions() const {
  SPIEL_CHECK_FALSE(IsSimultaneousNode());

  if (IsTerminal()) {
    return {};
  } else if (IsChanceNode()) {
    return LegalChanceOutcomes();
  }

  return state_->LegalActions();
}

std::string EFCEState::InformationStateString(Player player) const {
  // should look like <infoset string> <delimiter> <recommendation sequence,
  // (including current recommendation)>
  SPIEL_CHECK_FALSE(IsChanceNode());
  std::string rec_str = absl::StrJoin(recommendation_seq_[player], ",");
  if (!HasDefected(player)) {
    absl::StrAppend(&rec_str, ",", CurRecommendation());
  }
  std::string infoset_str = state_->InformationStateString(player);
  SPIEL_CHECK_EQ(infoset_str.find(config_.recommendation_delimiter),
                 std::string::npos);
  return absl::StrCat(infoset_str, config_.recommendation_delimiter, rec_str);
}

std::string EFCEState::ToString() const {
  std::string state_str = absl::StrFormat(
      "%s\nCur player: %i\nRec index %i\nDefected %s", state_->ToString(),
      CurrentPlayer(), rec_index_, absl::StrJoin(defected_, " "));
  for (Player p = 0; p < state_->NumPlayers(); ++p) {
    absl::StrAppend(&state_str, "\nPlayer ", p, " recommendation seq: ",
                    absl::StrJoin(recommendation_seq_[p], ","));
  }
  return state_str;
}

bool EFCEState::HasDefected(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, game_->NumPlayers());
  return defected_[player] == 1;
}

Action EFCEState::CurRecommendation() const {
  SPIEL_CHECK_GE(rec_index_, 0);
  SPIEL_CHECK_LT(rec_index_, mu_.size());
  ActionsAndProbs actions_and_probs =
      mu_[rec_index_].second.GetStatePolicy(state_->InformationStateString());
  Action rec_action = GetAction(actions_and_probs);
  SPIEL_CHECK_TRUE(rec_action != kInvalidAction);
  return rec_action;
}

void EFCEState::DoApplyAction(Action action_id) {
  if (rec_index_ < 0) {
    // Pick the joint policy which will provide recommendations.
    rec_index_ = action_id;
    SPIEL_CHECK_GE(rec_index_, 0);
    SPIEL_CHECK_LT(rec_index_, mu_.size());
  } else if (state_->IsChanceNode()) {
    // Regular chance node
    state_->ApplyAction(action_id);
  } else {
    // Check for defection at this point.
    const Action recommendation = CurRecommendation();

    Player cur_player = CurrentPlayer();
    SPIEL_CHECK_GE(cur_player, 0);
    SPIEL_CHECK_LT(cur_player, game_->NumPlayers());

    // If they have defected, don't add to the sequence
    if (!HasDefected(cur_player)) {
      recommendation_seq_[cur_player].push_back(recommendation);

      // If they chose an action other than the recommendation, they have now
      // defected.
      if (action_id != recommendation) {
        defected_[cur_player] = 1;
      }
    }

    state_->ApplyAction(action_id);
  }
}

ActionsAndProbs EFCETabularPolicy::GetStatePolicy(const State& state) const {
  // The best response code has to have a policy defined everywhere when it
  // builds its initial tree. For the fixed policies, the players will not
  // defect, so we define a uniform policy in the regions where players have
  // defected (which will not affect the best responding player, since the
  // opponents will never reach these regions).
  const auto* efce_state = dynamic_cast<const EFCEState*>(&state);
  SPIEL_CHECK_TRUE(efce_state != nullptr);
  if (efce_state->HasDefected(state.CurrentPlayer())) {
    return UniformStatePolicy(state);
  }

  // Otherwise, simply returns a fixed policy with prob 1 on the recommended
  // action (extrapolated from the information state string) and 0 on the
  // others.
  std::string info_state = state.InformationStateString();
  const size_t idx = info_state.find(config_.recommendation_delimiter);
  SPIEL_CHECK_NE(idx, std::string::npos);
  std::vector<std::string> rec_seq = absl::StrSplit(
      info_state.substr(idx + config_.recommendation_delimiter.length()), ',');
  SPIEL_CHECK_GE(rec_seq.size(), 1);
  Action rec_action;
  ActionsAndProbs state_policy;
  std::vector<Action> legal_actions = state.LegalActions();
  state_policy.reserve(legal_actions.size());
  SPIEL_CHECK_TRUE(absl::SimpleAtoi(rec_seq.back(), &rec_action));
  for (Action action : legal_actions) {
    state_policy.push_back({action, action == rec_action ? 1.0 : 0.0});
  }
  return state_policy;
}

}  // namespace algorithms
}  // namespace open_spiel
