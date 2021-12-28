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

#include "open_spiel/algorithms/corr_dist/afcce.h"

#include "open_spiel/abseil-cpp/absl/strings/str_format.h"
#include "open_spiel/abseil-cpp/absl/types/optional.h"
#include "open_spiel/spiel.h"

namespace open_spiel {
namespace algorithms {

AFCCEState::AFCCEState(std::shared_ptr<const Game> game,
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
      defection_infoset_(game->NumPlayers(), absl::nullopt),
      recommendation_seq_(game->NumPlayers(), std::vector<Action>({})) {}

Player AFCCEState::CurrentPlayer() const {
  // Only override this in the first chance actions.
  if (rec_index_ < 0) {
    return kChancePlayerId;
  } else {
    return state_->CurrentPlayer();
  }
}

ActionsAndProbs AFCCEState::ChanceOutcomes() const {
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

std::vector<Action> AFCCEState::LegalActions() const {
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
  } else if (HasDefected(CurrentPlayer()) &&
             !defection_infoset_[CurrentPlayer()].has_value()) {
    // Player just defected; now they must choose an action.
    return state_->LegalActions();
  } else {
    SPIEL_CHECK_TRUE(HasDefected(CurrentPlayer()));
    SPIEL_CHECK_TRUE(defection_infoset_[CurrentPlayer()].has_value());

    // This player already previously defected, so cannot do so any more.
    return {follow_action_};
  }
}

std::string AFCCEState::InformationStateString(Player player) const {
  // should look like <infoset string> <delimiter> <defected? true | false>
  // <recommendation sequence, (excluding current recommendation)>
  // <defection infoset, if available>
  std::string rec_str = absl::StrJoin(recommendation_seq_[player], ",");
  std::string infoset_str = state_->InformationStateString(player);
  SPIEL_CHECK_EQ(infoset_str.find(config_.recommendation_delimiter),
                 std::string::npos);
  // Note: no need to attach the defection location here because it can be
  // inferred from the -1 action in the recommendation sequence (due to perfect
  // recall), but we add it anyway if it's been determined yet.
  // Also note that there are two infosets for a defection:
  //   - The first one where a player chooses the defect action. Here, defected?
  //     is false and there is no defection infoset set yet
  //   - Directly after defection by the same player. Here defected? is true but
  //     the infoset is not yet set.
  // After the defection, defected? is set to to true and the defection infoset
  // is included in the infoset string.
  return absl::StrCat(infoset_str, config_.recommendation_delimiter,
                      HasDefected(player) ? "true " : "false ", rec_str,
                      defection_infoset_[player].has_value()
                          ? defection_infoset_[player].value()
                          : "");
}

std::string AFCCEState::ToString() const {
  std::string state_str = absl::StrFormat(
      "%s\nCur player: %i\nRec index %i\nDefected %s", state_->ToString(),
      CurrentPlayer(), rec_index_, absl::StrJoin(defected_, " "));
  for (Player p = 0; p < state_->NumPlayers(); ++p) {
    absl::StrAppend(&state_str, "\nPlayer ", p, " defection infoset: ",
                    !defection_infoset_[p].has_value()
                        ? "nullopt"
                        : defection_infoset_[p].value(),
                    "\n");
  }
  for (Player p = 0; p < state_->NumPlayers(); ++p) {
    absl::StrAppend(&state_str, "\nPlayer ", p, " recommendation seq: ",
                    absl::StrJoin(recommendation_seq_[p], ","), "\n");
  }
  return state_str;
}

bool AFCCEState::HasDefected(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, game_->NumPlayers());
  return defected_[player] == 1;
}

Action AFCCEState::CurRecommendation() const {
  ActionsAndProbs actions_and_probs =
      mu_[rec_index_].second.GetStatePolicy(state_->InformationStateString());
  Action rec_action = GetAction(actions_and_probs);
  SPIEL_CHECK_TRUE(rec_action != kInvalidAction);
  return rec_action;
}

void AFCCEState::DoApplyAction(Action action_id) {
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

      // Check for defection at this point.
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
    } else if (HasDefected(cur_player) &&
               !defection_infoset_[cur_player].has_value()) {
      // Player just defected: regular game from here on.
      state_->ApplyAction(action_id);
      defection_infoset_[cur_player] =
          state_->InformationStateString(cur_player);

      // Player is defecting, so fill this slot with invalid action. Defecting
      // players should never discover this recommendation.
      recommendation_seq_[cur_player].push_back(kInvalidAction);
    } else {
      SPIEL_CHECK_TRUE(HasDefected(cur_player));
      SPIEL_CHECK_TRUE(defection_infoset_[cur_player].has_value());

      // Already previously defected. Should only be follow.
      Action recommendation = CurRecommendation();
      SPIEL_CHECK_EQ(action_id, follow_action_);
      std::vector<Action> legal_actions = state_->LegalActions();
      SPIEL_CHECK_TRUE(absl::c_find(legal_actions, recommendation) !=
                       legal_actions.end());
      state_->ApplyAction(recommendation);
      recommendation_seq_[cur_player].push_back(recommendation);
    }
  }
}

ActionsAndProbs AFCCETabularPolicy::GetStatePolicy(const State& state) const {
  // The best response code has to have a policy defined everywhere when it
  // builds its initial tree. For the fixed policies, the players will not
  // defect, so we define a uniform policy in the regions where players have
  // defected (which will not affect the best responding player, since the
  // opponents will never reach these regions).
  const auto* AFCCE_state = dynamic_cast<const AFCCEState*>(&state);
  SPIEL_CHECK_TRUE(AFCCE_state != nullptr);
  if (AFCCE_state->HasDefected(state.CurrentPlayer())) {
    return UniformStatePolicy(state);
  }

  // Simply returns a fixed policy with prob 1 on the follow action
  return {{follow_action_, 1.0}, {defect_action_, 0.0}};
}

}  // namespace algorithms
}  // namespace open_spiel
