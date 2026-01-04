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

#include "open_spiel/algorithms/corr_dist/ce.h"

#include "open_spiel/abseil-cpp/absl/strings/str_format.h"
#include "open_spiel/algorithms/corr_dist/efcce.h"
#include "open_spiel/algorithms/get_all_infostates.h"
#include "open_spiel/policy.h"
#include "open_spiel/spiel_globals.h"

namespace open_spiel {
namespace algorithms {

CEState::CEState(std::shared_ptr<const Game> game, std::unique_ptr<State> state,
                 CorrDistConfig config, const CorrelationDevice& mu)
    : WrappedState(game, std::move(state)),
      config_(config),
      mu_(mu),
      rec_index_(-1) {}

Player CEState::CurrentPlayer() const {
  // Only override this in the first chance actions.
  if (rec_index_ < 0) {
    return kChancePlayerId;
  } else {
    return state_->CurrentPlayer();
  }
}

ActionsAndProbs CEState::ChanceOutcomes() const {
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

std::vector<Action> CEState::LegalActions() const {
  SPIEL_CHECK_FALSE(IsSimultaneousNode());

  if (IsTerminal()) {
    return {};
  } else if (IsChanceNode()) {
    return LegalChanceOutcomes();
  } else {
    return state_->LegalActions();
  }
}

std::string CEState::InformationStateString(Player player) const {
  // should look like <infoset string> <delimiter> <signal id>
  SPIEL_CHECK_FALSE(IsChanceNode());
  std::string infoset_str = state_->InformationStateString(player);
  SPIEL_CHECK_EQ(infoset_str.find(config_.recommendation_delimiter),
                 std::string::npos);
  const auto* parent_game = down_cast<const CEGame*>(game_.get());
  SPIEL_CHECK_GE(rec_index_, 0);
  int signal_id = parent_game->GetSignalId(rec_index_, player);
  return absl::StrCat(infoset_str, config_.recommendation_delimiter, signal_id);
}

std::string CEState::ToString() const {
  std::string state_str =
      absl::StrFormat("%s\nCur player: %i\nRec index %i", state_->ToString(),
                      CurrentPlayer(), rec_index_);
  return state_str;
}

void CEState::DoApplyAction(Action action_id) {
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

ActionsAndProbs CEState::RecommendedStatePolicy(
    const std::string& info_state) const {
  SPIEL_CHECK_GE(rec_index_, 0);
  return mu_[rec_index_].second.GetStatePolicy(info_state);
}

CEGame::CEGame(std::shared_ptr<const Game> game, CorrDistConfig config,
               const CorrelationDevice& mu)
    : WrappedGame(game, game->GetType(), game->GetParameters()),
      config_(config),
      mu_(mu),
      orig_num_distinct_actions_(game->NumDistinctActions()),
      signal_ids_(game->NumPlayers(), 0) {
  // First, build the map that will identify which information states belong
  // to which player.
  {
    std::vector<std::vector<std::string>> all_infostates =
        GetAllInformationStates(*game);
    SPIEL_CHECK_EQ(all_infostates.size(), game->NumPlayers());
    for (Player p = 0; p < all_infostates.size(); ++p) {
      for (const std::string& info_state : all_infostates[p]) {
        const auto iter = info_state_to_player_.find(info_state);
        if (iter != info_state_to_player_.end()) {
          SpielFatalError("Duplicate information set found!");
        }
        info_state_to_player_[info_state] = p;
      }
    }
  }

  // Now, go through each joint policy in the correlation device, splitting them
  // among players.
  for (int rec_index = 0; rec_index < mu_.size(); ++rec_index) {
    const TabularPolicy& joint_policy = mu_[rec_index].second;

    // Split the policies into individual player policies.
    std::vector<TabularPolicy> player_policies(game->NumPlayers());
    for (const auto& [info_state, action_probs] : joint_policy.PolicyTable()) {
      const auto player_iter = info_state_to_player_.find(info_state);
      SPIEL_CHECK_TRUE(player_iter != info_state_to_player_.end());
      Player player = player_iter->second;
      player_policies[player].SetStatePolicy(info_state, action_probs);
    }

    // Lookup / assign signals to each individual policy.
    for (Player p = 0; p < player_policies.size(); ++p) {
      std::string sorted_policy_string = player_policies[p].ToStringSorted();
      std::pair<std::string, Player> key = {sorted_policy_string, p};
      const auto iter = policy_player_to_signal_id_.find(key);
      int signal_id = -1;

      if (iter == policy_player_to_signal_id_.end()) {
        // Signal for this policy does not exist yet, use the next one.
        signal_id = signal_ids_[p]++;
        policy_player_to_signal_id_[key] = signal_id;
      } else {
        signal_id = iter->second;
      }

      recidx_player_to_signal_id_[{rec_index, p}] = signal_id;
    }
  }
}

ActionsAndProbs CETabularPolicy::GetStatePolicy(const State& state) const {
  // Here we must scrape off the signal id so that the BR code does a proper
  // lookup on the orginal info state string.
  const auto* ce_state = dynamic_cast<const CEState*>(&state);
  SPIEL_CHECK_TRUE(ce_state != nullptr);

  std::string info_state = state.InformationStateString();
  const size_t idx = info_state.find(config_.recommendation_delimiter);
  SPIEL_CHECK_NE(idx, std::string::npos);

  std::string orig_info_state = info_state.substr(0, idx);
  return ce_state->RecommendedStatePolicy(orig_info_state);
}

}  // namespace algorithms
}  // namespace open_spiel
