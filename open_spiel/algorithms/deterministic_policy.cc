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

#include "open_spiel/algorithms/deterministic_policy.h"

#include <climits>

#include "open_spiel/algorithms/get_legal_actions_map.h"
#include "open_spiel/policy.h"

namespace open_spiel {
namespace algorithms {

int64_t NumDeterministicPolicies(const Game& game, Player player) {
  int64_t num_policies = 1;
  std::unordered_map<std::string, std::vector<Action>> legal_actions_map =
      GetLegalActionsMap(game, -1, player);
  for (const auto& infostate_str_actions : legal_actions_map) {
    int64_t num_actions = infostate_str_actions.second.size();
    SPIEL_CHECK_GT(num_actions, 0);

    // Check for integer overflow.
    if (num_policies > INT64_MAX / num_actions) {
      return -1;
    }

    num_policies *= num_actions;
  }
  return num_policies;
}

DeterministicTabularPolicy::DeterministicTabularPolicy(
    const Game& game, Player player,
    const std::unordered_map<std::string, Action> policy)
    : table_(), player_(player) {
  CreateTable(game, player);
  for (const auto& info_state_action : policy) {
    auto iter = table_.find(info_state_action.first);
    SPIEL_CHECK_TRUE(iter != table_.end());
    iter->second.SetAction(info_state_action.second);
  }
}

DeterministicTabularPolicy::DeterministicTabularPolicy(const Game& game,
                                                       Player player)
    : table_(), player_(player) {
  CreateTable(game, player);
}

ActionsAndProbs DeterministicTabularPolicy::GetStatePolicy(
    const std::string& info_state) const {
  auto iter = table_.find(info_state);
  SPIEL_CHECK_TRUE(iter != table_.end());
  ActionsAndProbs state_policy;
  Action policy_action = iter->second.GetAction();
  for (const auto& action : iter->second.legal_actions_) {
    state_policy.push_back({action, action == policy_action ? 1.0 : 0.0});
  }
  return state_policy;
}

Action DeterministicTabularPolicy::GetAction(
    const std::string& info_state) const {
  auto iter = table_.find(info_state);
  SPIEL_CHECK_TRUE(iter != table_.end());
  return iter->second.GetAction();
}

TabularPolicy DeterministicTabularPolicy::GetTabularPolicy() const {
  TabularPolicy tabular_policy;
  for (const auto& infostate_and_legals : table_) {
    ActionsAndProbs state_policy;
    Action policy_action = infostate_and_legals.second.GetAction();
    for (const auto& action : infostate_and_legals.second.legal_actions_) {
      state_policy.push_back({action, action == policy_action ? 1.0 : 0.0});
    }
    tabular_policy.SetStatePolicy(infostate_and_legals.first, state_policy);
  }
  return tabular_policy;
}

bool DeterministicTabularPolicy::NextPolicy() {
  // Treat the current indices as digits in a mixed base. Starting at the
  // beginning of the table, add 1. If can't, continue trying. If we reach the
  // end without being able to add 1, then this is the end of the order.
  // Otherwise, increment the digit we land on by 1, and reset all the ones
  // we skipped over earlier in the order.
  for (auto iter = table_.begin(); iter != table_.end(); ++iter) {
    if (iter->second.TryIncIndex()) {
      for (auto iter2 = table_.begin(); iter2 != iter; ++iter2) {
        iter2->second.index = 0;
      }
      return true;
    }
  }
  return false;
}

void DeterministicTabularPolicy::ResetDefaultPolicy() {
  for (auto& info_state_entry : table_) {
    info_state_entry.second.index = 0;
  }
}

void DeterministicTabularPolicy::CreateTable(const Game& game, Player player) {
  std::unordered_map<std::string, std::vector<Action>> legal_actions_map =
      GetLegalActionsMap(game, -1, player);
  for (const auto& info_state_actions : legal_actions_map) {
    table_[info_state_actions.first] =
        LegalsWithIndex(info_state_actions.second);
  }
}

std::string DeterministicTabularPolicy::ToString(
    const std::string& delimiter) const {
  std::string str = "";
  for (const auto& info_state_entry : table_) {
    absl::StrAppend(&str, info_state_entry.first, " ", delimiter, " ",
                    "action = ", info_state_entry.second.GetAction(), "\n");
  }
  return str;
}

}  // namespace algorithms
}  // namespace open_spiel
