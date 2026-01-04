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

#include "open_spiel/policy.h"

#include <algorithm>
#include <list>
#include <memory>
#include <random>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "open_spiel/abseil-cpp/absl/algorithm/container.h"
#include "open_spiel/abseil-cpp/absl/container/flat_hash_set.h"
#include "open_spiel/abseil-cpp/absl/container/node_hash_map.h"
#include "open_spiel/abseil-cpp/absl/strings/charconv.h"
#include "open_spiel/abseil-cpp/absl/strings/numbers.h"
#include "open_spiel/abseil-cpp/absl/strings/string_view.h"
#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/abseil-cpp/absl/strings/str_format.h"
#include "open_spiel/abseil-cpp/absl/strings/str_split.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_globals.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {

void SetProb(ActionsAndProbs* actions_and_probs, Action action, double prob) {
  for (auto& iter : *actions_and_probs) {
    if (iter.first == action) {
      iter.second = prob;
      return;
    }
  }
  actions_and_probs->push_back({action, prob});
}

double GetProb(const ActionsAndProbs& action_and_probs, Action action) {
  auto it = absl::c_find_if(action_and_probs,
                            [&action](const std::pair<Action, double>& p) {
                              return p.first == action;
                            });
  if (it == action_and_probs.end()) return -1.;
  return it->second;
}

Action GetAction(const ActionsAndProbs& action_and_probs) {
  for (const auto& iter : action_and_probs) {
    if (iter.second == 1.0) {
      return iter.first;
    }
  }
  return kInvalidAction;
}

ActionsAndProbs ToDeterministicPolicy(const ActionsAndProbs& actions_and_probs,
                                      Action action) {
  ActionsAndProbs new_policy;
  new_policy.reserve(actions_and_probs.size());
  for (const auto& iter : actions_and_probs) {
    new_policy.push_back({iter.first, iter.first == action ? 1.0 : 0.0});
  }
  return new_policy;
}

bool StatePoliciesEqual(const ActionsAndProbs& state_policy1,
                        const ActionsAndProbs& state_policy2,
                        double float_tolerance) {
  if (state_policy1.size() != state_policy2.size()) {
    return false;
  }

  for (int i = 0; i < state_policy1.size(); ++i) {
    if (state_policy1[i].first != state_policy2[i].first) {
      return false;
    }

    if (!Near(state_policy1[i].second, state_policy2[i].second,
              float_tolerance)) {
      return false;
    }
  }

  return true;
}

ActionsAndProbs GetDeterministicPolicy(const std::vector<Action>& legal_actions,
                                       Action action) {
  ActionsAndProbs new_policy;
  new_policy.reserve(legal_actions.size());
  for (Action legal_action : legal_actions) {
    new_policy.push_back({legal_action, legal_action == action ? 1.0 : 0.0});
  }
  return new_policy;
}

ActionsAndProbs UniformStatePolicy(const std::vector<Action>& actions) {
  ActionsAndProbs actions_and_probs;
  absl::c_for_each(actions, [&actions_and_probs, &actions](Action a) {
    actions_and_probs.push_back({a, 1. / static_cast<double>(actions.size())});
  });
  return actions_and_probs;
}

ActionsAndProbs UniformStatePolicy(const State& state) {
  return UniformStatePolicy(state.LegalActions());
}

ActionsAndProbs UniformStatePolicy(const State& state, Player player) {
  return UniformStatePolicy(state.LegalActions(player));
}

ActionsAndProbs FirstActionStatePolicy(const State& state) {
  return FirstActionStatePolicy(state, state.CurrentPlayer());
}

ActionsAndProbs FirstActionStatePolicy(const State& state, Player player) {
  ActionsAndProbs actions_and_probs;
  std::vector<Action> legal_actions = state.LegalActions(player);
  actions_and_probs.reserve(legal_actions.size());
  for (int i = 0; i < legal_actions.size(); ++i) {
    actions_and_probs.push_back({legal_actions[i], i == 0 ? 1.0 : 0.0});
  }
  return actions_and_probs;
}

std::unique_ptr<Policy> DeserializePolicy(const std::string& serialized,
                                          std::string delimiter) {
  // Class’s identity is the very first line, see Policy::Serialize
  // for more info.
  std::pair<std::string, absl::string_view> cls_and_content =
      absl::StrSplit(serialized, absl::MaxSplits(':', 1));
  std::string class_identity = cls_and_content.first;

  if (class_identity == "TabularPolicy") {
    return DeserializeTabularPolicy(serialized, delimiter);
  } else if (class_identity == "UniformPolicy") {
    return std::make_unique<UniformPolicy>();
  } else {
    SpielFatalError(absl::StrCat("Deserialization of ", class_identity,
                                 " is not supported."));
  }
}

TabularPolicy::TabularPolicy(const Game& game)
    : TabularPolicy(GetRandomPolicy(game)) {}

std::unique_ptr<TabularPolicy> DeserializeTabularPolicy(
    const std::string& serialized, std::string delimiter) {
  // Class’s identity is the very first line, see Policy::Serialize
  // for more info.
  std::pair<std::string, absl::string_view> cls_and_content =
      absl::StrSplit(serialized, absl::MaxSplits(':', 1));
  SPIEL_CHECK_EQ(cls_and_content.first, "TabularPolicy");

  std::unique_ptr<TabularPolicy> res = std::make_unique<TabularPolicy>();
  if (cls_and_content.second.empty()) return res;

  std::vector<absl::string_view> splits =
      absl::StrSplit(cls_and_content.second, delimiter);

  // Insert the actual values
  Action action;
  double prob;
  for (int i = 0; i < splits.size(); i += 2) {
    std::vector<absl::string_view> policy_values =
        absl::StrSplit(splits.at(i + 1), ',');
    ActionsAndProbs res_policy;
    res_policy.reserve(policy_values.size());

    for (absl::string_view policy_value : policy_values) {
      std::pair<absl::string_view, absl::string_view> action_and_prob =
          absl::StrSplit(policy_value, '=');
      SPIEL_CHECK_TRUE(absl::SimpleAtoi(action_and_prob.first, &action));
      absl::from_chars(
          action_and_prob.second.data(),
          action_and_prob.second.data() + action_and_prob.second.size(), prob);
      res_policy.push_back({action, prob});
    }
    res->SetStatePolicy(std::string(splits.at(i)), res_policy);
  }
  return res;
}

std::string TabularPolicy::ToString() const {
  std::string str = "";
  for (const auto& infostate_and_policy : policy_table_) {
    absl::StrAppend(&str, infostate_and_policy.first, ": ");
    for (const auto& policy : infostate_and_policy.second) {
      absl::StrAppend(&str, " ", policy.first, "=", policy.second);
    }
    absl::StrAppend(&str, "\n");
  }
  return str;
}

std::string TabularPolicy::ToStringSorted() const {
  std::vector<std::string> keys;
  keys.reserve(policy_table_.size());

  for (const auto& infostate_and_policy : policy_table_) {
    keys.push_back(infostate_and_policy.first);
  }

  std::sort(keys.begin(), keys.end());
  std::string str = "";
  for (const std::string& key : keys) {
    absl::StrAppend(&str, key, ": ");
    for (const auto& policy : policy_table_.at(key)) {
      absl::StrAppend(&str, " ", policy.first, "=", policy.second);
    }
    absl::StrAppend(&str, "\n");
  }

  return str;
}

PartialTabularPolicy::PartialTabularPolicy()
    : TabularPolicy(),
      fallback_policy_(std::make_shared<UniformPolicy>()) {}

PartialTabularPolicy::PartialTabularPolicy(
      const std::unordered_map<std::string, ActionsAndProbs>& table)
    : TabularPolicy(table),
      fallback_policy_(std::make_shared<UniformPolicy>()) {}

PartialTabularPolicy::PartialTabularPolicy(
      const std::unordered_map<std::string, ActionsAndProbs>& table,
      std::shared_ptr<Policy> fallback_policy)
    : TabularPolicy(table),
      fallback_policy_(fallback_policy) {}

ActionsAndProbs PartialTabularPolicy::GetStatePolicy(const State& state) const {
  auto iter = policy_table_.find(state.InformationStateString());
  if (iter == policy_table_.end()) {
    return fallback_policy_->GetStatePolicy(state);
  } else {
    return iter->second;
  }
}

ActionsAndProbs PartialTabularPolicy::GetStatePolicy(const State& state,
                                                     Player player) const {
  auto iter = policy_table_.find(state.InformationStateString(player));
  if (iter == policy_table_.end()) {
    return fallback_policy_->GetStatePolicy(state);
  } else {
    return iter->second;
  }
}

ActionsAndProbs PartialTabularPolicy::GetStatePolicy(
    const std::string& info_state) const {
  auto iter = policy_table_.find(info_state);
  if (iter == policy_table_.end()) {
    return fallback_policy_->GetStatePolicy(info_state);
  } else {
    return iter->second;
  }
}

TabularPolicy GetEmptyTabularPolicy(const Game& game,
                                    bool initialize_to_uniform,
                                    Player player) {
  std::unordered_map<std::string, ActionsAndProbs> policy;
  if (game.GetType().dynamics != GameType::Dynamics::kSequential) {
    SpielFatalError("Game is not sequential.");
    return TabularPolicy(policy);
  }
  std::list<std::unique_ptr<State>> to_visit;
  to_visit.push_back(game.NewInitialState());
  while (!to_visit.empty()) {
    std::unique_ptr<State> state = std::move(to_visit.back());
    to_visit.pop_back();
    if (state->IsTerminal()) {
      continue;
    }
    if (state->IsChanceNode()) {
      for (const auto& outcome_and_prob : state->ChanceOutcomes()) {
        to_visit.emplace_back(state->Child(outcome_and_prob.first));
      }
    } else {
      ActionsAndProbs infostate_policy;
      std::vector<Action> legal_actions = state->LegalActions();
      const int num_legal_actions = legal_actions.size();
      SPIEL_CHECK_GT(num_legal_actions, 0.);
      for (Action action : legal_actions) {
        to_visit.push_back(state->Child(action));
      }
      if (player < 0 || state->IsPlayerActing(player)) {
        double action_probability = 1.;
        if (initialize_to_uniform) {
          action_probability = 1. / num_legal_actions;
        }
        ActionsAndProbs infostate_policy;
        infostate_policy.reserve(num_legal_actions);
        for (Action action : legal_actions) {
          infostate_policy.push_back({action, action_probability});
        }
        if (infostate_policy.empty()) {
          SpielFatalError("State has zero legal actions.");
        }
        policy.insert({state->InformationStateString(), infostate_policy});
      }
    }
  }
  return TabularPolicy(policy);
}

TabularPolicy GetUniformPolicy(const Game& game) {
  return GetEmptyTabularPolicy(game, /*initialize_to_uniform=*/true);
}

template <typename RandomNumberDistribution>
TabularPolicy SamplePolicy(
  const Game& game, int seed, RandomNumberDistribution& dist, Player player) {
  std::mt19937 gen(seed);
  TabularPolicy policy = GetEmptyTabularPolicy(game, false, player);
  std::unordered_map<std::string, ActionsAndProbs>& policy_table =
      policy.PolicyTable();
  for (auto& kv : policy_table) {
    ActionsAndProbs state_policy;
    if (kv.second.empty()) {
      SpielFatalError("State has zero legal actions.");
    }
    state_policy.reserve(kv.second.size());
    double sum = 0;
    double prob;
    for (const auto& action_and_prob : kv.second) {
      // We multiply the original probability by a random number greater than
      // 0. We then normalize. This has the effect of randomly permuting the
      // policy but all illegal actions still have zero probability.
      prob = dist(gen) * action_and_prob.second;
      sum += prob;
      state_policy.push_back({action_and_prob.first, prob});
    }
    // We normalize the policy to ensure it sums to 1.
    for (auto& action_and_prob : state_policy) {
      action_and_prob.second /= sum;
    }
    // This is included as a sanity check.
    double normalized_sum = 0;
    for (auto& action_and_prob : state_policy) {
      normalized_sum += action_and_prob.second;
    }
    SPIEL_CHECK_FLOAT_EQ(normalized_sum, 1.0);
    kv.second = state_policy;
  }
  return policy;
}

TabularPolicy GetRandomPolicy(const Game& game, int seed, Player player) {
  std::uniform_real_distribution<double> dist(0, 1);
  return SamplePolicy(game, seed, dist, player);
}

TabularPolicy GetFlatDirichletPolicy(
    const Game& game, int seed, Player player) {
  std::gamma_distribution<double> dist(1.0, 1.0);
  return SamplePolicy(game, seed, dist, player);
}

TabularPolicy GetRandomDeterministicPolicy(
  const Game& game, int seed, Player player) {
  std::mt19937 gen(seed);
  absl::node_hash_map<int, std::uniform_int_distribution<int>> dists;
  TabularPolicy policy = GetEmptyTabularPolicy(game, false, player);
  std::unordered_map<std::string, ActionsAndProbs>& policy_table =
      policy.PolicyTable();
  for (auto& kv : policy_table) {
    ActionsAndProbs state_policy;

    // Need to calculate how many legal actions there are. Illegal actions
    // can appear in kv.
    int num_legal_actions = 0;
    for (const auto& action_and_prob : kv.second) {
      if (action_and_prob.second > 0) {
        num_legal_actions += 1;
      }
    }
    if (num_legal_actions == 0) {
      SpielFatalError("State has zero legal actions.");
    }
    state_policy.reserve(num_legal_actions);

    // The distribution functions have are calculated over a fixed domain. If
    // the number of legal a ctions has not been encountered before, we need to
    // create a new distribution function.
    if (dists.count(num_legal_actions) == 0) {
      std::uniform_int_distribution<int> dist(0, num_legal_actions - 1);
      dists.insert({num_legal_actions, std::move(dist)});
    }

    const int action = dists[num_legal_actions](gen);
    int legal_action_index = 0;
    double prob = 0.0;
    for (const auto& action_and_prob : kv.second) {
      prob = 0.0;
      if (action_and_prob.second > 0) {
        if (legal_action_index == action) {
          prob = 1.0;
        }
        legal_action_index += 1;
      }
      state_policy.push_back({action_and_prob.first, prob});
    }

    // This is included as a sanity check.
    double normalized_sum = 0;
    for (auto& action_and_prob : state_policy) {
      normalized_sum += action_and_prob.second;
    }
    SPIEL_CHECK_FLOAT_EQ(normalized_sum, 1.0);
    kv.second = state_policy;
  }
  return policy;
}

TabularPolicy GetRandomDeterministicVisitPolicy(
    const Game& game, int seed, Player player) {
  std::mt19937 gen(seed);
  absl::node_hash_map<int, std::uniform_int_distribution<int>> dists;
  std::unordered_map<std::string, ActionsAndProbs> policy;
  if (game.GetType().dynamics != GameType::Dynamics::kSequential) {
    SpielFatalError("Game is not sequential.");
    return TabularPolicy(policy);
  }
  const GameType::Information information = game.GetType().information;
  std::list<std::unique_ptr<State>> to_visit;
  to_visit.push_back(game.NewInitialState());
  while (!to_visit.empty()) {
    std::unique_ptr<State> state = std::move(to_visit.back());
    to_visit.pop_back();
    if (state->IsTerminal()) {
      continue;
    } else if (state->IsChanceNode()) {
      for (const auto& outcome_and_prob : state->ChanceOutcomes()) {
        to_visit.emplace_back(state->Child(outcome_and_prob.first));
      }
    } else if (player < 0 || state->IsPlayerActing(player)) {
      std::vector<Action> legal_actions = state->LegalActions();
      const int num_legal_actions = legal_actions.size();
      SPIEL_CHECK_GT(num_legal_actions, 0.);
      if (dists.count(num_legal_actions) == 0) {
        std::uniform_int_distribution<int> dist(0, num_legal_actions - 1);
        dists.insert({num_legal_actions, std::move(dist)});
      }
      const int legal_action_index = dists[num_legal_actions](gen);
      SPIEL_CHECK_GE(legal_action_index, 0);
      SPIEL_CHECK_LT(legal_action_index, num_legal_actions);
      const int action = legal_actions[legal_action_index];
      ActionsAndProbs infostate_policy;
      infostate_policy.reserve(1);
      infostate_policy.push_back({action, 1.0});
      policy.insert({state->InformationStateString(), infostate_policy});
      if (information == GameType::Information::kPerfectInformation) {
        to_visit.push_back(state->Child(action));
      } else {
        for (Action action : legal_actions) {
          to_visit.push_back(state->Child(action));
        }
      }
    } else {
      std::vector<Action> legal_actions = state->LegalActions();
      const int num_legal_actions = legal_actions.size();
      SPIEL_CHECK_GT(num_legal_actions, 0.);
      for (Action action : legal_actions) {
        to_visit.push_back(state->Child(action));
      }
    }
  }
  return TabularPolicy(policy);
}

TabularPolicy GetFirstActionPolicy(const Game& game) {
  std::unordered_map<std::string, ActionsAndProbs> policy;
  if (game.GetType().dynamics != GameType::Dynamics::kSequential) {
    SpielFatalError("Game is not sequential.");
    return TabularPolicy(policy);
  }
  std::vector<std::unique_ptr<State>> to_visit;
  to_visit.push_back(game.NewInitialState());
  while (!to_visit.empty()) {
    std::unique_ptr<State> state = std::move(to_visit.back());
    to_visit.pop_back();
    if (state->IsTerminal()) {
      continue;
    }
    if (state->IsChanceNode()) {
      for (const auto& outcome_and_prob : state->ChanceOutcomes()) {
        to_visit.emplace_back(state->Child(outcome_and_prob.first));
      }
    } else {
      ActionsAndProbs infostate_policy;
      std::vector<Action> legal_actions = state->LegalActions();
      const int num_legal_actions = legal_actions.size();
      SPIEL_CHECK_GT(num_legal_actions, 0.);
      bool first_legal_action_found = false;

      infostate_policy.reserve(num_legal_actions);
      for (Action action : legal_actions) {
        to_visit.push_back(state->Child(action));
        if (!first_legal_action_found) {
          first_legal_action_found = true;
          infostate_policy.push_back({action, 1.});

        } else {
          infostate_policy.push_back({action, 0.});
        }
      }
      if (infostate_policy.empty()) {
        SpielFatalError("State has zero legal actions.");
      }
      policy[state->InformationStateString()] = std::move(infostate_policy);
    }
  }
  return TabularPolicy(policy);
}

ActionsAndProbs PreferredActionPolicy::GetStatePolicy(const State& state,
                                                      Player player) const {
  std::vector<Action> legal_actions = state.LegalActions(player);
  for (Action action : preference_order_) {
    if (absl::c_find(legal_actions, action) != legal_actions.end()) {
      return GetDeterministicPolicy(legal_actions, action);
    }
  }
  SpielFatalError("No preferred action found in the legal actions!");
}

TabularPolicy ToTabularPolicy(const Game& game, const Policy* policy) {
  TabularPolicy tabular_policy;
  std::vector<std::unique_ptr<State>> to_visit;
  to_visit.push_back(game.NewInitialState());
  for (int idx = 0; idx < to_visit.size(); ++idx) {
    const State* state = to_visit[idx].get();
    if (state->IsTerminal()) {
      continue;
    }

    if (!state->IsChanceNode()) {
      std::vector<Player> players(game.NumPlayers());
      if (state->IsSimultaneousNode()) {
        absl::c_iota(players, 0);
      } else {
        players = {state->CurrentPlayer()};
      }

      for (Player player : players) {
        ActionsAndProbs state_policy = policy->GetStatePolicy(*state);
        tabular_policy.SetStatePolicy(state->InformationStateString(player),
                                      state_policy);
      }
    }

    for (Action action : state->LegalActions()) {
      to_visit.push_back(state->Child(action));
    }
  }
  return tabular_policy;
}

TabularPolicy GetPrefActionPolicy(const Game& game,
                                  const std::vector<Action>& pref_actions) {
  PreferredActionPolicy policy(pref_actions);
  return ToTabularPolicy(game, &policy);
}

std::string PrintPolicy(const ActionsAndProbs& policy) {
  std::string policy_string;
  for (auto [a, p] : policy) {
    absl::StrAppend(&policy_string, absl::StrFormat("(%i, %f), ", a, p));
  }
  return policy_string;
}

TabularPolicy ToJointTabularPolicy(const std::vector<TabularPolicy>& policies,
                                   bool check_no_overlap) {
  TabularPolicy joint_policy;
  for (const TabularPolicy& policy : policies) {
    if (check_no_overlap) {
      for (const auto& key_and_val : policy.PolicyTable()) {
        SPIEL_CHECK_TRUE(joint_policy.PolicyTable().find(key_and_val.first) ==
                         joint_policy.PolicyTable().end());
      }
    }
    joint_policy.ImportPolicy(policy);
  }
  return joint_policy;
}

}  // namespace open_spiel
