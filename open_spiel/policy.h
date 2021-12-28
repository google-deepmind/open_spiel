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

#ifndef OPEN_SPIEL_POLICY_H_
#define OPEN_SPIEL_POLICY_H_

#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "open_spiel/abseil-cpp/absl/algorithm/container.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/utils/serialization.h"

namespace open_spiel {

// Returns the probability for the specified action, or -1 if not found.
double GetProb(const ActionsAndProbs& action_and_probs, Action action);

// Set an action probability for the specified action.
void SetProb(ActionsAndProbs* actions_and_probs, Action action, double prob);

// Helper for deterministic policies: returns the single action if the policy
// is deterministic, otherwise returns kInvalidAction.
Action GetAction(const ActionsAndProbs& action_and_probs);

// Returns a policy where every legal action has probability 1 / (number of
// legal actions) for the current player to play. The overloaded function is
// similar, and provided to support simultaneous move games.
ActionsAndProbs UniformStatePolicy(const State& state);
ActionsAndProbs UniformStatePolicy(const State& state, Player player);

// Returns a policy where the zeroth action has probability 1. The overloaded
// function is similar, and provided to support simultaneous move games.
ActionsAndProbs FirstActionStatePolicy(const State& state);
ActionsAndProbs FirstActionStatePolicy(const State& state, Player player);

// Return a new policy with all the same actions, but with probability 1 on the
// specified action, and 0 on the others.
ActionsAndProbs ToDeterministicPolicy(const ActionsAndProbs& actions_and_probs,
                                      Action action);

// Returns a policy with probability 1 on a specific action, and 0 on others.
ActionsAndProbs GetDeterministicPolicy(const std::vector<Action>& legal_actions,
                                       Action action);

// A general policy object. A policy is a mapping from states to list of
// (action, prob) pairs for all the legal actions at the state.
class Policy {
 public:
  virtual ~Policy() = default;

  // A convenience method for callers that want to use arrays.
  virtual std::pair<std::vector<Action>, std::vector<double>>
  GetStatePolicyAsParallelVectors(const State& state) const {
    std::pair<std::vector<Action>, std::vector<double>> parray;
    for (const auto& action_and_prob : GetStatePolicy(state)) {
      parray.first.push_back(action_and_prob.first);
      parray.second.push_back(action_and_prob.second);
    }
    return parray;
  }

  // A convenience method for callers that want to use arrays.
  virtual std::pair<std::vector<Action>, std::vector<double>>
  GetStatePolicyAsParallelVectors(const std::string info_state) const {
    std::pair<std::vector<Action>, std::vector<double>> parray;
    for (const auto& action_and_prob : GetStatePolicy(info_state)) {
      parray.first.push_back(action_and_prob.first);
      parray.second.push_back(action_and_prob.second);
    }
    return parray;
  }

  virtual std::unordered_map<Action, double> GetStatePolicyAsMap(
      const State& state) const {
    std::unordered_map<Action, double> pmap;
    for (const auto& action_and_prob : GetStatePolicy(state)) {
      pmap[action_and_prob.first] = action_and_prob.second;
    }
    return pmap;
  }

  virtual std::unordered_map<Action, double> GetStatePolicyAsMap(
      const std::string& info_state) const {
    std::unordered_map<Action, double> pmap;
    for (const auto& action_and_prob : GetStatePolicy(info_state)) {
      pmap[action_and_prob.first] = action_and_prob.second;
    }
    return pmap;
  }

  // Returns a list of (action, prob) pairs for the policy for the current
  // player at this state. If the policy is not available at the state, returns
  // an empty list.
  virtual ActionsAndProbs GetStatePolicy(const State& state) const {
    return GetStatePolicy(state, state.CurrentPlayer());
  }

  // Returns a list of (action, prob) pairs for the policy for the specified
  // player at this state. If the policy is not available at the state, returns
  // an empty list.
  virtual ActionsAndProbs GetStatePolicy(
      const State& state, Player player) const {
    return GetStatePolicy(state.InformationStateString(player));
  }

  // Returns a list of (action, prob) pairs for the policy at this info state.
  // If the policy is not available at the state, returns and empty list.
  // It is sufficient for subclasses to override only this method, but not all
  // forms of policies will be able to do so from just the information state.
  virtual ActionsAndProbs GetStatePolicy(const std::string& info_state) const {
    SpielFatalError("GetStatePolicy(const std::string&) unimplemented.");
  }

  // Each override must write out the class’s identity followed by ":" as the
  // very first thing so that the DeserializePolicy method can then call the
  // Deserialize method for the correct subclass. See TabularPolicy and
  // DeserializePolicy below for an example. The double_precision parameter
  // indicates the number of decimal places in floating point numbers
  // formatting, value -1 formats doubles with lossless, non-portable bitwise
  // representation hex strings.
  virtual std::string Serialize(int double_precision = -1,
                                std::string delimiter = "<~>") const {
    SpielFatalError("Serialize(std::string delimiter) unimplemented.");
  }
};

std::unique_ptr<Policy> DeserializePolicy(const std::string& serialized,
                                          std::string delimiter = "<~>");

// A tabular policy represented internally as a map. Note that this
// implementation is not directly compatible with the Python TabularPolicy
// implementation; the latter is implemented as a table of size
// [num_states, num_actions], while this is implemented as a map. It is
// non-trivial to convert between the two, but we have a function that does so
// in the open_spiel/python/policy.py file.
class TabularPolicy : public Policy {
 public:
  TabularPolicy() = default;
  TabularPolicy(const Game& game);  // Construct a uniform random policy.
  TabularPolicy(const TabularPolicy& other) = default;
  TabularPolicy(const std::unordered_map<std::string, ActionsAndProbs>& table)
      : policy_table_(table) {}

  // Converts a policy to a TabularPolicy.
  TabularPolicy(const Game& game, const Policy& policy) : TabularPolicy(game) {
    for (auto& [infostate, is_policy] : policy_table_) {
      is_policy = policy.GetStatePolicy(infostate);
    }
  }

  // Creates a new TabularPolicy from a deterministic policy encoded as a
  // {info_state_str -> action} dict. The dummy_policy is used to initialize
  // the initial mapping.
  TabularPolicy(const TabularPolicy& dummy_policy,
                const std::unordered_map<std::string, Action>& action_map)
      : policy_table_(dummy_policy.policy_table_) {
    for (const auto& entry : action_map) {
      std::string info_state = entry.first;
      Action action_taken = action_map.at(entry.first);
      for (auto& action_and_prob : policy_table_[info_state]) {
        action_and_prob.second =
            (action_and_prob.first == action_taken ? 1.0 : 0.0);
      }
    }
  }

  ActionsAndProbs GetStatePolicy(const std::string& info_state) const override {
    auto iter = policy_table_.find(info_state);
    if (iter == policy_table_.end()) {
      return {};
    } else {
      return iter->second;
    }
  }

  std::string Serialize(int double_precision = -1,
                        std::string delimiter = "<~>") const override {
    SPIEL_CHECK_GE(double_precision, -1);
    if (delimiter == "," || delimiter == "=") {
      // The two delimiters are used for de/serialization of policy_table_
      SpielFatalError(
          "Please select a different delimiter,"
          "invalid values are \",\" and \"=\".");
    }
    std::string str = "TabularPolicy:";
    if (policy_table_.empty()) return str;

    for (auto const& [info_state, policy] : policy_table_) {
      if (info_state.find(delimiter) != std::string::npos) {
        SpielFatalError(absl::StrCat(
            "Info state contains delimiter \"", delimiter,
            "\", please fix the info state or select a different delimiter."));
      }

      std::string policy_str;
      if (double_precision == -1) {
        policy_str =
            absl::StrJoin(policy, ",",
                          absl::PairFormatter(absl::AlphaNumFormatter(), "=",
                                              HexDoubleFormatter()));
      } else {
        policy_str = absl::StrJoin(
            policy, ",",
            absl::PairFormatter(absl::AlphaNumFormatter(), "=",
                                SimpleDoubleFormatter(double_precision)));
      }
      absl::StrAppend(&str, info_state, delimiter, policy_str, delimiter);
    }
    // Remove the trailing delimiter
    str.erase(str.length() - delimiter.length());
    return str;
  }

  // Set (overwrite) all the state policies contained in another policy within
  // this policy. Does not change other state policies not contained in this
  // policy.
  void ImportPolicy(const TabularPolicy& other_policy) {
    for (const auto& [info_state, actions_and_probs] :
         other_policy.policy_table_) {
      SetStatePolicy(info_state, actions_and_probs);
    }
  }

  // Set the probability for action at the info state. If the info state is not
  // in the policy, it is added. If the action is not in the info state policy,
  // it is added. Otherwise it is modified.
  void SetProb(const std::string& info_state, Action action, double prob) {
    auto iter = policy_table_.find(info_state);
    if (iter == policy_table_.end()) {
      auto iter_and_bool = policy_table_.insert({info_state, {}});
      iter = iter_and_bool.first;
    }
    open_spiel::SetProb(&(iter->second), action, prob);
  }

  void SetStatePolicy(const std::string& info_state,
                      const ActionsAndProbs& state_policy) {
    policy_table_[info_state] = state_policy;
  }

  std::unordered_map<std::string, ActionsAndProbs>& PolicyTable() {
    return policy_table_;
  }

  const std::unordered_map<std::string, ActionsAndProbs>& PolicyTable() const {
    return policy_table_;
  }

  const std::string ToString() const;

  // A ToString where the keys are sorted.
  const std::string ToStringSorted() const;

 private:
  std::unordered_map<std::string, ActionsAndProbs> policy_table_;
};

std::unique_ptr<TabularPolicy> DeserializeTabularPolicy(
    const std::string& serialized, std::string delimiter = "<~>");

// Chooses all legal actions with equal probability. This is equivalent to the
// tabular version, except that this works for large games.
class UniformPolicy : public Policy {
 public:
  ActionsAndProbs GetStatePolicy(
      const State& state, Player player) const override {
    if (state.IsSimultaneousNode()) {
      return UniformStatePolicy(state, player);
    } else {
      SPIEL_CHECK_TRUE(state.IsPlayerActing(player));
      return UniformStatePolicy(state);
    }
  }

  std::string Serialize(int double_precision = -1,
                        std::string delimiter = "") const override {
    return "UniformPolicy:";
  }
};

// Chooses all legal actions with equal probability. This is equivalent to the
// tabular version, except that this works for large games.
class FirstActionPolicy : public Policy {
 public:
  ActionsAndProbs GetStatePolicy(const State& state,
                                 Player player) const override {
    if (state.IsSimultaneousNode()) {
      return FirstActionStatePolicy(state, player);
    } else {
      SPIEL_CHECK_TRUE(state.IsPlayerActing(player));
      return FirstActionStatePolicy(state);
    }
  }

  std::string Serialize(int double_precision = -1,
                        std::string delimiter = "") const override {
    return "FirstActionPolicy:";
  }
};

// A deterministic policy with which takes legal actions in order of
// preference specified by pref_actions. The function will check-fail if none
// of the pref_action elements are legal for a state.
//
// For example, PreferredActionPolicy(leduc, {kRaise, kCall}) constructs a
// policy that always raises and only falls back to call if raise is not a legal
// action. If it is possible for nethier raise nor call to be valid actions in a
// state in leduc, the function will fail.
class PreferredActionPolicy : public Policy {
 public:
  PreferredActionPolicy(const std::vector<Action>& preference_order)
      : preference_order_(preference_order) {}

  ActionsAndProbs GetStatePolicy(const State& state,
                                 Player player) const override;

  std::string Serialize(int double_precision = -1,
                        std::string delimiter = "") const override {
    SpielFatalError("Unimplemented.");
  }

 private:
  std::vector<Action> preference_order_;
};

// Takes any policy and returns a tabular policy by traversing the game and
// building a tabular policy for it.
TabularPolicy ToTabularPolicy(const Game& game, const Policy* policy);

// Helper functions that generate policies for testing.
TabularPolicy GetEmptyTabularPolicy(const Game& game,
                                    bool initialize_to_uniform = false);
TabularPolicy GetUniformPolicy(const Game& game);
TabularPolicy GetRandomPolicy(const Game& game, int seed = 0);
TabularPolicy GetFirstActionPolicy(const Game& game);

// Returns a preferred action policy as a tabular policy.
TabularPolicy GetPrefActionPolicy(
    const Game& game, const std::vector<Action>& pref_action);

std::string PrintPolicy(const ActionsAndProbs& policy);

// Takes many tabular policy and merges them into one. If check_no_overlap is
// set, then a check is done to ensure that there is no intersection among the
// policies (slow: involves iterating over each).
TabularPolicy ToJointTabularPolicy(const std::vector<TabularPolicy>& policies,
                                   bool check_no_overlap);

}  // namespace open_spiel

#endif  // OPEN_SPIEL_POLICY_H_
