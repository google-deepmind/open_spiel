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

#ifndef OPEN_SPIEL_ALGORITHMS_DETERMINISTIC_POLICY_H_
#define OPEN_SPIEL_ALGORITHMS_DETERMINISTIC_POLICY_H_

#include <stdint.h>

#include <algorithm>
#include <iterator>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

#include "open_spiel/policy.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace algorithms {

// Returns the number of deterministic policies for this player in this game,
// if the number is less than 2^64-1. Otherwise, returns -1.
int64_t NumDeterministicPolicies(const Game& game, Player player);

// An simple container object used to store the legal actions (and chosen
// action) for each information state.
struct LegalsWithIndex {
  LegalsWithIndex() {}
  LegalsWithIndex(const std::vector<Action>& legal_actions)
      : legal_actions_(legal_actions), index(0) {}

  void SetAction(Action action) {
    auto iter = std::find(legal_actions_.begin(), legal_actions_.end(), action);
    SPIEL_CHECK_TRUE(iter != legal_actions_.end());
    index = std::distance(legal_actions_.begin(), iter);
  }

  Action GetAction() const { return legal_actions_[index]; }

  // Try to increment the index of the action. Used by the enumerator over
  // deterministic policies (DeterministicPolicy::NextPolicy) below.
  bool TryIncIndex() {
    if (index + 1 < legal_actions_.size()) {
      index += 1;
      return true;
    } else {
      return false;
    }
  }

  std::vector<Action> legal_actions_;
  int index;
};

class DeterministicTabularPolicy : public Policy {
 public:
  // Creates a deterministic policy and sets it to the specified policy.
  DeterministicTabularPolicy(
      const Game& game, Player player,
      const std::unordered_map<std::string, Action> policy);

  // Creates a default deterministic policy, with all actions set to their first
  // legal action (index 0 in the legal actions list).
  DeterministicTabularPolicy(const Game& game, Player player);

  ActionsAndProbs GetStatePolicy(const std::string& info_state) const override;
  Action GetAction(const std::string& info_state) const;

  // Returns the current deterministic policy as a TabularPolicy.
  TabularPolicy GetTabularPolicy() const;

  // Deterministic policies are ordered. First, we define some order to the
  // information states (which is the order defined by the legal_actions_map
  // for the game). Then the total order over policies is defined in a
  // "counting order according to their associated tuple (
  // legal_action_index[state] for state in ordered_states). The first
  // deterministic policy in the order is the one whose action is set is the
  // first legal action (legal action index = 0). The value of the index can be
  // interpreted as a digit in a mixed base integer, where the value of the
  // integer would represent the position of the deterministic policy in the
  // total order.
  //
  // This function sets this policy to the next deterministic policy in this
  // counting order. The function returns true if this changed the policy (i.e
  // there exists a next policy in the order), otherwise returns false.
  bool NextPolicy();

  // Resets the policy to the first one in the total order defined above: all
  // actions set to their first legal action (index = 0 in the legal actions
  // list).
  void ResetDefaultPolicy();

  // Returns a string representation of the policy, using the specified
  // delimiter to separate information state and action.
  std::string ToString(const std::string& delimiter) const;

 private:
  void CreateTable(const Game& game, Player player);

  std::map<std::string, LegalsWithIndex> table_;
  Player player_;
};

}  // namespace algorithms
}  // namespace open_spiel

#endif  // OPEN_SPIEL_ALGORITHMS_DETERMINISTIC_POLICY_H_
