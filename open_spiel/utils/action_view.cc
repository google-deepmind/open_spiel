// Copyright 2019 DeepMind Technologies Ltd. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <vector>

#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/utils/action_view.h"


namespace open_spiel {

std::vector<std::vector<Action>> CollectActions(const State& state) {
  std::vector<std::vector<Action>> legal_actions;
  if (state.IsSimultaneousNode()) {
    legal_actions = std::vector<std::vector<Action>>(state.NumPlayers());
    for (int i = 0; i < state.NumPlayers(); ++i) {
      legal_actions[i] = state.LegalActions(i);
    }
  } else {
    legal_actions = std::vector<std::vector<Action>>{state.LegalActions()};
  }
  return legal_actions;
}

ActionView::ActionView(const Player current_player,
                       const std::vector<std::vector<Action>> legal_actions)
    : current_player(current_player),
      legal_actions(std::move(legal_actions)) {}

ActionView::ActionView(const State& state)
    : ActionView(state.CurrentPlayer(), CollectActions(state)) {}

// FlatJointActions

ActionView::FlatJointActions ActionView::flat_joint_actions() const {
  int num_flat_actions = 1;
  for (const std::vector<Action>& actions : legal_actions) {
    if (!actions.empty()) num_flat_actions *= actions.size();
  }
  return FlatJointActions{num_flat_actions};
}

ActionView::FlatJointActions ActionView::FlatJointActions::begin() const {
  return *this;
}
ActionView::FlatJointActions ActionView::FlatJointActions::end() const {
  return FlatJointActions{prod, prod};
}
ActionView::FlatJointActions& ActionView::FlatJointActions::operator++() {
  current_action++;
  return *this;
}
bool ActionView::FlatJointActions::operator==(
    ActionView::FlatJointActions other) const {
  return current_action == other.current_action && prod == other.prod;
}
bool ActionView::FlatJointActions::operator!=(
    ActionView::FlatJointActions other) const {
  return !(*this == other);
}
Action ActionView::FlatJointActions::operator*() const {
  return current_action;
}

// FixedActions

ActionView::FixedActions ActionView::fixed_action(
    Player player, int action_index) const {
  SPIEL_CHECK_EQ(current_player, kSimultaneousPlayerId);
  int prod_after = 1;
  for (int pl = player + 1; pl < legal_actions.size(); pl++) {
    const std::vector<Action>& actions = legal_actions[pl];
    if (!actions.empty()) prod_after *= actions.size();
  }
  int prod_before = 1;
  for (int pl = 0; pl < player; pl++) {
    const std::vector<Action>& actions = legal_actions[pl];
    if (!actions.empty()) prod_before *= actions.size();
  }
  int num_actions = legal_actions[player].size();
  return FixedActions{action_index, prod_before, num_actions, prod_after};
}

ActionView::FixedActions ActionView::FixedActions::begin() const {
  return *this;
}
ActionView::FixedActions ActionView::FixedActions::end() const {
  return FixedActions{fixed_action, prod_before,
                      num_actions, prod_after,
                      /*i=*/prod_after, /*j=*/0};
}

// This essentially imitates a generator that uses a nested for loop:
//
// for i in range(prod_after):
//   for j in range(prod_before):
//     yield prod_before * (fixed_action + i * num_actions) + j
ActionView::FixedActions& ActionView::FixedActions::operator++() {
  if (j + 1 < prod_before) {
    ++j;
    return *this;
  } else {
    j = 0;
    i++;
    SPIEL_CHECK_LE(i, prod_after);
    return *this;
  }
}
Action ActionView::FixedActions::operator*() const {
  return prod_before * (fixed_action + i * num_actions) + j;
}
bool ActionView::FixedActions::operator==(
    const ActionView::FixedActions& rhs) const {
  return j == rhs.j
      && i == rhs.i
      && fixed_action == rhs.fixed_action
      && prod_before == rhs.prod_before
      && num_actions == rhs.num_actions
      && prod_after == rhs.prod_after;
}
bool ActionView::FixedActions::operator!=(
    const ActionView::FixedActions& rhs) const { return !(rhs == *this); }

}  // namespace open_spiel
