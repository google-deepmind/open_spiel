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

#include "open_spiel/action_view.h"

#include <vector>

#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

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
    : current_player(current_player), legal_actions(std::move(legal_actions)) {}

ActionView::ActionView(const State& state)
    : ActionView(state.CurrentPlayer(), CollectActions(state)) {}

// FlatJointActions

FlatJointActions ActionView::flat_joint_actions() const {
  int num_flat_actions = 1;
  for (const std::vector<Action>& actions : legal_actions) {
    if (!actions.empty()) num_flat_actions *= actions.size();
  }
  return FlatJointActions{num_flat_actions};
}

FlatJointActionsIterator FlatJointActions::begin() const {
  return FlatJointActionsIterator{0};
}
FlatJointActionsIterator FlatJointActions::end() const {
  return FlatJointActionsIterator{num_flat_joint_actions};
}
FlatJointActionsIterator& FlatJointActionsIterator::operator++() {
  current_action_++;
  return *this;
}
bool FlatJointActionsIterator::operator==(
    FlatJointActionsIterator other) const {
  return current_action_ == other.current_action_;
}
bool FlatJointActionsIterator::operator!=(
    FlatJointActionsIterator other) const {
  return !(*this == other);
}
Action FlatJointActionsIterator::operator*() const { return current_action_; }
FlatJointActionsIterator::FlatJointActionsIterator(int current_action)
    : current_action_(current_action) {}

// FixedActions

FixedActions ActionView::fixed_action(Player player, int action_index) const {
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
  return FixedActions{action_index, num_actions, prod_before, prod_after};
}

FixedActionsIterator FixedActions::begin() const {
  return FixedActionsIterator(fixed_action, num_actions, prod_before,
                              prod_after,
                              /*i=*/0, /*j=*/0);
}
FixedActionsIterator FixedActions::end() const {
  return FixedActionsIterator(fixed_action, num_actions, prod_before,
                              prod_after,
                              /*i=*/prod_after, /*j=*/0);
}

// This essentially imitates a generator that uses a nested for loop:
//
// for i in range(prod_after):
//   for j in range(prod_before):
//     yield prod_before * (fixed_action + i * num_actions) + j
FixedActionsIterator& FixedActionsIterator::operator++() {
  if (j_ + 1 < prod_before_) {
    ++j_;
    return *this;
  } else {
    j_ = 0;
    ++i_;
    SPIEL_CHECK_LE(i_, prod_after_);
    return *this;
  }
}
Action FixedActionsIterator::operator*() const {
  return prod_before_ * (fixed_action_ + i_ * num_actions_) + j_;
}
bool FixedActionsIterator::operator==(const FixedActionsIterator& rhs) const {
  return j_ == rhs.j_ && i_ == rhs.i_ && fixed_action_ == rhs.fixed_action_ &&
         prod_before_ == rhs.prod_before_ && num_actions_ == rhs.num_actions_ &&
         prod_after_ == rhs.prod_after_;
}
bool FixedActionsIterator::operator!=(const FixedActionsIterator& rhs) const {
  return !(rhs == *this);
}
FixedActionsIterator::FixedActionsIterator(int fixed_action, int num_actions,
                                           int prod_before, int prod_after,
                                           int i, int j)
    : fixed_action_(fixed_action),
      num_actions_(num_actions),
      prod_before_(prod_before),
      prod_after_(prod_after),
      i_(i),
      j_(j) {}

}  // namespace open_spiel
