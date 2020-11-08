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

#ifndef OPEN_SPIEL_UTILS_ACTION_VIEW_
#define OPEN_SPIEL_UTILS_ACTION_VIEW_

#include <vector>

#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"


namespace open_spiel {

// Provides a number of iterators that are useful for dealing 
// with simultaneous move nodes.
struct ActionView {
  const Player current_player;
  const std::vector<std::vector<Action>> legal_actions;
  // Collects legal actions at the specified state.
  ActionView(const State& state);

  ActionView(const Player current_player,
             const std::vector<std::vector<Action>> legal_actions);

  int num_players() const { return legal_actions.size(); }
  int num_actions(Player pl) const { return legal_actions.at(pl).size(); }

  // Provides an iterator over flattened actions where we fix an action
  // for the specified player.
  struct FixedActions {
    int fixed_action;
    int prod_before;
    int num_actions;
    int prod_after;
    int i = 0;  // Outer loop
    int j = 0;  // Inner loop

    FixedActions begin() const;
    FixedActions end() const;
    FixedActions& operator++();
    Action operator*() const;
    bool operator==(const FixedActions& rhs) const;
    bool operator!=(const FixedActions& rhs) const;
  };
  FixedActions fixed_action(Player player, int action_index) const;

  // Provides an iterator over flattened actions. This is equivalent to calling
  // SimMoveState::LegalFlatJointActions() and iterating over the result, but it
  // does not allocate memory for the whole cartesian product of the actions.
  struct FlatJointActions {
    int prod;
    int current_action = 0;
    FlatJointActions begin() const;
    FlatJointActions end() const;
    FlatJointActions& operator++();
    bool operator==(FlatJointActions other) const;
    bool operator!=(FlatJointActions other) const;
    Action operator*() const;
  };
  FlatJointActions flat_joint_actions() const;
};

}  // namespace open_spiel

#endif OPEN_SPIEL_UTILS_ACTION_VIEW_
