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

#ifndef OPEN_SPIEL_ACTION_VIEW_
#define OPEN_SPIEL_ACTION_VIEW_

#include <vector>

#include "open_spiel/spiel.h"

// ActionView provides a number of iterators that are useful for dealing
// with simultaneous move nodes.

namespace open_spiel {

class FixedActionsIterator {
  const int fixed_action_;
  const int num_actions_;
  const int prod_before_;
  const int prod_after_;
  int i_;  // Outer loop
  int j_;  // Inner loop
 public:
  FixedActionsIterator(int fixed_action, int num_actions, int prod_before,
                       int prod_after, int i, int j);
  FixedActionsIterator& operator++();
  Action operator*() const;
  bool operator==(const FixedActionsIterator& rhs) const;
  bool operator!=(const FixedActionsIterator& rhs) const;
};

struct FixedActions {
  const int fixed_action;
  const int num_actions;
  const int prod_before;
  const int prod_after;
  FixedActionsIterator begin() const;
  FixedActionsIterator end() const;
};

class FlatJointActionsIterator {
  int current_action_;

 public:
  FlatJointActionsIterator(int current_action);
  FlatJointActionsIterator& operator++();
  bool operator==(FlatJointActionsIterator other) const;
  bool operator!=(FlatJointActionsIterator other) const;
  Action operator*() const;
};

struct FlatJointActions {
  const int num_flat_joint_actions;
  FlatJointActionsIterator begin() const;
  FlatJointActionsIterator end() const;
};

// Provides a number of iterators that are useful for dealing
// with simultaneous move nodes.
struct ActionView {
  const Player current_player;
  const std::vector<std::vector<Action>> legal_actions;
  // Collects legal actions at the specified state.
  explicit ActionView(const State& state);
  // Construct a custom action view.
  ActionView(const Player current_player,
             const std::vector<std::vector<Action>> legal_actions);

  int num_players() const { return legal_actions.size(); }
  int num_actions(Player pl) const { return legal_actions.at(pl).size(); }

  // Provides an iterator over all flattened joint actions.
  //
  // It computes the number of possible joint actions = \prod #actions(i)
  // over all the players with any legal actions available.
  // The possible joint actions are just numbered 0, 1, 2, .... and can be
  // decomposed into the individual actions of the players.
  //
  // As this is an iterator, it does not allocate memory for the whole cartesian
  // product of the actions.
  FlatJointActions flat_joint_actions() const;

  // Provides an iterator over flattened actions, while we fix one action
  // for the specified player.
  FixedActions fixed_action(Player player, int action_index) const;
};

}  // namespace open_spiel

#endif  // OPEN_SPIEL_ACTION_VIEW_
