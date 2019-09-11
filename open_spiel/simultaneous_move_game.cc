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

#include "open_spiel/simultaneous_move_game.h"

namespace open_spiel {

std::vector<Action> SimMoveState::FlatJointActionToActions(
    Action flat_action) const {
  std::vector<Action> actions(num_players_, kInvalidAction);
  for (auto player = Player{0}; player < num_players_; ++player) {
    // For each player with legal actions available:
    const auto legal_actions = LegalActions(player);
    int num_actions = legal_actions.size();
    if (num_actions > 0) {
      // Extract the least-significant digit (radix = the number legal actions
      // for the current player) from flat_action. Use the digit as an index
      // into the player's set of legal actions.
      actions[player] = legal_actions[flat_action % num_actions];
      // Update the flat_action to be for the remaining players only.
      flat_action /= num_actions;
    }
  }
  return actions;
}

void SimMoveState::ApplyFlatJointAction(Action flat_action) {
  ApplyActions(FlatJointActionToActions(flat_action));
}

std::vector<Action> SimMoveState::LegalFlatJointActions() const {
  // Compute the number of possible joint actions = \prod #actions(i)
  // over all players with any legal actions available.
  int number_joint_actions = 1;
  for (auto player = Player{0}; player < num_players_; ++player) {
    int num_actions = LegalActions(player).size();
    if (num_actions > 1) number_joint_actions *= num_actions;
  }
  // The possible joint actions are just numbered 0, 1, 2, ....
  // So build a vector of the right size containing consecutive integers.
  std::vector<Action> joint_actions(number_joint_actions);
  std::iota(joint_actions.begin(), joint_actions.end(), 0);
  return joint_actions;
}

std::string SimMoveState::FlatJointActionToString(Action flat_action) const {
  // Assembles the string for each individual player action into a single
  // string. For example, [Heads, Tails] would mean than player 0 chooses Heads,
  // and player 1 chooses Tails.
  std::string str;
  for (auto player = Player{0}; player < num_players_; ++player) {
    if (!str.empty()) str.append(", ");
    const auto legal_actions = LegalActions(player);
    int num_actions = legal_actions.size();
    str.append(
        ActionToString(player, legal_actions[flat_action % num_actions]));
    flat_action /= num_actions;
  }
  return absl::StrCat("[", str, "]");
}

}  // namespace open_spiel
