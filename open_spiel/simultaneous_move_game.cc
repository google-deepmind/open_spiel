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

#include "open_spiel/simultaneous_move_game.h"

#include <numeric>
#include <string>
#include <vector>

#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/action_view.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {

std::vector<Action> SimMoveState::FlatJointActionToActions(
    Action flat_action) const {
  std::vector<Action> actions(num_players_, kInvalidAction);
  for (Player player = 0; player < num_players_; ++player) {
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
  ActionView view(*this);
  FlatJointActions flat_joint_actions = view.flat_joint_actions();
  std::vector<Action> joint_actions;
  joint_actions.reserve(flat_joint_actions.num_flat_joint_actions);
  for (Action flat_joint_action : flat_joint_actions) {
    joint_actions.push_back(flat_joint_action);
  }
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
