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

#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace {

void TestFixedActions() {
  ActionView view(/*current_player=*/kSimultaneousPlayerId,
                  /*legal_actions=*/{{0, 1}, {2, 3, 4}, {5, 6}});

  std::vector<                   // Player
      std::vector<               // Fixed action
          std::vector<Action>>>  // Expected joint actions.
      expected_joint_actions = {{{0, 2, 4, 6, 8, 10}, {1, 3, 5, 7, 9, 11}},
                                {{0, 1, 6, 7}, {2, 3, 8, 9}, {4, 5, 10, 11}},
                                {{0, 1, 2, 3, 4, 5}, {6, 7, 8, 9, 10, 11}}};

  for (int pl = 0; pl < view.num_players(); ++pl) {
    for (int action_index = 0; action_index < view.num_actions(pl);
         ++action_index) {
      int i = 0;
      for (Action actual_joint_action : view.fixed_action(pl, action_index)) {
        SPIEL_CHECK_EQ(expected_joint_actions[pl][action_index][i++],
                       actual_joint_action);
      }
    }
  }
}

void TestFlatJointActions() {
  ActionView view(/*current_player=*/kSimultaneousPlayerId,
                  /*legal_actions=*/{{0, 1}, {2, 3, 4}, {5, 6}});

  int expected_joint_action = 0;
  for (Action actual_joint_action : view.flat_joint_actions()) {
    SPIEL_CHECK_EQ(expected_joint_action++, actual_joint_action);
  }
  SPIEL_CHECK_EQ(expected_joint_action, 2 * 3 * 2);
}

}  // namespace
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::TestFixedActions();
  open_spiel::TestFlatJointActions();
}
