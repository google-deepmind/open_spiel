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

#include <string>

#include "open_spiel/algorithms/value_iteration.h"
#include "open_spiel/games/tic_tac_toe.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

namespace ttt = open_spiel::tic_tac_toe;

// Example code for using value iteration algorithm to solve tic-tac-toe.
int main(int argc, char** argv) {
  ttt::TicTacToeGame game({});

  auto solution = open_spiel::algorithms::ValueIteration(game, -1, 0.01);
  for (auto kv : solution) {
    std::cerr << "State: " << std::endl
              << kv.first << std::endl
              << "Value: " << kv.second << std::endl;
  }

  std::string initial_state = "...\n...\n...";
  std::string cross_win_state = "...\n...\n.ox";
  std::string naught_win_state = "x..\noo.\nxx.";
  SPIEL_CHECK_EQ(solution[initial_state], 0);
  SPIEL_CHECK_EQ(solution[cross_win_state], 1);
  SPIEL_CHECK_EQ(solution[naught_win_state], -1);
}
