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

#include <memory>
#include <string>
#include <vector>

#include "open_spiel/abseil-cpp/absl/container/flat_hash_map.h"
#include "open_spiel/algorithms/sarsa.h"
#include "open_spiel/games/tic_tac_toe.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_globals.h"
#include "open_spiel/spiel_utils.h"

using open_spiel::Action;
using open_spiel::Game;
using open_spiel::Player;
using open_spiel::State;

// Given a player and a state, gets the optimal action from this state
Action GetOptimalAction(
    absl::flat_hash_map<std::pair<std::string, Action>, double> q_values,
    const std::unique_ptr<State>& state) {
  std::vector<Action> legal_actions = state->LegalActions();
  Action optimal_action = open_spiel::kInvalidAction;
  Player player = state->CurrentPlayer();

  // Initialize value to be the minimum utility if current player
  // is the maximizing player (i.e. player 0), and to maximum utility
  // if current player is the minimizing player (i.e. player 1).
  double value = (player == Player{0}) ? -1 : 1;
  for (const Action& action : legal_actions) {
    if (q_values.contains({state->ToString(), action})) {
      double q_val = q_values.at({state->ToString(), action});
      bool is_best_so_far = (player == Player{0} && q_val >= value) ||
                            (player == Player{1} && q_val <= value);
      if (is_best_so_far) {
        value = q_val;
        optimal_action = action;
      }
    }
  }
  return optimal_action;
}

void SolveTicTacToe() {
  std::shared_ptr<const Game> game = open_spiel::LoadGame("tic_tac_toe");
  open_spiel::algorithms::SarsaSolver sarsa_solver(*(game.get()));

  std::unique_ptr<State> state = game->NewInitialState();
  while (!state->IsTerminal()) {
    sarsa_solver.RunIteration();

    const absl::flat_hash_map<std::pair<std::string, Action>, double>&
        q_values = sarsa_solver.GetQValueTable();
    Action optimal_action = GetOptimalAction(q_values, state);
    state->ApplyAction(optimal_action);
  }

  // Tie.
  SPIEL_CHECK_EQ(state->Rewards()[0], 0);
  SPIEL_CHECK_EQ(state->Rewards()[1], 0);
}

int main(int argc, char** argv) {
  SolveTicTacToe();
  return 0;
}
