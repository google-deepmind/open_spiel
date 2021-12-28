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

#include <memory>
#include <string>
#include <vector>

#include "open_spiel/abseil-cpp/absl/container/flat_hash_map.h"
#include "open_spiel/algorithms/tabular_sarsa.h"
#include "open_spiel/games/tic_tac_toe.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_globals.h"
#include "open_spiel/spiel_utils.h"

using open_spiel::Action;
using open_spiel::Game;
using open_spiel::Player;
using open_spiel::State;

Action GetOptimalAction(
    absl::flat_hash_map<std::pair<std::string, Action>, double> q_values,
    const std::unique_ptr<State>& state) {
  std::vector<Action> legal_actions = state->LegalActions();
  Action optimal_action = open_spiel::kInvalidAction;

  double value = -1;
  for (const Action& action : legal_actions) {
    double q_val = q_values[{state->ToString(), action}];
    if (q_val >= value) {
      value = q_val;
      optimal_action = action;
    }
  }
  return optimal_action;
}

void SolveTicTacToe() {
  std::shared_ptr<const Game> game = open_spiel::LoadGame("tic_tac_toe");
  open_spiel::algorithms::TabularSarsaSolver tabular_sarsa_solver(game);

  int iter = 100000;
  while (iter-- > 0) {
    tabular_sarsa_solver.RunIteration();
  }

  const absl::flat_hash_map<std::pair<std::string, Action>, double>& q_values =
      tabular_sarsa_solver.GetQValueTable();
  std::unique_ptr<State> state = game->NewInitialState();
  while (!state->IsTerminal()) {
    Action optimal_action = GetOptimalAction(q_values, state);
    state->ApplyAction(optimal_action);
  }

  // Tie.
  SPIEL_CHECK_EQ(state->Rewards()[0], 0);
  SPIEL_CHECK_EQ(state->Rewards()[1], 0);
}

void SolveCatch() {
  std::shared_ptr<const Game> game = open_spiel::LoadGame("catch");
  open_spiel::algorithms::TabularSarsaSolver tabular_sarsa_solver(game);

  int training_iter = 100000;
  while (training_iter-- > 0) {
    tabular_sarsa_solver.RunIteration();
  }
  const absl::flat_hash_map<std::pair<std::string, Action>, double>& q_values =
      tabular_sarsa_solver.GetQValueTable();

  int eval_iter = 1000;
  int total_reward = 0;
  while (eval_iter-- > 0) {
    std::unique_ptr<State> state = game->NewInitialState();
    while (!state->IsTerminal()) {
      Action optimal_action = GetOptimalAction(q_values, state);
      state->ApplyAction(optimal_action);
      total_reward += state->Rewards()[0];
    }
  }

  SPIEL_CHECK_GT(total_reward, 0);
}

int main(int argc, char** argv) {
  SolveTicTacToe();
  SolveCatch();
  return 0;
}
