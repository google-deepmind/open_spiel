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
#include "open_spiel/games/tic_tac_toe/tic_tac_toe.h"
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
  const auto state_str = state->ToString();

  Action optimal_action = open_spiel::kInvalidAction;
  double value = -1;
  for (const Action& action : legal_actions) {
    double q_val = q_values[{state_str, action}];
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

void SolveTicTacToeEligibilityTraces() {
  std::shared_ptr<const Game> game = open_spiel::LoadGame("tic_tac_toe");
  open_spiel::algorithms::TabularSarsaSolver tabular_sarsa_solver_lambda00(
      /*game=*/game,
      /*depth_limit=*/-1.0,
      /*epsilon=*/0.1,
      /*learning_rate=*/0.01,
      /*discount_factor=*/0.99,
      /*lambda=*/0.0);
  open_spiel::algorithms::TabularSarsaSolver tabular_sarsa_solver_lambda03(
      /*game=*/game,
      /*depth_limit=*/-1.0,
      /*epsilon=*/0.1,
      /*learning_rate=*/0.01,
      /*discount_factor=*/0.99,
      /*lambda=*/0.3);

  const int runs = 1000;
  int count_tie_games_lambda00 = 0;
  int count_tie_games_lambda03 = 0;
  for (int i = 0; i < runs; ++i) {
    tabular_sarsa_solver_lambda00.RunIteration();

    const absl::flat_hash_map<std::pair<std::string, Action>, double>
        &q_values_lambda00 = tabular_sarsa_solver_lambda00.GetQValueTable();
    std::unique_ptr<State> state = game->NewInitialState();

    while (!state->IsTerminal()) {
      state->ApplyAction(GetOptimalAction(q_values_lambda00, state));
    }

    count_tie_games_lambda00 += state->Rewards()[0] == 0 ? 1 : 0;
  }

  for (int i = 0; i < runs; ++i) {
    tabular_sarsa_solver_lambda03.RunIteration();

    const absl::flat_hash_map<std::pair<std::string, Action>, double>
        &q_values_lambda01 = tabular_sarsa_solver_lambda03.GetQValueTable();
    std::unique_ptr<State> state = game->NewInitialState();

    while (!state->IsTerminal()) {
      state->ApplyAction(GetOptimalAction(q_values_lambda01, state));
    }

    count_tie_games_lambda03 += state->Rewards()[0] == 0 ? 1 : 0;
  }

  // SARSA(0.3) gets equilibrium faster than SARSA(0.0). More ties in the same
  // amount of time.
  SPIEL_CHECK_GT(count_tie_games_lambda03, count_tie_games_lambda00);
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
  SolveTicTacToeEligibilityTraces();
  SolveCatch();
  return 0;
}
