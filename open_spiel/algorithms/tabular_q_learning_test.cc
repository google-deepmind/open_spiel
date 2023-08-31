// Copyright 2023 DeepMind Technologies Limited
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

#include "open_spiel/algorithms/tabular_q_learning.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "open_spiel/abseil-cpp/absl/random/distributions.h"
#include "open_spiel/abseil-cpp/absl/random/random.h"
#include "open_spiel/games/catch/catch.h"
#include "open_spiel/spiel.h"

namespace open_spiel {
namespace {

Action GetOptimalAction(
    absl::flat_hash_map<std::pair<std::string, Action>, double> q_values,
    const std::unique_ptr<State> &state) {
  std::vector<Action> legal_actions = state->LegalActions();
  const auto state_str = state->ToString();

  Action optimal_action = open_spiel::kInvalidAction;
  double value = -1;
  for (const Action &action : legal_actions) {
    double q_val = q_values[{state_str, action}];
    if (q_val >= value) {
      value = q_val;
      optimal_action = action;
    }
  }
  return optimal_action;
}

Action GetRandomAction(const std::unique_ptr<State> &state, int seed) {
  std::vector<Action> legal_actions = state->LegalActions();
  if (legal_actions.empty()) {
    return kInvalidAction;
  }
  std::mt19937 rng(seed);
  return legal_actions[absl::Uniform<int>(rng, 0, legal_actions.size())];
}

double PlayCatch(
    absl::flat_hash_map<std::pair<std::string, Action>, double> q_values,
    const std::unique_ptr<State> &state, double seed) {
  // First action determines the starting column. Do the first action before the
  // main loop, where the optimal action is chosen.
  // Example: Initial state with random seed 42
  // ...o.
  // .....
  // .....
  // .....
  // .....
  // .....
  // .....
  // .....
  // .....
  // ..x..
  std::mt19937 gen(seed);
  std::uniform_int_distribution<int> distribution(0,
                                                  catch_::kDefaultColumns - 1);
  int ball_starting_column = distribution(gen);
  state->ApplyAction(ball_starting_column);

  while (!state->IsTerminal()) {
    Action optimal_action = GetOptimalAction(q_values, state);
    state->ApplyAction(optimal_action);
  }

  return state->Rewards()[0];
}

std::unique_ptr<open_spiel::algorithms::TabularQLearningSolver> QLearningSolver(
    std::shared_ptr<const Game> game, double lambda) {
  return std::make_unique<open_spiel::algorithms::TabularQLearningSolver>(
      /*game=*/game,
      /*depth_limit=*/-1.0,
      /*epsilon=*/0.1,
      /*learning_rate=*/0.01,
      /*discount_factor=*/0.99,
      /*lambda=*/lambda);
}

void TabularQLearningTest_Catch_Lambda00_Loss() {
  // Classic Q-learning. No bootstraping (lambda=0.0)
  // Player loses after only 1 train iteration.
  std::shared_ptr<const Game> game = LoadGame("catch");
  auto tabular_q_learning_solver = QLearningSolver(game, 0);

  tabular_q_learning_solver->RunIteration();
  const absl::flat_hash_map<std::pair<std::string, Action>, double>& q_values =
      tabular_q_learning_solver->GetQValueTable();
  std::unique_ptr<State> state = game->NewInitialState();

  double reward = PlayCatch(q_values, state, 42);
  SPIEL_CHECK_EQ(reward, -1);
}

void TabularQLearningTest_Catch_Lambda00_Win() {
  // Classic Q-learning. No bootstraping (lambda=0.0)
  // Player wins after 100 train iterations
  std::shared_ptr<const Game> game = LoadGame("catch");
  auto tabular_q_learning_solver = QLearningSolver(game, 0);

  for (int i = 1; i < 100; i++) {
    tabular_q_learning_solver->RunIteration();
  }
  const absl::flat_hash_map<std::pair<std::string, Action>, double>& q_values =
      tabular_q_learning_solver->GetQValueTable();
  std::unique_ptr<State> state = game->NewInitialState();

  double reward = PlayCatch(q_values, state, 42);
  SPIEL_CHECK_EQ(reward, 1);
}

void TabularQLearningTest_Catch_Lambda01_Win() {
  // Player wins after 100 train iterations
  std::shared_ptr<const Game> game = LoadGame("catch");
  auto tabular_q_learning_solver = QLearningSolver(game, 0.1);

  for (int i = 1; i < 100; i++) {
    tabular_q_learning_solver->RunIteration();
  }
  const absl::flat_hash_map<std::pair<std::string, Action>, double>& q_values =
      tabular_q_learning_solver->GetQValueTable();
  std::unique_ptr<State> state = game->NewInitialState();

  double reward = PlayCatch(q_values, state, 42);
  SPIEL_CHECK_EQ(reward, 1);
}

void TabularQLearningTest_Catch_Lambda01FasterThanLambda00() {
  // Eligibility traces (lambda > 0.0) always achieves victory with less
  // training steps w.r.t. Q-learning(lambda=0.0)
  std::shared_ptr<const Game> game = LoadGame("catch");
  auto tabular_q_learning_solver_lambda00 = QLearningSolver(game, 0);
  auto tabular_q_learning_solver_lambda01 = QLearningSolver(game, 0.1);

  for (int seed = 0; seed < 100; seed++) {
    int lambda_00_train_iter = 0;
    int lambda_01_train_iter = 0;
    double lambda_00_reward = -1.0;
    double lambda_01_reward = -1.0;

    while (lambda_00_reward == -1.0) {
      tabular_q_learning_solver_lambda00->RunIteration();
      std::unique_ptr<State> state = game->NewInitialState();
      lambda_00_reward = PlayCatch(
          tabular_q_learning_solver_lambda00->GetQValueTable(), state, seed);
      lambda_00_train_iter++;
    }
    while (lambda_01_reward == -1.0) {
      tabular_q_learning_solver_lambda01->RunIteration();
      std::unique_ptr<State> state = game->NewInitialState();
      lambda_01_reward = PlayCatch(
          tabular_q_learning_solver_lambda01->GetQValueTable(), state, seed);
      lambda_01_train_iter++;
    }
    SPIEL_CHECK_GE(lambda_00_train_iter, lambda_01_train_iter);
  }
}

void TabularQLearningTest_TicTacToe_Lambda01_Win() {
  std::shared_ptr<const Game> game = open_spiel::LoadGame("tic_tac_toe");
  auto tabular_q_learning_solver = QLearningSolver(game, 0.1);

  for (int i = 1; i < 100; i++) {
    tabular_q_learning_solver->RunIteration();
  }

  const absl::flat_hash_map<std::pair<std::string, Action>, double>& q_values =
      tabular_q_learning_solver->GetQValueTable();
  std::unique_ptr<State> state = game->NewInitialState();

  while (!state->IsTerminal()) {
    Action random_action = GetRandomAction(state, 42);
    state->ApplyAction(random_action);  // player 0
    if (random_action == kInvalidAction) break;
    state->ApplyAction(GetOptimalAction(q_values, state));  // player 1
  }

  SPIEL_CHECK_EQ(state->Rewards()[0], -1);
}

void TabularQLearningTest_TicTacToe_Lambda01_Tie() {
  std::shared_ptr<const Game> game = open_spiel::LoadGame("tic_tac_toe");
  auto tabular_q_learning_solver = QLearningSolver(game, 0.1);

  for (int i = 1; i < 1000; i++) {
    tabular_q_learning_solver->RunIteration();
  }

  const absl::flat_hash_map<std::pair<std::string, Action>, double>& q_values =
      tabular_q_learning_solver->GetQValueTable();
  std::unique_ptr<State> state = game->NewInitialState();

  while (!state->IsTerminal()) {
    state->ApplyAction(GetOptimalAction(q_values, state));
  }

  SPIEL_CHECK_EQ(state->Rewards()[0], 0);
}

}  // namespace
}  // namespace open_spiel

int main(int argc, char **argv) {
  open_spiel::TabularQLearningTest_Catch_Lambda00_Loss();
  open_spiel::TabularQLearningTest_Catch_Lambda00_Win();
  open_spiel::TabularQLearningTest_Catch_Lambda01_Win();
  open_spiel::TabularQLearningTest_Catch_Lambda01FasterThanLambda00();
  open_spiel::TabularQLearningTest_TicTacToe_Lambda01_Win();
  open_spiel::TabularQLearningTest_TicTacToe_Lambda01_Tie();
}
