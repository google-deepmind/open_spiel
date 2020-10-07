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

#include "open_spiel/algorithms/ortools/lp_solver.h"

#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/spiel.h"
#include "ortools/linear_solver/linear_solver.h"

namespace open_spiel {
namespace algorithms {
namespace ortools {

namespace opres = operations_research;

std::pair<opres::MPVariable*, opres::MPObjective*> SetupVariablesAndObjective(
    opres::MPSolver* solver, std::vector<opres::MPVariable*>* variables,
    int num_strategy_variables, double min_utility, double max_utility) {
  // Value and strategy probability variables
  opres::MPVariable* v = solver->MakeNumVar(min_utility, max_utility, "v");
  variables->reserve(num_strategy_variables);
  for (int i = 0; i < num_strategy_variables; ++i) {
    variables->push_back(solver->MakeNumVar(0.0, 1.0, absl::StrCat("var ", i)));
  }

  // Strategy probs sum to one
  opres::MPConstraint* const sum_to_one =
      solver->MakeRowConstraint(1.0, 1.0, "sum_to_one");
  for (int i = 0; i < num_strategy_variables; ++i) {
    sum_to_one->SetCoefficient((*variables)[i], 1.0);
  }

  opres::MPObjective* objective = solver->MutableObjective();
  objective->SetCoefficient(v, 1.0);
  objective->SetMaximization();

  return {v, objective};
}

ZeroSumGameSolution SolveZeroSumMatrixGame(
    const matrix_game::MatrixGame& matrix_game) {
  SPIEL_CHECK_EQ(matrix_game.GetType().information,
                 GameType::Information::kOneShot);
  SPIEL_CHECK_EQ(matrix_game.GetType().utility, GameType::Utility::kZeroSum);
  int num_rows = matrix_game.NumRows();
  int num_cols = matrix_game.NumCols();
  double min_utility = matrix_game.MinUtility();
  double max_utility = matrix_game.MaxUtility();

  // Solving a game for player i (e.g. row player) requires finding a mixed
  // policy over player i's pure strategies (actions) such that a value of the
  // mixed strategy against every opponent pure strategy is maximized.
  //
  // For more detail, please refer to Sec 4.1 of Shoham & Leyton-Brown, 2009:
  // Multiagent Systems: Algorithmic, Game-Theoretic, and Logical Foundations
  // http://www.masfoundations.org/mas.pdf
  //
  // For the row player the LP looks like:
  //    max V
  //     st. sigma_a1 \dot col_0 >= V
  //         sigma_a2 \dot col_1 >= V
  //              .
  //              .
  //         sigma_am \cot col_n >= V
  //         for all i, sigma_ai >= 0
  //         sigma \dot 1 = 1

  ZeroSumGameSolution solution{
      {0, 0},
      {std::vector<double>(num_rows, 0), std::vector<double>(num_cols, 0)}};

  // First, the row player (player 0).
  opres::MPSolver p0_solver("solver", opres::MPSolver::GLOP_LINEAR_PROGRAMMING);
  std::vector<opres::MPVariable*> p0_vars;
  auto [p0_v, p0_objective] = SetupVariablesAndObjective(
      &p0_solver, &p0_vars, num_rows, min_utility, max_utility);
  // Utility constriants
  for (int c = 0; c < num_cols; ++c) {
    opres::MPConstraint* const constraint = p0_solver.MakeRowConstraint();
    constraint->SetLB(0.0);
    constraint->SetCoefficient(p0_v, -1.0);
    for (int r = 0; r < num_rows; ++r) {
      constraint->SetCoefficient(p0_vars[r],
                                 matrix_game.PlayerUtility(0, r, c));
    }
  }

  p0_solver.Solve();
  solution.values[0] = p0_objective->Value();
  for (int r = 0; r < num_rows; ++r) {
    solution.strategies[0][r] = p0_vars[r]->solution_value();
  }

  // Now, the column player.
  opres::MPSolver p1_solver("solver", opres::MPSolver::GLOP_LINEAR_PROGRAMMING);
  std::vector<opres::MPVariable*> p1_vars;
  auto [p1_v, p1_objective] = SetupVariablesAndObjective(
      &p1_solver, &p1_vars, num_cols, min_utility, max_utility);

  // Utility constriants
  for (int r = 0; r < num_rows; ++r) {
    opres::MPConstraint* const constraint = p1_solver.MakeRowConstraint();
    constraint->SetLB(0.0);
    constraint->SetCoefficient(p1_v, -1.0);
    for (int c = 0; c < num_cols; ++c) {
      constraint->SetCoefficient(p1_vars[c],
                                 matrix_game.PlayerUtility(1, r, c));
    }
  }

  p1_solver.Solve();
  solution.values[1] = p1_objective->Value();
  for (int c = 0; c < num_cols; ++c) {
    solution.strategies[1][c] = p1_vars[c]->solution_value();
  }

  return solution;
}

}  // namespace ortools
}  // namespace algorithms
}  // namespace open_spiel
