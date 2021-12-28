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

#include "open_spiel/algorithms/ortools/lp_solver.h"

#include <numeric>

#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/algorithms/corr_dist.h"
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

NormalFormCorrelationDevice ComputeCorrelatedEquilibrium(
    const NormalFormGame& normal_form_game, CorrEqObjType obj_type,
    double social_welfare_lower_bound) {
  // Implements an LP solver as explained in Section 4.6 of Shoham and
  // Leyton-Brown '09: http://masfoundations.org/

  // The NormalFormState inherits from SimultaneousGame, which conveniently
  // provides a flattened joint action space, which is useful for setting up
  // the LP.
  std::unique_ptr<State> initial_state = normal_form_game.NewInitialState();
  NFGState* nfg_state = static_cast<NFGState*>(initial_state.get());
  std::vector<Action> flat_joint_actions = nfg_state->LegalActions();

  opres::MPSolver solver("solver", opres::MPSolver::GLOP_LINEAR_PROGRAMMING);
  std::vector<opres::MPVariable*> variables;
  variables.reserve(flat_joint_actions.size());

  // Probability and distribution constraints.
  opres::MPConstraint* const sum_to_one =
      solver.MakeRowConstraint(1.0, 1.0, "sum_to_one");
  for (int i = 0; i < flat_joint_actions.size(); ++i) {
    variables.push_back(solver.MakeNumVar(0.0, 1.0, absl::StrCat("var ", i)));
    sum_to_one->SetCoefficient(variables[i], 1.0);
  }

  // Utility constraints.
  for (Player p = 0; p < normal_form_game.NumPlayers(); ++p) {
    // This player's legal actions a_i
    for (Action a_i : nfg_state->LegalActions(p)) {
      // This player's alternative legal actions a_i'
      for (Action a_ip : nfg_state->LegalActions(p)) {
        // Consider only alternatives a_i' != a_i
        if (a_ip == a_i) {
          continue;
        }

        // Now add the constraint:
        // \sum_{a \in A | a_i \in a} [u_i(a) - u_i(a_i', a_{-i})] p(a) >= 0.
        opres::MPConstraint* const constraint = solver.MakeRowConstraint();
        constraint->SetLB(0.0);

        for (int ja_idx = 0; ja_idx < flat_joint_actions.size(); ++ja_idx) {
          std::vector<Action> joint_action =
              nfg_state->FlatJointActionToActions(flat_joint_actions[ja_idx]);
          // Skip this joint action if a_i is not taken for this player.
          if (joint_action[p] != a_i) {
            continue;
          }

          std::vector<Action> alternative_joint_action = joint_action;
          alternative_joint_action[p] = a_ip;

          double coeff =
              normal_form_game.GetUtility(p, joint_action) -
              normal_form_game.GetUtility(p, alternative_joint_action);
          constraint->SetCoefficient(variables[ja_idx], coeff);
        }
      }
    }
  }

  opres::MPObjective* objective = solver.MutableObjective();
  objective->SetMaximization();

  // Objective depends on the type.
  if (obj_type == CorrEqObjType::kSocialWelfareAtLeast) {
    // Add constraint expected SW >= k.
    opres::MPConstraint* constraint = solver.MakeRowConstraint();
    constraint->SetLB(social_welfare_lower_bound);
    for (int i = 0; i < variables.size(); ++i) {
      std::vector<Action> joint_action =
          nfg_state->FlatJointActionToActions(flat_joint_actions[i]);
      std::vector<double> utilities =
          normal_form_game.GetUtilities(joint_action);
      constraint->SetCoefficient(
          variables[i],
          std::accumulate(utilities.begin(), utilities.end(), 0.0));
    }
  } else if (obj_type == CorrEqObjType::kSocialWelfareMax) {
    // Set the objective to the max social welfare
    for (int i = 0; i < variables.size(); ++i) {
      std::vector<Action> joint_action =
          nfg_state->FlatJointActionToActions(flat_joint_actions[i]);
      std::vector<double> utilities =
          normal_form_game.GetUtilities(joint_action);
      objective->SetCoefficient(
          variables[i],
          std::accumulate(utilities.begin(), utilities.end(), 0.0));
    }
  }

  solver.Solve();

  NormalFormCorrelationDevice mu;
  mu.reserve(variables.size());
  for (int i = 0; i < variables.size(); ++i) {
    mu.push_back({variables[i]->solution_value(),  // probability, actions
                  nfg_state->FlatJointActionToActions(flat_joint_actions[i])});
  }

  return mu;
}

}  // namespace ortools
}  // namespace algorithms
}  // namespace open_spiel
