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

#include "open_spiel/algorithms/ortools/sequence_form_lp.h"

#include <map>
#include <memory>
#include <utility>

#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include "ortools/linear_solver/linear_solver.h"

namespace open_spiel {
namespace algorithms {
namespace ortools {

namespace opres = operations_research;

SequenceFormLpSpecification::SequenceFormLpSpecification(
    const Game& game, const std::string& solver_id)
    : SequenceFormLpSpecification(
          {
              MakeInfostateTree(game, 0),
              MakeInfostateTree(game, 1),
          },
          solver_id) {}

SequenceFormLpSpecification::SequenceFormLpSpecification(
    std::vector<std::shared_ptr<InfostateTree>> trees,
    const std::string& solver_id)
    : trees_(std::move(trees)),
      terminal_bijection_(ConnectTerminals(*trees_[0], *trees_[1])),
      solver_(MPSolver::CreateSolver(solver_id)),
      node_spec_() {
  SPIEL_CHECK_TRUE(solver_);
  SPIEL_CHECK_EQ(trees_.size(), 2);
}

void SequenceFormLpSpecification::SpecifyReachProbsConstraints(
    InfostateNode* player_node) {
  node_spec_[player_node].var_reach_prob = solver_->MakeNumVar(
      /*lb=*/0.0, /*ub=*/1., "");

  if (player_node->type() == kTerminalInfostateNode) return;  // Nothing to do.
  if (player_node->type() == kObservationInfostateNode) {
    for (InfostateNode* player_child : player_node->child_iterator()) {
      SpecifyReachProbsConstraints(player_child);

      // Equality constraint: parent = child
      opres::MPConstraint* ct = node_spec_[player_child].ct_parent_reach_prob =
          solver_->MakeRowConstraint(/*lb=*/0, /*ub=*/0, "");
      ct->SetCoefficient(node_spec_[player_node].var_reach_prob, -1);
      ct->SetCoefficient(node_spec_[player_child].var_reach_prob, 1);
    }
    return;
  }
  if (player_node->type() == kDecisionInfostateNode) {
    // Equality constraint: parent = sum of children
    opres::MPConstraint* ct = node_spec_[player_node].ct_child_reach_prob =
        solver_->MakeRowConstraint(/*lb=*/0, /*ub=*/0, "");
    ct->SetCoefficient(node_spec_[player_node].var_reach_prob, -1);
    for (InfostateNode* player_child : player_node->child_iterator()) {
      SpecifyReachProbsConstraints(player_child);
      ct->SetCoefficient(node_spec_[player_child].var_reach_prob, 1);
    }
    return;
  }

  SpielFatalError("Exhausted pattern match!");
}

void SequenceFormLpSpecification::SpecifyCfValuesConstraints(
    InfostateNode* opponent_node) {
  node_spec_[opponent_node].var_cf_value = solver_->MakeNumVar(
      /*lb=*/-opres::MPSolver::infinity(),
      /*ub=*/opres::MPSolver::infinity(), "");

  if (opponent_node->type() == kDecisionInfostateNode) {
    for (InfostateNode* opponent_child : opponent_node->child_iterator()) {
      SpecifyCfValuesConstraints(opponent_child);
      opres::MPConstraint* ct = node_spec_[opponent_child].ct_parent_cf_value =
          solver_->MakeRowConstraint();
      ct->SetUB(0.);
      ct->SetCoefficient(node_spec_[opponent_node].var_cf_value, -1);
      ct->SetCoefficient(node_spec_[opponent_child].var_cf_value, 1);
    }
    return;
  }

  opres::MPConstraint* ct = node_spec_[opponent_node].ct_child_cf_value =
      solver_->MakeRowConstraint();
  ct->SetUB(0.);
  ct->SetCoefficient(node_spec_[opponent_node].var_cf_value, -1);

  if (opponent_node->type() == kTerminalInfostateNode) {
    const std::map<const InfostateNode*, const InfostateNode*>& terminal_map =
        terminal_bijection_.association(opponent_node->tree().acting_player());
    const InfostateNode* player_node = terminal_map.at(opponent_node);
    const double value = opponent_node->terminal_utility() *
                         opponent_node->terminal_chance_reach_prob();
    // Terminal value constraint comes from the opponent.
    ct->SetCoefficient(node_spec_[player_node].var_reach_prob, value);
    return;
  }
  if (opponent_node->type() == kObservationInfostateNode) {
    // Value constraint: sum of children = parent
    ct->SetLB(0.);
    for (InfostateNode* opponent_child : opponent_node->child_iterator()) {
      SpecifyCfValuesConstraints(opponent_child);
      ct->SetCoefficient(node_spec_[opponent_child].var_cf_value, 1);
    }
    return;
  }

  SpielFatalError("Exhausted pattern match!");
}

void SequenceFormLpSpecification::SpecifyRootConstraints(
    const InfostateNode* player_root_node) {
  SPIEL_CHECK_TRUE(player_root_node->is_root_node());
  NodeSpecification& root_data = node_spec_.at(player_root_node);
  root_data.var_reach_prob->SetLB(1.);
  root_data.var_reach_prob->SetUB(1.);
}

void SequenceFormLpSpecification::SpecifyObjective(
    const InfostateNode* opponent_root_node) {
  opres::MPObjective* const objective = solver_->MutableObjective();
  objective->SetCoefficient(node_spec_[opponent_root_node].var_cf_value, 1);
  objective->SetMinimization();
}

void SequenceFormLpSpecification::ClearSpecification() {
  solver_->Clear();
  for (auto& [node, spec] : node_spec_) {
    spec.var_cf_value = nullptr;
    spec.var_reach_prob = nullptr;
    spec.ct_child_cf_value = nullptr;
    spec.ct_parent_cf_value = nullptr;
    spec.ct_child_reach_prob = nullptr;
    spec.ct_parent_reach_prob = nullptr;
  }
}

void SequenceFormLpSpecification::SpecifyLinearProgram(Player pl) {
  SPIEL_CHECK_TRUE(pl == 0 || pl == 1);
  ClearSpecification();
  SpecifyReachProbsConstraints(
      /*player_node=*/trees_[pl]->mutable_root());
  SpecifyRootConstraints(
      /*player_root_node=*/trees_[pl]->mutable_root());
  SpecifyCfValuesConstraints(
      /*opponent_node=*/trees_[1 - pl]->mutable_root());
  SpecifyObjective(
      /*opponent_root_node=*/trees_[1 - pl]->mutable_root());
}

double SequenceFormLpSpecification::Solve() {
  opres::MPSolver::ResultStatus status = solver_->Solve();
  //  // Export the model if the result was not optimal.
  //  // You can then use external debugging tools (like cplex studio).
  //  if (status != opres::MPSolver::ResultStatus::OPTIMAL) {
  //    std::string out;
  //    // Pick the format.
  //    solver_->ExportModelAsMpsFormat(false, false, &out);
  //    solver_->ExportModelAsLpFormat(false, &out);
  //    std::cout << out << "\n";
  //  }
  SPIEL_CHECK_EQ(status, opres::MPSolver::ResultStatus::OPTIMAL);
  return -solver_->Objective().Value();
}

TabularPolicy SequenceFormLpSpecification::OptimalPolicy(Player for_player) {
  SPIEL_CHECK_TRUE(for_player == 0 || for_player == 1);
  const InfostateTree* tree = trees_[for_player].get();
  TabularPolicy policy;
  for (DecisionId id : tree->AllDecisionIds()) {
    const InfostateNode* node = tree->decision_infostate(id);
    absl::Span<const Action> actions = node->legal_actions();
    SPIEL_CHECK_EQ(actions.size(), node->num_children());
    ActionsAndProbs state_policy;
    state_policy.reserve(node->num_children());
    double rp_sum = 0.;
    for (int i = 0; i < actions.size(); ++i) {
      rp_sum += node_spec_[node->child_at(i)].var_reach_prob->solution_value();
    }
    for (int i = 0; i < actions.size(); ++i) {
      double prob;
      if (rp_sum) {
        prob = node_spec_[node->child_at(i)].var_reach_prob->solution_value() /
               rp_sum;
      } else {
        // If the infostate is unreachable, the strategy is not defined.
        // However some code in the library may require having the strategy,
        // so we just put an uniform strategy here.
        prob = 1. / actions.size();
      }
      state_policy.push_back({actions[i], prob});
    }
    policy.SetStatePolicy(node->infostate_string(), state_policy);
  }
  return policy;
}

SfStrategy SequenceFormLpSpecification::OptimalSfStrategy(Player for_player) {
  SPIEL_CHECK_TRUE(for_player == 0 || for_player == 1);
  const InfostateTree* tree = trees_[for_player].get();
  SfStrategy strategy(tree);
  for (SequenceId id : tree->AllSequenceIds()) {
    const InfostateNode* node = tree->observation_infostate(id);
    strategy[id] = node_spec_[node].var_reach_prob->solution_value();
  }
  return strategy;
}

BijectiveContainer<const InfostateNode*> ConnectTerminals(
    const InfostateTree& tree_a, const InfostateTree& tree_b) {
  BijectiveContainer<const InfostateNode*> out;

  using History = absl::Span<const Action>;
  std::map<History, const InfostateNode*> history_map;
  for (InfostateNode* node_b : tree_b.leaf_nodes()) {
    history_map[node_b->TerminalHistory()] = node_b;
  }

  for (InfostateNode* node_a : tree_a.leaf_nodes()) {
    const InfostateNode* node_b = history_map[node_a->TerminalHistory()];
    out.put({node_a, node_b});
  }
  return out;
}

void SequenceFormLpSpecification::PrintProblemSpecification() {
  const std::vector<opres::MPVariable*>& variables = solver_->variables();
  const std::vector<opres::MPConstraint*>& constraints = solver_->constraints();
  const opres::MPObjective& objective = solver_->Objective();

  std::cout << "Objective:" << std::endl;
  if (objective.maximization()) {
    std::cout << "max ";
  } else {
    std::cout << "min ";
  }
  bool first_obj = true;
  for (int i = 0; i < variables.size(); ++i) {
    const double coef = objective.GetCoefficient(variables[i]);
    if (coef) {
      if (!first_obj) std::cout << "+ ";
      std::cout << coef << "*x" << i << " ";
      first_obj = false;
    }
  }
  std::cout << std::endl;

  std::cout << "Constraints:" << std::endl;
  for (auto& ct : constraints) {
    std::cout << ct->lb() << " <= ";
    bool first_ct = true;
    for (int i = 0; i < variables.size(); ++i) {
      const double coef = ct->GetCoefficient(variables[i]);
      if (coef) {
        if (!first_ct) std::cout << "+ ";
        std::cout << coef << "*x" << i << " ";
        first_ct = false;
      }
    }
    std::cout << "<= " << ct->ub() << " (" << ct->name() << ")" << std::endl;
  }

  std::cout << "Variables:" << std::endl;
  for (int i = 0; i < variables.size(); i++) {
    const auto& var = variables[i];
    std::cout << var->lb() << " <= "
              << "x" << i << " <= " << var->ub() << " (" << var->name() << ")"
              << std::endl;
  }
}

}  // namespace ortools
}  // namespace algorithms
}  // namespace open_spiel
