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

#include "open_spiel/algorithms/mcts.h"

#include <cmath>
#include <limits>
#include <random>

namespace open_spiel {
namespace algorithms {
namespace {

// A node in the search tree for MCTS
struct SearchNode {
  int explore_count = 0;    // number of times this node was explored
  int player_sign = 0;      // 1 for player 0, -1 for player 1
  double total_reward = 0;  // total reward passing through this node

  // The two following vectors are aligned: actions[i] applied to this state
  // gives the state corresponding to node children[i].
  std::vector<Action> actions;
  std::vector<SearchNode> children;

  // UCT value of given child
  double ChildValue(int child_index, double uct_c) const {
    const auto& child = children[child_index];
    // Unexplored nodes have infinite value
    if (child.explore_count == 0)
      return std::numeric_limits<double>::infinity();

    // The "greedy-value" of choosing a given child is always with respect to
    // the current player for this node.
    return player_sign * child.total_reward / child.explore_count +
           uct_c * sqrt(log(explore_count) / child.explore_count);
  }

  // Returns the most visited action in this node.
  Action MostVisitedAction() {
    Action chosen_action = actions[0];
    int largest_visit = children[0].explore_count;
    for (int i = 0; i < children.size(); ++i) {
      if (children[i].explore_count > largest_visit) {
        largest_visit = children[i].explore_count;
        chosen_action = actions[i];
      }
    }
    return chosen_action;
  }

  SearchNode() {}
};

// The expansion portion of the MCTS algorithm.
// Starting from the initial state, apply actions according to UCT until a new
// node is added
std::unique_ptr<State> ApplyTreePolicy(SearchNode* root,
                                       std::vector<SearchNode*>* visit_path,
                                       const State& state, double uct_c,
                                       std::mt19937* rng) {
  // visit_path records each SearchNode that was visited during this expansion
  visit_path->push_back(root);
  auto working_state = state.Clone();
  SearchNode* current_node = root;
  while (!working_state->IsTerminal()) {
    if (current_node->explore_count == 0) {
      // This node is explored for the first time, so initialize this node.
      for (auto action : working_state->LegalActions()) {
        current_node->actions.push_back(action);
        current_node->children.emplace_back();
      }
      current_node->player_sign =
          (working_state->CurrentPlayer() == 0) ? 1 : -1;
      return working_state;
    }

    // Find next state to visit.
    // For decision nodes, choose child with highest UCT value
    // For chance nodes, sample according to the distribution of that node
    int max_index = -1;
    if (working_state->IsChanceNode()) {
      auto outcomes = working_state->ChanceOutcomes();
      double rand = std::uniform_real_distribution<double>(0.0, 1.0)(*rng);
      double s = 0;

      for (max_index = 0; s < rand; ++max_index) {
        s += outcomes[max_index].second;
      }
    } else {
      double max_value = -std::numeric_limits<double>::infinity();
      for (int index = 0; index < current_node->actions.size(); ++index) {
        double val = current_node->ChildValue(index, uct_c);
        if (val > max_value) {
          max_index = index;
          max_value = val;
        }
      }
    }

    // Apply the action and visit the next node
    working_state->ApplyAction(current_node->actions[max_index]);
    current_node = &current_node->children[max_index];
    visit_path->push_back(current_node);
  }

  return working_state;
}
}  // namespace

double RandomRolloutEvaluator::evaluate(const State& state) const {
  double result = 0;
  for (int i = 0; i < n_rollouts_; ++i) {
    auto working_state = state.Clone();
    while (!working_state->IsTerminal()) {
      if (working_state->IsChanceNode()) {
        auto outcomes = working_state->ChanceOutcomes();
        Action action = SampleChanceOutcome(
            outcomes, std::uniform_real_distribution<double>(0.0, 1.0)(rng_));
        working_state->ApplyAction(action);
      } else {
        auto actions = working_state->LegalActions();
        std::uniform_int_distribution<int> dist(0, actions.size() - 1);
        int index = dist(rng_);
        working_state->ApplyAction(actions[index]);
      }
    }
    result += working_state->PlayerReturn(0);
  }
  return result / n_rollouts_;
}

Action MCTSearch(const State& state, double uct_c, int max_search_nodes,
                 const Evaluator& evaluator) {
  std::mt19937 rng;
  SearchNode root;
  std::vector<SearchNode*> visit_path;
  visit_path.reserve(64);
  for (int i = 0; i < max_search_nodes; ++i) {
    visit_path.clear();
    // First expand the node
    auto working_state =
        ApplyTreePolicy(&root, &visit_path, state, uct_c, &rng);

    // Now evaluate this node
    double node_value;
    if (working_state->IsTerminal())
      node_value = working_state->PlayerReturn(0);
    else
      node_value = evaluator.evaluate(*working_state);

    // Propagate values back
    for (auto node : visit_path) {
      node->total_reward += node_value;
      node->explore_count += 1;
    }
  }

  return root.MostVisitedAction();
}

std::pair<ActionsAndProbs, Action> MCTSBot::Step(const State& state) {
  auto mcts_action = MCTSearch(state, uct_c_, max_search_nodes_, evaluator_);
  return {{{mcts_action, 1.0}}, mcts_action};
}

}  // namespace algorithms
}  // namespace open_spiel
