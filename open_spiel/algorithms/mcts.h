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

#ifndef THIRD_PARTY_OPEN_SPIEL_ALGORITHMS_MCTS_H_
#define THIRD_PARTY_OPEN_SPIEL_ALGORITHMS_MCTS_H_

#include <memory>
#include <random>

#include "open_spiel/spiel.h"
#include "open_spiel/spiel_bots.h"

namespace open_spiel {
namespace algorithms {

// Abstract class representing an evaluation function for a game.
// The evaluation function takes in an intermediate state in the game and
// returns an evaluation of that state, which should correlate with chances of
// winning the game for player 0.
class Evaluator {
 public:
  virtual ~Evaluator() = default;
  virtual double evaluate(const State& state) const = 0;
};

// A simple evaluator that returns the average outcome of playing random actions
// from the given state until the end of the game.
// n_rollouts is the number of random outcomes to be considered.
class RandomRolloutEvaluator : public Evaluator {
 public:
  explicit RandomRolloutEvaluator(int n_rollouts) : n_rollouts_{n_rollouts} {}
  double evaluate(const State& state) const override;

 private:
  int n_rollouts_;
  mutable std::mt19937 rng_;
};

// A vanilla Monte-Carlo Tree Search algorithm.
//
// This algorithm searches the game tree from the given state.
// At the leaf, the evaluator is called if the game state is not terminal.
// A total of max_search_nodes states are explored.
//
// At every node, the algorithm chooses the action with the highest UCT value,
// defined as: Q/N + c * sqrt(log(N) / N), where Q is the total reward after the
// action, and N is the number of times the action was explored in this
// position.  The input parameter c controls the balance between exploration and
// exploitation; higher values of c encourage exploration of under-explored
// nodes. Unseen actions are always explored first.
//
// At the end of the search, the chosen action is the action that has been
// explored most often. This is the action that is returned.
//
// This implementation only supports sequential 1-player or 2-player zero-sum
// games, with or without chance nodes.
Action MCTSearch(const State& state, double uct_c, int max_search_nodes,
                 const Evaluator& evaluator);

// A SpielBot that uses the MCTS algorithm as its policy.
class MCTSBot : public Bot {
 public:
  MCTSBot(const Game& game, Player player, double uct_c, int max_search_nodes,
          const Evaluator& evaluator)
      : Bot{game, player},
        uct_c_{uct_c},
        max_search_nodes_{max_search_nodes},
        evaluator_{evaluator} {}

  std::pair<ActionsAndProbs, Action> Step(const State& state) override;

 private:
  double uct_c_;
  int max_search_nodes_;
  const Evaluator& evaluator_;
};

}  // namespace algorithms
}  // namespace open_spiel

#endif  // THIRD_PARTY_OPEN_SPIEL_ALGORITHMS_MCTS_H_
