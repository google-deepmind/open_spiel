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

#ifndef OPEN_SPIEL_ALGORITHMS_TABULAR_Q_LEARNING_H_
#define OPEN_SPIEL_ALGORITHMS_TABULAR_Q_LEARNING_H_

#include "open_spiel/abseil-cpp/absl/container/flat_hash_map.h"
#include "open_spiel/abseil-cpp/absl/random/distributions.h"
#include "open_spiel/abseil-cpp/absl/random/random.h"
#include "open_spiel/algorithms/get_all_states.h"
#include "open_spiel/spiel.h"

namespace open_spiel {
namespace algorithms {

// Tabular Q-learning algorithm: solves for the optimal action value function
// of a game.
// It considers all states with depth at most depth_limit from the
// initial state (so if depth_limit is 0, only the root is considered).
// If depth limit is negative, all states are considered.
//
// Currently works for sequential 1-player or 2-player zero-sum games.
//
// Based on the implementation in Sutton and Barto, Intro to RL. Second Edition,
// 2018. Section 6.5.
// Note: current implementation only supports full bootstrapping (lambda = 0).

class TabularQLearningSolver {
  static inline constexpr double kDefaultDepthLimit = -1;
  static inline constexpr double kDefaultEpsilon = 0.01;
  static inline constexpr double kDefaultLearningRate = 0.01;
  static inline constexpr double kDefaultDiscountFactor = 0.99;
  static inline constexpr double kDefaultLambda = 0;

 public:
  TabularQLearningSolver(std::shared_ptr<const Game> game);

  TabularQLearningSolver(std::shared_ptr<const Game> game, double depth_limit,
                         double epsilon, double learning_rate,
                         double discount_factor, double lambda);

  void RunIteration();

  const absl::flat_hash_map<std::pair<std::string, Action>, double>&
  GetQValueTable() const;

 private:
  // Given a player and a state, gets the best possible action from this state
  Action GetBestAction(const State& state, double min_utility);

  // Given a state, gets the best possible action value from this state
  double GetBestActionValue(const State& state, double min_utility);

  // Given a player and a state, gets the action, sampled from an epsilon-greedy
  // policy
  Action SampleActionFromEpsilonGreedyPolicy(const State& state,
                                             double min_utility);

  // Moves a chance node to the next decision/terminal node by sampling from
  // the legal actions repeatedly
  void SampleUntilNextStateOrTerminal(State* state);

  std::shared_ptr<const Game> game_;
  int depth_limit_;
  double epsilon_;
  double learning_rate_;
  double discount_factor_;
  double lambda_;
  std::mt19937 rng_;
  absl::flat_hash_map<std::pair<std::string, Action>, double> values_;
};

}  // namespace algorithms
}  // namespace open_spiel

#endif  // OPEN_SPIEL_ALGORITHMS_TABULAR_Q_LEARNING_H_
