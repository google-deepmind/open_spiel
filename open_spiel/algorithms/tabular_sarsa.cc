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

#include "open_spiel/algorithms/tabular_sarsa.h"

#include <algorithm>
#include <memory>
#include <random>
#include <string>

#include "open_spiel/abseil-cpp/absl/container/flat_hash_map.h"
#include "open_spiel/abseil-cpp/absl/random/distributions.h"
#include "open_spiel/abseil-cpp/absl/random/random.h"
#include "open_spiel/spiel_globals.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace algorithms {

using std::vector;

Action TabularSarsaSolver::GetBestAction(const State& state,
                                         double min_utility) {
  vector<Action> legal_actions = state.LegalActions();
  SPIEL_CHECK_GT(legal_actions.size(), 0);
  Action best_action = legal_actions[0];

  double value = min_utility;
  for (const Action& action : legal_actions) {
    double q_val = values_[{state.ToString(), action}];
    if (q_val >= value) {
      value = q_val;
      best_action = action;
    }
  }
  return best_action;
}

Action TabularSarsaSolver::SampleActionFromEpsilonGreedyPolicy(
    const State& state, double min_utility) {
  vector<Action> legal_actions = state.LegalActions();
  if (legal_actions.empty()) {
    return kInvalidAction;
  }

  if (absl::Uniform(rng_, 0.0, 1.0) < epsilon_) {
    // Choose a random action
    return legal_actions[absl::Uniform<int>(rng_, 0, legal_actions.size())];
  }
  // Choose the best action
  return GetBestAction(state, min_utility);
}

void TabularSarsaSolver::SampleUntilNextStateOrTerminal(State* state) {
  // Repeatedly sample while chance node, so that we end up at a decision node
  while (state->IsChanceNode() && !state->IsTerminal()) {
    std::vector<std::pair<Action, double>> outcomes = state->ChanceOutcomes();
    state->ApplyAction(SampleAction(outcomes, rng_).first);
  }
}

TabularSarsaSolver::TabularSarsaSolver(std::shared_ptr<const Game> game)
    : game_(game),
      depth_limit_(kDefaultDepthLimit),
      epsilon_(kDefaultEpsilon),
      learning_rate_(kDefaultLearningRate),
      discount_factor_(kDefaultDiscountFactor),
      lambda_(kDefaultLambda) {
  // Only support lambda=0 for now.
  SPIEL_CHECK_EQ(lambda_, 0);

  // Currently only supports 1-player or 2-player zero sum games
  SPIEL_CHECK_TRUE(game_->NumPlayers() == 1 || game_->NumPlayers() == 2);
  if (game_->NumPlayers() == 2) {
    SPIEL_CHECK_EQ(game_->GetType().utility, GameType::Utility::kZeroSum);
  }

  // No support for simultaneous games (needs an LP solver). And so also must
  // be a perfect information game.
  SPIEL_CHECK_EQ(game_->GetType().dynamics, GameType::Dynamics::kSequential);
  SPIEL_CHECK_EQ(game_->GetType().information,
                 GameType::Information::kPerfectInformation);
}

TabularSarsaSolver::TabularSarsaSolver(std::shared_ptr<const Game> game,
                                       double depth_limit, double epsilon,
                                       double learning_rate,
                                       double discount_factor, double lambda)
    : game_(game),
      depth_limit_(depth_limit),
      epsilon_(epsilon),
      learning_rate_(learning_rate),
      discount_factor_(discount_factor),
      lambda_(lambda) {
  // Only support lambda=0 for now.
  SPIEL_CHECK_EQ(lambda_, 0);

  // Currently only supports 1-player or 2-player zero sum games
  SPIEL_CHECK_TRUE(game_->NumPlayers() == 1 || game_->NumPlayers() == 2);
  if (game_->NumPlayers() == 2) {
    SPIEL_CHECK_EQ(game_->GetType().utility, GameType::Utility::kZeroSum);
  }

  // No support for simultaneous games (needs an LP solver). And so also must
  // be a perfect information game.
  SPIEL_CHECK_EQ(game_->GetType().dynamics, GameType::Dynamics::kSequential);
  SPIEL_CHECK_EQ(game_->GetType().information,
                 GameType::Information::kPerfectInformation);
}

const absl::flat_hash_map<std::pair<std::string, Action>, double>&
TabularSarsaSolver::GetQValueTable() const {
  return values_;
}

void TabularSarsaSolver::RunIteration() {
  double min_utility = game_->MinUtility();
  // Choose start state
  std::unique_ptr<State> curr_state = game_->NewInitialState();
  SampleUntilNextStateOrTerminal(curr_state.get());

  Player player = curr_state->CurrentPlayer();
  // Sample action from the state using an epsilon-greedy policy
  Action curr_action =
      SampleActionFromEpsilonGreedyPolicy(*curr_state, min_utility);
  std::unique_ptr<State> next_state;
  SPIEL_CHECK_NE(curr_action, kInvalidAction);

  while (!curr_state->IsTerminal()) {
    player = curr_state->CurrentPlayer();

    next_state = curr_state->Child(curr_action);
    SampleUntilNextStateOrTerminal(next_state.get());
    const double reward = next_state->Rewards()[player];

    const Action next_action =
        next_state->IsTerminal()
            ? kInvalidAction
            : SampleActionFromEpsilonGreedyPolicy(*next_state, min_utility);

    // Update the new q value
    std::string key = curr_state->ToString();
    // Next q-value in perspective of player to play at curr_state (important
    // note: exploits property of two-player zero-sum). Define the value of
    // q(s', a') to be 0 if s' is terminal.
    const double future_value =
        next_state->IsTerminal()
            ? 0
            : values_[{next_state->ToString(), next_action}];
    const double next_q_value =
        (player != next_state->CurrentPlayer() ? -1 : 1) * future_value;
    double new_q_value = reward + discount_factor_ * next_q_value;

    double prev_q_val = values_[{key, curr_action}];
    values_[{key, curr_action}] += learning_rate_ * (new_q_value - prev_q_val);

    curr_state = std::move(next_state);
    curr_action = next_action;
  }
}
}  // namespace algorithms
}  // namespace open_spiel
