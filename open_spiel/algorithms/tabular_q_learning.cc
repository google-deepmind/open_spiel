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

#include "open_spiel/algorithms/tabular_q_learning.h"

#include <algorithm>
#include <random>

#include "open_spiel/abseil-cpp/absl/container/flat_hash_map.h"
#include "open_spiel/abseil-cpp/absl/random/distributions.h"
#include "open_spiel/abseil-cpp/absl/random/random.h"
#include "open_spiel/spiel_globals.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace algorithms {

using std::vector;

Action TabularQLearningSolver::GetBestAction(const State& state,
                                             double min_utility) {
  vector<Action> legal_actions = state.LegalActions();
  Action optimal_action = kInvalidAction;

  double value = min_utility;
  for (const Action& action : legal_actions) {
    double q_val = values_[{state.ToString(), action}];
    if (q_val >= value) {
      value = q_val;
      optimal_action = action;
    }
  }
  return optimal_action;
}

double TabularQLearningSolver::GetBestActionValue(const State& state,
                                                  double min_utility) {
  if (state.IsTerminal()) {
    return 0;
  }
  return values_[{state.ToString(), GetBestAction(state, min_utility)}];
}

Action TabularQLearningSolver::SampleActionFromEpsilonGreedyPolicy(
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

void TabularQLearningSolver::SampleUntilNextStateOrTerminal(State* state) {
  // Repeatedly sample while chance node, so that we end up at a decision node
  while (state->IsChanceNode() && !state->IsTerminal()) {
    vector<Action> legal_actions = state->LegalActions();
    state->ApplyAction(
        legal_actions[absl::Uniform<int>(rng_, 0, legal_actions.size())]);
  }
}

TabularQLearningSolver::TabularQLearningSolver(std::shared_ptr<const Game> game)
    : game_(game),
      depth_limit_(kDefaultDepthLimit),
      epsilon_(kDefaultEpsilon),
      learning_rate_(kDefaultLearningRate),
      discount_factor_(kDefaultDiscountFactor) {
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
TabularQLearningSolver::GetQValueTable() const {
  return values_;
}

void TabularQLearningSolver::RunIteration() {
  const double min_utility = game_->MinUtility();
  // Choose start state
  std::unique_ptr<State> curr_state = game_->NewInitialState();
  SampleUntilNextStateOrTerminal(curr_state.get());

  while (!curr_state->IsTerminal()) {
    const Player player = curr_state->CurrentPlayer();

    // Sample action from the state using an epsilon-greedy policy
    Action curr_action =
        SampleActionFromEpsilonGreedyPolicy(*(curr_state.get()), min_utility);

    std::unique_ptr<State> next_state = curr_state->Child(curr_action);
    SampleUntilNextStateOrTerminal(curr_state.get());

    const double reward = next_state->Rewards()[player];
    // Next q-value in perspective of player to play at curr_state (important
    // note: exploits property of two-player zero-sum)
    const double next_q_value =
        (player != next_state->CurrentPlayer() ? -1 : 1) *
        GetBestActionValue(*(next_state.get()), min_utility);

    // Update action value
    std::string key = curr_state->ToString();
    double prev_q_val = values_[{key, curr_action}];
    double new_q_val =
        prev_q_val +
        learning_rate_ *
            (reward + discount_factor_ * next_q_value - prev_q_val);

    values_[{key, curr_action}] = new_q_val;
    curr_state = next_state->Clone();
  }
}
}  // namespace algorithms
}  // namespace open_spiel
