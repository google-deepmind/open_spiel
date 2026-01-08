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

#include "open_spiel/algorithms/value_iteration.h"

#include <algorithm>

#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace algorithms {
namespace {

using std::map;
using std::vector;
using state_pointer = std::unique_ptr<State>;
using state_action = std::pair<std::string, Action>;
using state_prob = std::pair<std::string, double>;

// Adds transitions and transition probability from a given state
void AddTransition(map<state_action, vector<state_prob>>* transitions,
                   std::string key, const state_pointer& state) {
  for (auto action : state->LegalActions()) {
    auto next_state = state->Clone();
    next_state->ApplyAction(action);
    vector<state_prob> possibilities;
    if (next_state->IsChanceNode()) {
      // For a chance node, record the transition probabilities
      for (const auto& actionprob : next_state->ChanceOutcomes()) {
        auto realized_next_state = next_state->Clone();
        realized_next_state->ApplyAction(actionprob.first);
        possibilities.emplace_back(realized_next_state->ToString(),
                                   actionprob.second);
      }
    } else {
      // A non-chance node is equivalent to transition with probability 1
      possibilities.emplace_back(next_state->ToString(), 1.0);
    }
    (*transitions)[std::make_pair(key, action)] = possibilities;
  }
}

// Initialize transition map and value map
void InitializeMaps(const map<std::string, state_pointer>& states,
                    map<std::string, double>* values,
                    map<state_action, vector<state_prob>>* transitions) {
  for (const auto& kv : states) {
    auto key = kv.first;
    if (kv.second->IsTerminal()) {
      // For both 1-player and 2-player zero sum games, suffices to look at
      // player 0's utility
      (*values)[key] = kv.second->PlayerReturn(Player{0});
    } else {
      (*values)[key] = 0;
      AddTransition(transitions, key, kv.second);
    }
  }
}

}  // namespace

std::map<std::string, double> ValueIteration(const Game& game, int depth_limit,
                                             double threshold) {
  using state_action = std::pair<std::string, Action>;
  using state_prob = std::pair<std::string, double>;

  // Currently only supports 1-player or 2-player zero sum games
  SPIEL_CHECK_TRUE(game.NumPlayers() == 1 || game.NumPlayers() == 2);
  if (game.NumPlayers() == 2) {
    SPIEL_CHECK_EQ(game.GetType().utility, GameType::Utility::kZeroSum);
  }

  // No support for simultaneous games (needs an LP solver). And so also must
  // be a perfect information game.
  SPIEL_CHECK_EQ(game.GetType().dynamics, GameType::Dynamics::kSequential);
  SPIEL_CHECK_EQ(game.GetType().information,
                 GameType::Information::kPerfectInformation);

  auto states = GetAllStates(game, depth_limit, /*include_terminals=*/true,
                             /*include_chance_states=*/false,
                             /*stop_at_duplicates*/true);
  std::map<std::string, double> values;
  std::map<state_action, std::vector<state_prob>> transitions;

  InitializeMaps(states, &values, &transitions);

  double error;
  double min_utility = game.MinUtility();
  double max_utility = game.MaxUtility();
  do {
    error = 0;
    for (const auto& kv : states) {
      auto key = kv.first;

      if (kv.second->IsTerminal()) continue;

      auto player = kv.second->CurrentPlayer();

      // Initialize value to be the minimum utility if current player
      // is the maximizing player (i.e. player 0), and to maximum utility
      // if current player is the minimizing player (i.e. player 1).
      double value = (player == Player{0}) ? min_utility : max_utility;
      for (auto action : kv.second->LegalActions()) {
        auto possibilities = transitions[std::make_pair(key, action)];
        double q_value = 0;
        for (const auto& outcome : possibilities) {
          q_value += outcome.second * values[outcome.first];
        }
        // Player 0 is maximizing the value (which is w.r.t. player 0)
        // Player 1 is minimizing the value
        if (player == Player{0})
          value = std::max(value, q_value);
        else
          value = std::min(value, q_value);
      }

      double* stored_value = &values[key];
      error = std::max(std::abs(*stored_value - value), error);
      *stored_value = value;
    }
  } while (error > threshold);

  return values;
}

}  // namespace algorithms
}  // namespace open_spiel
