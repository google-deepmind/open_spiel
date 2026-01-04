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

#include "open_spiel/algorithms/policy_iteration.h"

#include <algorithm>

#include "open_spiel/abseil-cpp/absl/container/flat_hash_map.h"
#include "open_spiel/spiel_globals.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace algorithms {
namespace {

using std::vector;

struct MDPState {
  std::unique_ptr<State> state;  // The state of the MDP
  double value;                  // The value of this state
  absl::flat_hash_map<Action, std::vector<std::pair<std::string, double>>>
      transitions;  // The transitions from this state, for each action, with
                    // the correponding probability
  Action optimal_action;  // The optimal action from this state
};

// Adds transitions and transition probability from a given state
void AddTransition(
    absl::flat_hash_map<Action, vector<std::pair<std::string, double>>>*
        transitions,
    const std::string& key, const std::unique_ptr<State>& state) {
  for (Action action : state->LegalActions()) {
    std::unique_ptr<State> next_state = state->Child(action);
    vector<std::pair<std::string, double>> possibilities;
    if (next_state->IsChanceNode()) {
      // For a chance node, record the transition probabilities
      for (const auto& actionprob : next_state->ChanceOutcomes()) {
        std::unique_ptr<State> realized_next_state = next_state->Child(action);
        possibilities.emplace_back(realized_next_state->ToString(),
                                   actionprob.second);
      }
    } else {
      // A non-chance node is equivalent to transition with probability 1
      possibilities.emplace_back(next_state->ToString(), 1.0);
    }
    (*transitions)[action] = possibilities;
  }
}

// Initialize transition map and value map
void InitializeMaps(
    const std::map<std::string, std::unique_ptr<State>>& states,
    absl::flat_hash_map<std::string, MDPState>* mdp_state_nodes) {
  for (const auto& kv : states) {
    const std::string& key = kv.first;
    if (kv.second->IsTerminal()) {
      // For both 1-player and 2-player zero sum games, suffices to look at
      // player 0's utility
      (*mdp_state_nodes)[key].value = kv.second->PlayerReturn(Player{0});
      // No action possible from a terminal state.
      (*mdp_state_nodes)[key].optimal_action = kInvalidAction;
    } else {
      absl::flat_hash_map<Action, std::vector<std::pair<std::string, double>>>&
          transitions = (*mdp_state_nodes)[key].transitions;
      AddTransition(&transitions, key, kv.second);
      (*mdp_state_nodes)[key].value = 0;
      // Assign any random action as the optimal action, initially.
      (*mdp_state_nodes)[key].optimal_action = kv.second->LegalActions()[0];
    }
  }
}

double QValue(const absl::flat_hash_map<std::string, MDPState>& mdp_state_nodes,
              const std::unique_ptr<State>& state, const Action& action) {
  if (!mdp_state_nodes.contains(state->ToString()) ||
      !mdp_state_nodes.at(state->ToString()).transitions.contains(action)) {
    // This action is not possible from this state.
    return 0;
  }

  double value = 0;
  const vector<std::pair<std::string, double>>& possibilities =
      mdp_state_nodes.at(state->ToString()).transitions.at(action);
  for (const auto& outcome : possibilities) {
    if (mdp_state_nodes.contains(outcome.first)) {
      value += outcome.second * mdp_state_nodes.at(outcome.first).value;
    }
  }
  return value;
}

// Given a player and a state, gets the best possible action from this state
Action GetBestAction(
    const absl::flat_hash_map<std::string, MDPState>& mdp_state_nodes,
    const std::unique_ptr<State>& state, const Player& player,
    const double& min_utility, const double& max_utility) {
  vector<Action> legal_actions = state->LegalActions();
  Action optimal_action = kInvalidAction;

  // Initialize value to be the minimum utility if current player
  // is the maximizing player (i.e. player 0), and to maximum utility
  // if current player is the minimizing player (i.e. player 1).
  double value = (player == Player{0}) ? min_utility : max_utility;
  for (Action action : legal_actions) {
    double q_val = QValue(mdp_state_nodes, state, action);
    bool is_best_so_far = (player == Player{0} && q_val >= value) ||
                          (player == Player{1} && q_val <= value);
    if (is_best_so_far) {
      value = q_val;
      optimal_action = action;
    }
  }
  return optimal_action;
}

}  // namespace

absl::flat_hash_map<std::string, double> PolicyIteration(const Game& game,
                                                         int depth_limit,
                                                         double threshold) {
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

  std::map<std::string, std::unique_ptr<State>> states =
      GetAllStates(game, depth_limit, /*include_terminals=*/true,
                   /*include_chance_states=*/false);
  absl::flat_hash_map<std::string, MDPState> mdp_state_nodes;

  InitializeMaps(states, &mdp_state_nodes);

  bool policy_stable;
  do {
    // Policy evaluation done in place
    double error;
    do {
      error = 0;
      for (const auto& kv : states) {
        const std::string& key = kv.first;
        if (kv.second->IsTerminal()) continue;

        // Evaluate the state value function
        Action curr_optimal_action = mdp_state_nodes.at(key).optimal_action;
        double value = QValue(mdp_state_nodes, kv.second, curr_optimal_action);

        double* stored_value = &mdp_state_nodes.at(key).value;
        error = std::max(std::abs(*stored_value - value), error);
        *stored_value = value;
      }
    } while (error > threshold);

    // Policy improvement
    double min_utility = game.MinUtility();
    double max_utility = game.MaxUtility();
    policy_stable = true;
    for (const auto& kv : states) {
      const std::string& key = kv.first;
      if (kv.second->IsTerminal()) continue;

      Player player = kv.second->CurrentPlayer();

      // Choose the action with the highest possible action value function
      Action curr_optimal_action = GetBestAction(
          mdp_state_nodes, kv.second, player, min_utility, max_utility);

      double curr_value =
          QValue(mdp_state_nodes, kv.second, curr_optimal_action);

      double* stored_value = &mdp_state_nodes.at(key).value;
      Action* stored_optimal_action = &mdp_state_nodes.at(key).optimal_action;
      if (std::abs(*stored_value - curr_value) > threshold) {
        policy_stable = false;
        *stored_optimal_action = curr_optimal_action;
      }
    }
  } while (!policy_stable);

  absl::flat_hash_map<std::string, double> values;
  for (const auto& kv : states) {
    std::string state_string = kv.first;
    values[state_string] = mdp_state_nodes[state_string].value;
  }
  return values;
}
}  // namespace algorithms
}  // namespace open_spiel
