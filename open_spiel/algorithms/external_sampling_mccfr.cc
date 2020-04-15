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

#include "open_spiel/algorithms/external_sampling_mccfr.h"

#include <memory>
#include <numeric>
#include <random>

#include "open_spiel/algorithms/cfr.h"
#include "open_spiel/policy.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace algorithms {

ExternalSamplingMCCFRSolver::ExternalSamplingMCCFRSolver(const Game& game,
                                                         int seed,
                                                         AverageType avg_type)
    : ExternalSamplingMCCFRSolver(game, std::make_shared<UniformPolicy>(), seed,
                                  avg_type) {}

ExternalSamplingMCCFRSolver::ExternalSamplingMCCFRSolver(
    const Game& game, std::shared_ptr<Policy> default_policy, int seed,
    AverageType avg_type)
    : game_(game.Clone()),
      rng_(new std::mt19937(seed)),
      avg_type_(avg_type),
      dist_(0.0, 1.0),
      default_policy_(default_policy) {
  if (game_->GetType().dynamics != GameType::Dynamics::kSequential) {
    SpielFatalError(
        "MCCFR requires sequential games. If you're trying to run it "
        "on a simultaneous (or normal-form) game, please first transform it "
        "using turn_based_simultaneous_game.");
  }
}

void ExternalSamplingMCCFRSolver::RunIteration() { RunIteration(rng_.get()); }

void ExternalSamplingMCCFRSolver::RunIteration(std::mt19937* rng) {
  for (auto p = Player{0}; p < game_->NumPlayers(); ++p) {
    UpdateRegrets(*game_->NewInitialState(), p, rng);
  }

  if (avg_type_ == AverageType::kFull) {
    std::vector<double> reach_probs(game_->NumPlayers(), 1.0);
    FullUpdateAverage(*game_->NewInitialState(), reach_probs);
  }
}

double ExternalSamplingMCCFRSolver::UpdateRegrets(const State& state,
                                                  Player player,
                                                  std::mt19937* rng) {
  if (state.IsTerminal()) {
    return state.PlayerReturn(player);
  } else if (state.IsChanceNode()) {
    Action action = SampleAction(state.ChanceOutcomes(), dist_(*rng)).first;
    return UpdateRegrets(*state.Child(action), player, rng);
  } else if (state.IsSimultaneousNode()) {
    SpielFatalError(
        "Simultaneous moves not supported. Use "
        "TurnBasedSimultaneousGame to convert the game first.");
  }

  Player cur_player = state.CurrentPlayer();
  std::string is_key = state.InformationStateString(cur_player);
  std::vector<Action> legal_actions = state.LegalActions();

  // The insert here only inserts the default value if the key is not found,
  // otherwise returns the entry in the map.
  auto iter_and_result = info_states_.insert(
      {is_key, CFRInfoStateValues(legal_actions, kInitialTableValues)});

  CFRInfoStateValues info_state_copy = iter_and_result.first->second;
  info_state_copy.ApplyRegretMatching();

  double value = 0;
  std::vector<double> child_values(legal_actions.size(), 0);

  if (cur_player != player) {
    // Sample at opponent nodes.
    int aidx = info_state_copy.SampleActionIndex(0.0, dist_(*rng));
    value = UpdateRegrets(*state.Child(legal_actions[aidx]), player, rng);
  } else {
    // Walk over all actions at my nodes
    for (int aidx = 0; aidx < legal_actions.size(); ++aidx) {
      child_values[aidx] =
          UpdateRegrets(*state.Child(legal_actions[aidx]), player, rng);
      value += info_state_copy.current_policy[aidx] * child_values[aidx];
    }
  }

  // Now the regret and avg strategy updates.
  CFRInfoStateValues& info_state = info_states_[is_key];

  if (cur_player == player) {
    // Update regrets
    for (int aidx = 0; aidx < legal_actions.size(); ++aidx) {
      info_state.cumulative_regrets[aidx] += (child_values[aidx] - value);
    }
  }

  // Simple average does averaging on the opponent node. To do this in a game
  // with more than two players, we only update the player + 1 mod num_players,
  // which reduces to the standard rule in 2 players.
  if (avg_type_ == AverageType::kSimple &&
      cur_player == ((player + 1) % game_->NumPlayers())) {
    for (int aidx = 0; aidx < legal_actions.size(); ++aidx) {
      info_state.cumulative_policy[aidx] +=
          info_state_copy.current_policy[aidx];
    }
  }

  return value;
}

void ExternalSamplingMCCFRSolver::FullUpdateAverage(
    const State& state, const std::vector<double>& reach_probs) {
  if (state.IsTerminal()) {
    return;
  } else if (state.IsChanceNode()) {
    for (Action action : state.LegalActions()) {
      FullUpdateAverage(*state.Child(action), reach_probs);
    }
    return;
  } else if (state.IsSimultaneousNode()) {
    SpielFatalError(
        "Simultaneous moves not supported. Use "
        "TurnBasedSimultaneousGame to convert the game first.");
  }

  // If all the probs are zero, no need to keep going.
  double sum = std::accumulate(reach_probs.begin(), reach_probs.end(), 0.0);
  if (sum == 0.0) return;

  Player cur_player = state.CurrentPlayer();
  std::string is_key = state.InformationStateString(cur_player);
  std::vector<Action> legal_actions = state.LegalActions();

  // The insert here only inserts the default value if the key is not found,
  // otherwise returns the entry in the map.
  auto iter_and_result = info_states_.insert(
      {is_key, CFRInfoStateValues(legal_actions, kInitialTableValues)});

  CFRInfoStateValues info_state_copy = iter_and_result.first->second;
  info_state_copy.ApplyRegretMatching();

  for (int aidx = 0; aidx < legal_actions.size(); ++aidx) {
    std::vector<double> new_reach_probs = reach_probs;
    new_reach_probs[cur_player] *= info_state_copy.current_policy[aidx];
    FullUpdateAverage(*state.Child(legal_actions[aidx]), new_reach_probs);
  }

  // Now update the cumulative policy.
  CFRInfoStateValues& info_state = info_states_[is_key];
  for (int aidx = 0; aidx < legal_actions.size(); ++aidx) {
    info_state.cumulative_policy[aidx] +=
        (reach_probs[cur_player] * info_state_copy.current_policy[aidx]);
  }
}

}  // namespace algorithms
}  // namespace open_spiel
