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

#include "open_spiel/algorithms/cfr_br.h"

#include "open_spiel/algorithms/cfr.h"
#include "open_spiel/policy.h"

namespace open_spiel {
namespace algorithms {

CFRBRSolver::CFRBRSolver(const Game& game)
    : CFRSolverBase(game,
                    /*alternating_updates=*/false,
                    /*linear_averaging=*/false,
                    /*regret_matching_plus=*/false),
      policy_overrides_(game.NumPlayers(), nullptr),
      uniform_policy_(UniformPolicy()) {
  InitializeBestResponseComputers();
}

CFRBRSolver::CFRBRSolver(std::shared_ptr<const Game> game, int iteration)
    : CFRSolverBase(game,
                    /*alternating_updates=*/false,
                    /*linear_averaging=*/false,
                    /*regret_matching_plus=*/false, iteration),
      policy_overrides_(game->NumPlayers(), nullptr),
      uniform_policy_(UniformPolicy()) {
  InitializeBestResponseComputers();
}

void CFRBRSolver::InitializeBestResponseComputers() {
  for (int p = 0; p < game_->NumPlayers(); ++p) {
    best_response_computers_.push_back(std::unique_ptr<TabularBestResponse>(
        new TabularBestResponse(*game_, p, &uniform_policy_)));
  }
}

void CFRBRSolver::EvaluateAndUpdatePolicy() {
  ++iteration_;

  std::vector<TabularPolicy> br_policies(game_->NumPlayers());
  std::shared_ptr<Policy> current_policy = CurrentPolicy();

  // Set all the player's policies first.
  for (int p = 0; p < game_->NumPlayers(); ++p) {
    // Need to have an exception here because the CFR policy objects are
    // wrappers around information that is contained in a table, and those do
    // not exist until there's been a tree traversal to compute regrets below.
    if (iteration_ > 1) {
      best_response_computers_[p]->SetPolicy(current_policy.get());
    }
  }

  // Now, for each player compute a best response
  for (int p = 0; p < game_->NumPlayers(); ++p) {
    br_policies[p] = best_response_computers_[p]->GetBestResponsePolicy();
  }

  for (int p = 0; p < game_->NumPlayers(); ++p) {
    // Override every player except p.
    for (int opp = 0; opp < game_->NumPlayers(); ++opp) {
      policy_overrides_[opp] = (opp == p ? nullptr : &br_policies[opp]);
    }

    // Then collect regret and update p's average strategy.
    ComputeCounterFactualRegret(*root_state_, p, root_reach_probs_,
                                &policy_overrides_);
  }
  ApplyRegretMatching();
}

std::unique_ptr<CFRBRSolver> DeserializeCFRBRSolver(
    const std::string& serialized, std::string delimiter) {
  auto partial = PartiallyDeserializeCFRSolver(serialized);
  SPIEL_CHECK_EQ(partial.solver_type, "CFRBRSolver");
  auto solver = std::make_unique<CFRBRSolver>(
      partial.game, std::stoi(partial.solver_specific_state));
  DeserializeCFRInfoStateValuesTable(partial.serialized_cfr_values_table,
                                     &solver->InfoStateValuesTable(),
                                     delimiter);
  return solver;
}

}  // namespace algorithms
}  // namespace open_spiel
