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

#include "open_spiel/algorithms/corr_dev_builder.h"

#include <iostream>
#include <memory>

#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/abseil-cpp/absl/strings/str_format.h"
#include "open_spiel/abseil-cpp/absl/types/optional.h"
#include "open_spiel/algorithms/cfr.h"
#include "open_spiel/algorithms/corr_dist.h"
#include "open_spiel/algorithms/deterministic_policy.h"
#include "open_spiel/algorithms/expected_returns.h"
#include "open_spiel/game_transforms/turn_based_simultaneous_game.h"
#include "open_spiel/games/efg_game.h"
#include "open_spiel/policy.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace algorithms {
namespace {
inline constexpr int kSeed = 23894982;

TabularPolicy MergeIndependentPolicies(const TabularPolicy& policy1,
                                       const TabularPolicy& policy2) {
  TabularPolicy merged_policy;
  for (const auto& infostate_and_state_policy : policy1.PolicyTable()) {
    merged_policy.SetStatePolicy(infostate_and_state_policy.first,
                                 infostate_and_state_policy.second);
  }
  for (const auto& infostate_and_state_policy : policy2.PolicyTable()) {
    merged_policy.SetStatePolicy(infostate_and_state_policy.first,
                                 infostate_and_state_policy.second);
  }
  return merged_policy;
}

void BasicCorrDevBuilderTest() {
  // Build a uniform correlation device for Kuhn poker.
  std::shared_ptr<const Game> game = LoadGame("kuhn_poker");
  CorrDevBuilder full_cd_builder;

  DeterministicTabularPolicy p1_policy(*game, 0);
  DeterministicTabularPolicy p2_policy(*game, 1);
  do {
    do {
      full_cd_builder.AddDeterminsticJointPolicy(MergeIndependentPolicies(
          p1_policy.GetTabularPolicy(), p2_policy.GetTabularPolicy()));
    } while (p2_policy.NextPolicy());
    p2_policy.ResetDefaultPolicy();
  } while (p1_policy.NextPolicy());

  CorrelationDevice mu = full_cd_builder.GetCorrelationDevice();
  SPIEL_CHECK_EQ(mu.size(), 64 * 64);
  for (const auto& prob_and_policy : mu) {
    SPIEL_CHECK_FLOAT_NEAR(prob_and_policy.first, 1.0 / (64 * 64), 1e-10);
  }

  std::vector<double> uniform_returns =
      ExpectedReturns(*game->NewInitialState(), GetUniformPolicy(*game), -1);
  std::vector<double> corr_dev_uniform_returns = ExpectedValues(*game, mu);
  for (Player p = 0; p < game->NumPlayers(); ++p) {
    SPIEL_CHECK_FLOAT_NEAR(uniform_returns[p], corr_dev_uniform_returns[p],
                           1e-10);
  }
}

void BasicSamplingCorrDevBuilderTest() {
  std::shared_ptr<const Game> game = LoadGame("kuhn_poker");
  CorrDevBuilder cd_builder;
  TabularPolicy uniform_policy = GetUniformPolicy(*game);
  for (int i = 0; i < 10; ++i) {
    cd_builder.AddSampledJointPolicy(uniform_policy, 1000);
  }
  CorrelationDevice mu = cd_builder.GetCorrelationDevice();
  SPIEL_CHECK_LE(mu.size(), 64 * 64);
}

void CFRShapleysCorrDistTest() {
  std::shared_ptr<const Game> game =
      LoadGame("turn_based_simultaneous_game(game=matrix_shapleys_game())");
  CorrDevBuilder cd_builder;
  CFRSolverBase solver(*game,
                       /*alternating_updates=*/true,
                       /*linear_averaging=*/false,
                       /*regret_matching_plus=*/false,
                       /*random_initial_regrets*/ true,
                       /*seed*/ kSeed);
  CorrDistConfig config;
  for (int i = 0; i < 100; i++) {
    solver.EvaluateAndUpdatePolicy();
    TabularPolicy current_policy =
        static_cast<CFRCurrentPolicy*>(solver.CurrentPolicy().get())
            ->AsTabular();
    cd_builder.AddMixedJointPolicy(current_policy);
    if (i % 10 == 0) {
      CorrelationDevice mu = cd_builder.GetCorrelationDevice();
      double afcce_dist = AFCCEDist(*game, config, mu);
      double afce_dist = AFCEDist(*game, config, mu);
      double efcce_dist = EFCCEDist(*game, config, mu);
      double efce_dist = EFCEDist(*game, config, mu);
      std::vector<double> values = ExpectedValues(*game, mu);
      std::cout
          << absl::StrFormat(
                 "CFRTest %d %2.10lf %2.10lf %2.10lf %2.10lf %2.3lf %2.3lf", i,
                 afcce_dist, afce_dist, efcce_dist, efce_dist, values[0],
                 values[1])
          << std::endl;
    }
  }

  CorrelationDevice mu = cd_builder.GetCorrelationDevice();
  std::cout << ToString(mu) << std::endl;
}

void CFRGoofspielCorrDistTest() {
  std::shared_ptr<const Game> game = LoadGame(
      "turn_based_simultaneous_game(game=goofspiel(num_cards=3,points_order="
      "descending,returns_type=total_points))");
  CorrDevBuilder cd_builder;
  CFRSolverBase solver(*game,
                       /*alternating_updates=*/true,
                       /*linear_averaging=*/false,
                       /*regret_matching_plus=*/false,
                       /*random_initial_regrets*/ true,
                       /*seed*/ kSeed);
  CorrDistConfig config;
  for (int i = 0; i < 10; i++) {
    solver.EvaluateAndUpdatePolicy();
    TabularPolicy current_policy =
        static_cast<CFRCurrentPolicy*>(solver.CurrentPolicy().get())
            ->AsTabular();
    cd_builder.AddSampledJointPolicy(current_policy, 100);
  }
  CorrelationDevice mu = cd_builder.GetCorrelationDevice();
  double afcce_dist = AFCCEDist(*game, config, mu);
  double afce_dist = AFCEDist(*game, config, mu);
  double efcce_dist = EFCCEDist(*game, config, mu);
  double efce_dist = EFCEDist(*game, config, mu);
  std::vector<double> values = ExpectedValues(*game, mu);
  std::cout << absl::StrFormat(
                   "CFRTest %2.10lf %2.10lf %2.10lf, %2.10lf %2.3lf %2.3lf",
                   afcce_dist, afce_dist, efcce_dist, efce_dist, values[0],
                   values[1])
            << std::endl;
}

}  // namespace
}  // namespace algorithms
}  // namespace open_spiel

namespace algorithms = open_spiel::algorithms;

int main(int argc, char** argv) {
  algorithms::BasicCorrDevBuilderTest();
  algorithms::BasicSamplingCorrDevBuilderTest();
  algorithms::CFRShapleysCorrDistTest();
  algorithms::CFRGoofspielCorrDistTest();
}
