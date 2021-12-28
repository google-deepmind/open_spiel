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

#include "open_spiel/algorithms/external_sampling_mccfr.h"

#include <cmath>
#include <iostream>
#include <random>

#include "open_spiel/algorithms/tabular_exploitability.h"
#include "open_spiel/games/kuhn_poker.h"
#include "open_spiel/games/leduc_poker.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace algorithms {
namespace {

constexpr int kSeed = 230398247;

void MCCFR_2PGameTest(const std::string& game_name, std::mt19937* rng,
                      int iterations, double nashconv_upperbound) {
  std::shared_ptr<const Game> game = LoadGame(game_name);
  ExternalSamplingMCCFRSolver solver(*game);
  for (int i = 0; i < iterations; i++) {
    solver.RunIteration(rng);
  }
  const std::shared_ptr<Policy> average_policy = solver.AveragePolicy();
  double nash_conv = NashConv(*game, *average_policy, true);
  std::cout << "Game: " << game_name << ", iters = " << iterations
            << ", NashConv: " << nash_conv << std::endl;
  SPIEL_CHECK_LE(nash_conv, nashconv_upperbound);
}

void MCCFR_KuhnPoker3PTest(std::mt19937* rng) {
  std::shared_ptr<const Game> game = LoadGame("kuhn_poker(players=3)");
  ExternalSamplingMCCFRSolver solver(*game);
  for (int i = 0; i < 100; i++) {
    solver.RunIteration(rng);
  }
  const std::shared_ptr<Policy> average_policy = solver.AveragePolicy();
  std::cout << "Kuhn 3P (standard averaging) NashConv = "
            << NashConv(*game, *average_policy, true) << std::endl;

  ExternalSamplingMCCFRSolver full_solver(*game, 39693847, AverageType::kFull);
  for (int i = 0; i < 100; i++) {
    full_solver.RunIteration(rng);
  }
  auto full_average_policy = full_solver.AveragePolicy();
  std::cout << "Kuhn 3P (full averaging) NashConv = "
            << NashConv(*game, *full_average_policy) << std::endl;
}

void MCCFR_SerializationTest() {
  auto game = LoadGame("kuhn_poker");
  ExternalSamplingMCCFRSolver solver = ExternalSamplingMCCFRSolver(*game);
  double exploitability0 = Exploitability(*game, *solver.AveragePolicy());

  for (int i = 0; i < 200; i++) {
    solver.RunIteration();
  }
  double exploitability1 = Exploitability(*game, *solver.AveragePolicy());
  SPIEL_CHECK_GT(exploitability0, exploitability1);

  std::string serialized = solver.Serialize();
  std::unique_ptr<ExternalSamplingMCCFRSolver> deserialized_solver =
      DeserializeExternalSamplingMCCFRSolver(serialized);
  SPIEL_CHECK_EQ(solver.InfoStateValuesTable().size(),
                 deserialized_solver->InfoStateValuesTable().size());
  double exploitability2 =
      Exploitability(*game, *deserialized_solver->AveragePolicy());
  SPIEL_CHECK_FLOAT_NEAR(exploitability1, exploitability2, 1e-15);

  for (int i = 0; i < 200; i++) {
    deserialized_solver->RunIteration();
  }
  double exploitability3 =
      Exploitability(*game, *deserialized_solver->AveragePolicy());
  SPIEL_CHECK_GT(exploitability2, exploitability3);
}

}  // namespace
}  // namespace algorithms
}  // namespace open_spiel

namespace algorithms = open_spiel::algorithms;

int main(int argc, char** argv) {
  // Values double-checked with the original implementation used in (Lanctot,
  // "Monte Carlo Sampling and Regret Minimization For Equilibrium Computation
  // and Decision-Making in Large Extensive Form Games", 2013).
  std::mt19937 rng(algorithms::kSeed);
  algorithms::MCCFR_2PGameTest("kuhn_poker", &rng, 1000, 0.05);
  algorithms::MCCFR_2PGameTest("leduc_poker", &rng, 1000, 2.5);
  algorithms::MCCFR_2PGameTest("liars_dice", &rng, 100, 1.6);
  algorithms::MCCFR_KuhnPoker3PTest(&rng);
  algorithms::MCCFR_SerializationTest();
}
