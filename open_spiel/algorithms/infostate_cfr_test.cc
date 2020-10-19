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

#include "open_spiel/algorithms/infostate_cfr.h"

#include <cmath>
#include <iostream>

#include "open_spiel/algorithms/expected_returns.h"
#include "open_spiel/algorithms/tabular_exploitability.h"
#include "open_spiel/game_transforms/turn_based_simultaneous_game.h"
#include "open_spiel/games/goofspiel.h"
#include "open_spiel/games/kuhn_poker.h"
#include "open_spiel/games/leduc_poker.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace algorithms {
namespace {

double RootExpectedReturns(const State& root, const Policy& policy) {
  const std::vector<double> values =
      ExpectedReturns(root, policy,
          /*depth_limit=*/-1, /*use_infostate_get_policy=*/false);
  return values[0];
}

void CheckNashKuhnPoker(const Game& game, const Policy& policy) {
  const std::vector<double> game_value =
      ExpectedReturns(*game.NewInitialState(), policy,
          /*depth_limit=*/-1, /*use_infostate_get_policy=*/false);

  // 1/18 is the Nash value. See https://en.wikipedia.org/wiki/Kuhn_poker
  constexpr float nash_value = 1.0 / 18.0;
  constexpr float eps = 1e-3;

  SPIEL_CHECK_EQ(2, game_value.size());
  SPIEL_CHECK_FLOAT_NEAR((float)game_value[0], -nash_value, eps);
  SPIEL_CHECK_FLOAT_NEAR((float)game_value[1], nash_value, eps);
}

void CheckExploitabilityKuhnPoker(const Game& game, const Policy& policy) {
  SPIEL_CHECK_LE(Exploitability(game, policy), 0.05);
}

void CheckReturnsMatchingPennies(const Game& game, const Policy& policy) {
    const std::vector<double> game_value =
        ExpectedReturns(*game.NewInitialState(), policy,
            /*depth_limit=*/-1, /*use_infostate_get_policy=*/false);
    SPIEL_CHECK_EQ(game_value[0], 0.);
    SPIEL_CHECK_EQ(game_value[1], 0.);
}

void CFRTest_MatchingPennies() {
  std::shared_ptr<const Game> game = LoadGame("matrix_mp");
  InfostateCFR solver(*game);
  const std::shared_ptr<Policy> average_policy = solver.AveragePolicy();
  CheckReturnsMatchingPennies(*game, *average_policy);

  // Running iterations should not change the policy,
  // as uniform is already an equilibrium.
  solver.RunAlternatingIterations(10);
  CheckReturnsMatchingPennies(*game, *average_policy);

  solver.RunSimultaneousIterations(10);
  CheckReturnsMatchingPennies(*game, *average_policy);
}

void CFRTest_KuhnPoker() {
  std::shared_ptr<const Game> game = LoadGame("kuhn_poker");
  InfostateCFR solver(*game);
  const std::shared_ptr<Policy> average_policy = solver.AveragePolicy();
  solver.RunSimultaneousIterations(300);
  CheckNashKuhnPoker(*game, *average_policy);
  CheckExploitabilityKuhnPoker(*game, *average_policy);
}

void CFRTest_IIGoof4() {
  // Random points order.
  std::shared_ptr<const Game> game = LoadGame(
      "goofspiel", {{"imp_info", GameParameter(true)},
                    {"points_order", GameParameter(std::string("random"))},
                    {"num_cards", GameParameter(4)}});

  InfostateCFR solver(*game);
  solver.RunAlternatingIterations(100);

  // Values checked with Marc's thesis implementation.
  const std::shared_ptr<Policy> average_policy = solver.AveragePolicy();
  SPIEL_CHECK_LE(
      RootExpectedReturns(*game->NewInitialState(), *average_policy), 0.1);

  // Fixed points order.
  game  = LoadGame(
      "goofspiel", {{"imp_info", GameParameter(true)},
                    {"points_order", GameParameter(std::string("descending"))},
                    {"num_cards", GameParameter(4)}});

  InfostateCFR solver2(*game);
  solver2.RunAlternatingIterations(1000);

  // Values checkes with Marc's thesis implementation.
  const std::shared_ptr<Policy> average_policy2 = solver2.AveragePolicy();
  SPIEL_CHECK_LE(
      RootExpectedReturns(*game->NewInitialState(), *average_policy), 0.01);
}

void TestImplementationsHaveSameIterations() {
  std::shared_ptr<const Game> game = LoadGame("kuhn_poker");
  const int cfr_iterations = 10;
  InfostateCFR vec_solver(*game);
  // Use simultaneous updates.
  CFRSolverBase str_solver(*game, /*alternating_updates=*/false,
                                  /*linear_averaging=*/false,
                                  /*regret_matching_plus=*/false);

  std::unordered_map<std::string, const CFRInfoStateValues*> vec_ptable =
      vec_solver.InfoStateValuesPtrTable();
  std::unordered_map<std::string, CFRInfoStateValues>& str_table =
      str_solver.InfoStateValuesTable();
  SPIEL_CHECK_EQ(vec_ptable.size(), str_table.size());

  for (int i = 0; i < cfr_iterations; ++i) {
    str_solver.EvaluateAndUpdatePolicy();
    vec_solver.RunSimultaneousIterations(1);

    for (const auto& [infostate, str_values] : str_table) {
      const CFRInfoStateValues& vec_values = *(vec_ptable.at(infostate));
      SPIEL_CHECK_EQ(str_values.num_actions(), vec_values.num_actions());

      // Check regrets.
      for (int j = 0; j < vec_values.num_actions(); ++j) {
        SPIEL_CHECK_TRUE(fabs(vec_values.cumulative_regrets[j]
                              - str_values.cumulative_regrets[j]) < 1e-6);
      }
      // Cumulative policy is more tricky: we need to normalize it first.
      double str_cumul_sum = 0, vec_cumul_sum = 0;
      for (int j = 0; j < vec_values.num_actions(); ++j) {
        str_cumul_sum += str_values.cumulative_policy[j];
        vec_cumul_sum += vec_values.cumulative_policy[j];
      }
      for (int j = 0; j < vec_values.num_actions(); ++j) {
        SPIEL_CHECK_TRUE(fabs(
            vec_values.cumulative_policy[j] / vec_cumul_sum
            - str_values.cumulative_policy[j] / str_cumul_sum) < 1e-6);
      }
    }
  }
}

double Benchmark(int repetitions, std::function<void()> fn) {
  const absl::Time start = absl::Now();
  for (int i = 0; i < repetitions; ++i) { fn(); }
  const absl::Time end = absl::Now();
  const double seconds = absl::ToDoubleSeconds(end - start);
  return seconds / (double) repetitions;
}

void BenchmarkImplementations(const std::string& game_name) {
  const int cfr_iterations = 100;
  const int benchmark_reps = 10;
  std::shared_ptr<const Game> game = LoadGame(game_name);

  double vec_make_time, vec_run_time,
         str_make_time, str_run_time;
  {
    InfostateCFR solver(*game);
    auto make_tree = [&]() { InfostateCFR construct_tree(*game); };
    auto run_solver =
        [&]() { solver.RunSimultaneousIterations(cfr_iterations); };
    vec_make_time = Benchmark(benchmark_reps, make_tree);
    vec_run_time = Benchmark(benchmark_reps, run_solver);
  }

  // String implementation does not support simultaneous move games.
  if (game->GetType().dynamics == GameType::Dynamics::kSimultaneous) {
    game = ConvertToTurnBased(*game);
  }

  {
    CFRSolver solver(*game);
    auto make_tree = [&]() { InfostateCFR construct_tree(*game); };
    auto run_solver = [&]() {
      for (int i = 0; i < cfr_iterations; i++) {
        solver.EvaluateAndUpdatePolicy();
      }
    };
    str_make_time = Benchmark(benchmark_reps, make_tree);
    str_run_time = Benchmark(benchmark_reps, run_solver);
  }

  std::cout << "Game:      " << game_name << "\n"
            << "           Vec  Str  Speedup\n"
            << "Start:     " << vec_make_time << "  " <<  str_make_time << "  "
                             << str_make_time / vec_make_time << "\n"
            << "CFR iters: " << vec_run_time << "  " <<  str_run_time << "  "
                             << str_run_time / vec_run_time << "\n\n";
}

}  // namespace
}  // namespace algorithms
}  // namespace open_spiel

namespace algorithms = open_spiel::algorithms;

int main(int argc, char** argv) {
  algorithms::CFRTest_MatchingPennies();
  algorithms::CFRTest_KuhnPoker();
  algorithms::CFRTest_IIGoof4();
  algorithms::TestImplementationsHaveSameIterations();

  // These are disabled, as they are not tests.
  // Useful for future reference. Compiled using BUILD_TYPE=Release

  //             Vec         Str         Speedup
  //  Start:     0.00137985  0.00143035  1.0366
  //  CFR iters: 0.00199572  0.02147450  10.7603
//  algorithms::BenchmarkImplementations("kuhn_poker");

  //             Vec         Str         Speedup
  //  Start:     0.509767    0.333749    0.654708
  //  CFR iters: 1.304290    4.962850    3.80503
//  algorithms::BenchmarkImplementations("leduc_poker");

  //             Vec         Str         Speedup
  //  Start:     0.913013     1.60385    1.75666
  //  CFR iters: 0.800187    15.95760    19.9423
//  algorithms::BenchmarkImplementations("goofspiel(num_cards=4,imp_info=True)");
}
