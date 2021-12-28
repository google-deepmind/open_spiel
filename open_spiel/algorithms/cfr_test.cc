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

#include "open_spiel/algorithms/cfr.h"

#include <cmath>
#include <iostream>

#include "open_spiel/algorithms/expected_returns.h"
#include "open_spiel/algorithms/history_tree.h"
#include "open_spiel/algorithms/tabular_exploitability.h"
#include "open_spiel/game_transforms/turn_based_simultaneous_game.h"
#include "open_spiel/games/kuhn_poker.h"
#include "open_spiel/games/leduc_poker.h"
#include "open_spiel/games/liars_dice.h"
#include "open_spiel/games/matching_pennies_3p.h"
#include "open_spiel/games/tic_tac_toe.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace algorithms {
namespace {

void CheckNashKuhnPoker(const Game& game, const Policy& policy) {
  const std::vector<double> game_value =
      ExpectedReturns(*game.NewInitialState(), policy, -1);

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

void CFRTest_KuhnPoker() {
  std::shared_ptr<const Game> game = LoadGame("kuhn_poker");
  CFRSolver solver(*game);
  for (int i = 0; i < 300; i++) {
    solver.EvaluateAndUpdatePolicy();
  }
  const std::shared_ptr<Policy> average_policy = solver.AveragePolicy();
  CheckNashKuhnPoker(*game, *average_policy);
  CheckExploitabilityKuhnPoker(*game, *average_policy);
}

void CFRTest_IIGoof4() {
  // Random points order.
  std::shared_ptr<const Game> game = LoadGameAsTurnBased(
      "goofspiel", {{"imp_info", GameParameter(true)},
                    {"points_order", GameParameter(std::string("random"))},
                    {"num_cards", GameParameter(4)}});

  CFRSolver solver(*game);
  for (int i = 0; i < 100; i++) {
    solver.EvaluateAndUpdatePolicy();
  }
  // Values checked with Marc's thesis implementation.
  const std::shared_ptr<Policy> average_policy = solver.AveragePolicy();
  SPIEL_CHECK_LE(Exploitability(*game, *average_policy), 0.1);

  // Fixed points order.
  game = LoadGameAsTurnBased(
      "goofspiel", {{"imp_info", GameParameter(true)},
                    {"points_order", GameParameter(std::string("descending"))},
                    {"num_cards", GameParameter(4)}});

  CFRSolver solver2(*game);
  for (int i = 0; i < 1000; i++) {
    solver2.EvaluateAndUpdatePolicy();
  }
  // Values checkes with Marc's thesis implementation.
  const std::shared_ptr<Policy> average_policy2 = solver2.AveragePolicy();
  SPIEL_CHECK_LE(Exploitability(*game, *average_policy2), 0.01);
}

void CFRPlusTest_KuhnPoker() {
  std::shared_ptr<const Game> game = LoadGame("kuhn_poker");
  CFRPlusSolver solver(*game);
  for (int i = 0; i < 200; i++) {
    solver.EvaluateAndUpdatePolicy();
  }
  const std::shared_ptr<Policy> average_policy = solver.AveragePolicy();
  CheckNashKuhnPoker(*game, *average_policy);
  CheckExploitabilityKuhnPoker(*game, *average_policy);
}

void CFRTest_KuhnPokerRunsWithThreePlayers(bool linear_averaging,
                                           bool regret_matching_plus,
                                           bool alternating_updates) {
  int num_players = 3;
  std::shared_ptr<const Game> game =
      LoadGame("kuhn_poker", {{"players", GameParameter(num_players)}});
  CFRSolverBase solver(*game, alternating_updates, linear_averaging,
                       regret_matching_plus);
  for (int i = 0; i < 10; i++) {
    solver.EvaluateAndUpdatePolicy();
  }
  std::shared_ptr<Policy> average_policy = solver.AveragePolicy();
  // Value upper-bounds inspired by Fig 2 of (Srinivasan et al., Actor-Critic
  // Policy Optimization in Partially Observable Multiagent Environments, 2018)
  // https://arxiv.org/abs/1810.09026
  SPIEL_CHECK_LE(NashConv(*game, *average_policy), 1);
}

// Tests the convergence of CFR in a specific game. Only computes nashconv
// if the upper bound is positive.
void CFRTest_GeneralMultiplePlayerTest(const std::string& game_name,
                                       int num_players, int num_iterations,
                                       double nashconv_upper_bound) {
  std::shared_ptr<const Game> game =
      LoadGame(game_name, {{"players", GameParameter(num_players)}});
  CFRSolverBase solver(*game,
                       /*alternating_updates=*/true,
                       /*linear_averaging=*/false,
                       /*regret_matching_plus=*/false);
  for (int i = 0; i < num_iterations; i++) {
    solver.EvaluateAndUpdatePolicy();
  }

  if (nashconv_upper_bound > 0) {
    std::shared_ptr<Policy> average_policy = solver.AveragePolicy();
    SPIEL_CHECK_LE(NashConv(*game, *average_policy), nashconv_upper_bound);
  }
}

void CFRTest_OneShotGameTest(int iterations, std::string one_shot_game,
                             double nashconv_upper_bound) {
  // Note: this is a 3-player general sum game with a unique uniform mixed
  // strategy. However, CFR is not guaranteed to converge, and indeed fails to,
  // just like was shown for fictitious play.
  std::cout << one_shot_game << " convergence test." << std::endl;
  std::shared_ptr<const Game> game = LoadGameAsTurnBased(one_shot_game);
  CFRSolverBase solver(*game,
                       /*alternating_updates=*/true,
                       /*linear_averaging=*/false,
                       /*regret_matching_plus=*/false);
  double nash_conv = 0;
  for (int i = 0; i < iterations; i++) {
    solver.EvaluateAndUpdatePolicy();
    if (i % 10 == 0) {
      std::shared_ptr<Policy> average_policy = solver.AveragePolicy();
      nash_conv = NashConv(*game, *average_policy);
      std::cout << "iter " << i << ", nashconv = " << nash_conv << std::endl;

      if (game->GetType().utility == GameType::Utility::kConstantSum ||
          game->GetType().utility == GameType::Utility::kZeroSum) {
        double expl = Exploitability(*game, *average_policy);
        SPIEL_CHECK_FLOAT_NEAR(expl, nash_conv / game->NumPlayers(), 1e-10);
      }
    }
  }
  SPIEL_CHECK_LE(nash_conv, nashconv_upper_bound);
}

// Tests the convergence of CFR in a specific game. Only computes nashconv
// if the upper bound is positive.
void CFRTest_TicTacToe(int num_iterations, double nashconv_upper_bound) {
  std::shared_ptr<const Game> game = LoadGame("tic_tac_toe");
  CFRSolverBase solver(*game,
                       /*alternating_updates=*/true,
                       /*linear_averaging=*/false,
                       /*regret_matching_plus=*/false);
  for (int i = 0; i < num_iterations; i++) {
    solver.EvaluateAndUpdatePolicy();
  }

  if (nashconv_upper_bound > 0) {
    std::shared_ptr<Policy> average_policy = solver.AveragePolicy();
    SPIEL_CHECK_LE(NashConv(*game, *average_policy), nashconv_upper_bound);
  }
}

void CFRTest_InfoStateValuesTableSerialization() {
  // Check empty
  CFRInfoStateValuesTable info_state_values_table = {};
  std::string serialized0 = "";
  SerializeCFRInfoStateValuesTable(info_state_values_table, &serialized0, -1);
  CFRInfoStateValuesTable deserialized0;
  DeserializeCFRInfoStateValuesTable(serialized0, &deserialized0);
  SPIEL_CHECK_TRUE(deserialized0.empty());

  // Check non-empty
  info_state_values_table = {
      {"", CFRInfoStateValues({0}, 1.0)},
      {"0:0,0;0", CFRInfoStateValues({0, 1, 2}, 0.1)},
      {"<->\n<->", CFRInfoStateValues({0, 1, 2}, 0.1)},
      {"1:1,1;1", CFRInfoStateValues({0, 1, 2, 3}, 0.2)}};
  std::string serialized1 = "";
  SerializeCFRInfoStateValuesTable(info_state_values_table, &serialized1, -1);
  CFRInfoStateValuesTable deserialized1;
  DeserializeCFRInfoStateValuesTable(serialized1, &deserialized1);

  SPIEL_CHECK_EQ(info_state_values_table.size(),
                 info_state_values_table.size());
  for (const auto& [info_state, values] : info_state_values_table) {
    for (int i = 0; i < values.legal_actions.size(); i++) {
      SPIEL_CHECK_EQ(values.legal_actions.at(i),
                     deserialized1.at(info_state).legal_actions.at(i));
      SPIEL_CHECK_FLOAT_NEAR(
          values.cumulative_regrets.at(i),
          deserialized1.at(info_state).cumulative_regrets.at(i), 1e-15);
      SPIEL_CHECK_FLOAT_NEAR(
          values.cumulative_policy.at(i),
          deserialized1.at(info_state).cumulative_policy.at(i), 1e-15);
      SPIEL_CHECK_FLOAT_NEAR(values.current_policy.at(i),
                             deserialized1.at(info_state).current_policy.at(i),
                             1e-15);
    }
  }
}

void CFRTest_CFRSolverSerialization() {
  auto game = LoadGame("kuhn_poker");
  CFRSolver solver = CFRSolver(*game);
  double exploitability0 = Exploitability(*game, *solver.AveragePolicy());

  for (int i = 0; i < 50; i++) {
    solver.EvaluateAndUpdatePolicy();
  }
  double exploitability1 = Exploitability(*game, *solver.AveragePolicy());
  SPIEL_CHECK_GT(exploitability0, exploitability1);

  std::string serialized = solver.Serialize();
  std::unique_ptr<CFRSolver> deserialized_solver =
      DeserializeCFRSolver(serialized);
  SPIEL_CHECK_EQ(solver.InfoStateValuesTable().size(),
                 deserialized_solver->InfoStateValuesTable().size());
  double exploitability2 =
      Exploitability(*game, *deserialized_solver->AveragePolicy());
  SPIEL_CHECK_FLOAT_NEAR(exploitability1, exploitability2, 1e-15);

  for (int i = 0; i < 50; i++) {
    deserialized_solver->EvaluateAndUpdatePolicy();
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
  algorithms::CFRTest_KuhnPoker();
  algorithms::CFRTest_IIGoof4();
  algorithms::CFRPlusTest_KuhnPoker();
  algorithms::CFRTest_KuhnPokerRunsWithThreePlayers(
      /*linear_averaging=*/false,
      /*regret_matching_plus=*/false,
      /*alternating_updates=*/false);
  algorithms::CFRTest_KuhnPokerRunsWithThreePlayers(
      /*linear_averaging=*/true,
      /*regret_matching_plus=*/false,
      /*alternating_updates=*/false);
  algorithms::CFRTest_KuhnPokerRunsWithThreePlayers(
      /*linear_averaging=*/true,
      /*regret_matching_plus=*/true,
      /*alternating_updates=*/false);
  algorithms::CFRTest_KuhnPokerRunsWithThreePlayers(
      /*linear_averaging=*/true,
      /*regret_matching_plus=*/true,
      /*alternating_updates=*/true);

  // Value upper bounds for Kuhn taken from Figure 2 of (Lanctot, Further
  // Developments of Extensive-Form Replicator Dynamics using the Sequence-Form
  // Representation, 2014).
  algorithms::CFRTest_GeneralMultiplePlayerTest(
      /*game_name=*/"kuhn_poker", /*num_players=*/3, /*num_iterations=*/10,
      /*nashconv_upper_bound=*/1.0);
  algorithms::CFRTest_GeneralMultiplePlayerTest(
      /*game_name=*/"kuhn_poker", /*num_players=*/4, /*num_iterations=*/10,
      /*nashconv_upper_bound=*/1.0);

  // Value upper-bounds for Leduc taken from Figure 17 of (Lanctot et al.,
  // A Unified Game-Theoretic Approach to Multiagent Reinforcement Learning,
  // 2017). See https://arxiv.org/abs/1711.00832. For Liar's Dice, number's were
  // obtained using Marc's thesis code.
  algorithms::CFRTest_GeneralMultiplePlayerTest(
      /*game_name=*/"leduc_poker", /*num_players=*/2, /*num_iterations=*/10,
      /*nashconv_upper_bound=*/2.0);

  // Disabled as they make the test too long for a unit test.
  // algorithms::CFRTest_GeneralMultiplePlayerTest(
  //    /*game_name=*/"liars_dice", /*num_players=*/2, /*num_iterations=*/10,
  //    /*nashconv_upper_bound=*/1.0);
  // algorithms::CFRTest_GeneralMultiplePlayerTest(
  //   /*game_name=*/"leduc_poker", /*num_players=*/3, /*num_iterations=*/2,
  //   /*nashconv_upper_bound=*/10.0);

  // Test a few one-shot games.
  algorithms::CFRTest_OneShotGameTest(1000, "matrix_rps", 1e-6);
  algorithms::CFRTest_OneShotGameTest(1000, "matrix_shapleys_game", 1.0);
  algorithms::CFRTest_OneShotGameTest(1000, "matching_pennies_3p", 3.0);

  // Try out a perfect information game that has an InformationState that obeys
  // perfect recall. Test is disabled because it adds 20 seconds. Let's revisit
  // when we add a version that can handle safe imperfect recall information
  // states.
  // algorithms::CFRTest_TicTacToe(10, 2.0);

  algorithms::CFRTest_InfoStateValuesTableSerialization();
  algorithms::CFRTest_CFRSolverSerialization();
}
