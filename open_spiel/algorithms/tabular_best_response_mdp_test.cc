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

#include "open_spiel/algorithms/tabular_best_response_mdp.h"

#include <memory>

#include "open_spiel/abseil-cpp/absl/types/optional.h"
#include "open_spiel/algorithms/cfr.h"
#include "open_spiel/algorithms/tabular_exploitability.h"
#include "open_spiel/policy.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace algorithms {
namespace {
constexpr double kFloatTolerance = 1e-12;

double NashConvTest(const std::string &game_string, const Policy &policy,
                    absl::optional<double> expected_nash_conv = absl::nullopt) {
  std::shared_ptr<const Game> game = LoadGame(game_string);
  TabularBestResponseMDP tbr(*game, policy);
  TabularBestResponseMDPInfo br_info = tbr.NashConv();
  if (expected_nash_conv.has_value()) {
    SPIEL_CHECK_FLOAT_NEAR(br_info.nash_conv, expected_nash_conv.value(),
                           kFloatTolerance);
  }
  return br_info.nash_conv;
}

void KuhnNashConvTests() {
  UniformPolicy uniform_policy;
  NashConvTest("kuhn_poker", uniform_policy, 0.916666666666667);
  FirstActionPolicy first_action_policy;
  NashConvTest("kuhn_poker", first_action_policy, 2.0);
}

void LeducNashConvTests() {
  UniformPolicy uniform_policy;
  NashConvTest("leduc_poker", uniform_policy, 4.747222222222222);
  FirstActionPolicy first_action_policy;
  NashConvTest("leduc_poker", first_action_policy, 2.0);
}

void KuhnLeduc3pTests() {
  UniformPolicy uniform_policy;
  NashConvTest("kuhn_poker(players=3)", uniform_policy, 2.0625);
  NashConvTest("leduc_poker(players=3)", uniform_policy, 12.611221340388003);
}

void TicTacToeTests() {
  UniformPolicy uniform_policy;
  std::shared_ptr<const Game> game = LoadGame("tic_tac_toe");
  TabularBestResponseMDP tbr1(*game, uniform_policy);
  TabularBestResponseMDPInfo br_info = tbr1.NashConv();
  SPIEL_CHECK_EQ(tbr1.TotalNumNonterminals(), 4520);

  // This will be < 2 because there are drawing lines with nonzero
  // probability. Verified with other best-response algorithm.
  SPIEL_CHECK_FLOAT_NEAR(br_info.nash_conv, 1.919659391534391, kFloatTolerance);

  // First action policy is fully exploitable (easy to check by hand).
  FirstActionPolicy first_action_policy;
  TabularBestResponseMDP tbr2(*game, first_action_policy);
  TabularBestResponseMDPInfo br_info2 = tbr2.NashConv();
  SPIEL_CHECK_FLOAT_NEAR(br_info2.nash_conv, 2.0, kFloatTolerance);
}

void RPSGameTests() {
  UniformPolicy uniform_policy;
  FirstActionPolicy first_action_policy;

  std::shared_ptr<const Game> game = LoadGame("matrix_rps");
  TabularBestResponseMDP tbr1(*game, uniform_policy);
  TabularBestResponseMDPInfo br_info = tbr1.NashConv();
  SPIEL_CHECK_FLOAT_NEAR(br_info.nash_conv, 0.0, kFloatTolerance);

  TabularBestResponseMDP tbr2(*game, first_action_policy);
  TabularBestResponseMDPInfo br_info2 = tbr2.NashConv();
  SPIEL_CHECK_FLOAT_NEAR(br_info2.nash_conv, 2.0, kFloatTolerance);
}

void GoofspielGameTests() {
  UniformPolicy uniform_policy;
  FirstActionPolicy first_action_policy;

  // There The values in Goofspiel are inconsistent across BR implementations.
  // This gets: 1.333333333333333 1.666666666666667
  {
    std::shared_ptr<const Game> game = LoadGame(
        "turn_based_simultaneous_game(game=goofspiel(num_cards=3))");
    double value = NashConv(*game, uniform_policy, true);
    printf("%0.15lf\n", value);
    double value2 = NashConv(*game, first_action_policy, true);
    printf("%0.15lf\n", value2);
  }

  {
    // This gets: 1.33333 2.0
    std::shared_ptr<const Game> game = LoadGame(
        "turn_based_simultaneous_game("
        "game=goofspiel(num_cards=3,points_order=descending))");
    double value = NashConv(*game, uniform_policy, true);
    printf("%0.15lf\n", value);
    double value2 = NashConv(*game, first_action_policy, true);
    printf("%0.15lf\n", value2);
  }

  /* These get much lower values. I suspect some issues with the observation
   * string.
  std::shared_ptr<const Game> game = LoadGame("goofspiel(num_cards=3)");
  std::shared_ptr<const Game> game = LoadGame("goofspiel(num_cards=3,points_order=descending)");
  TabularBestResponseMDP tbr1(*game, uniform_policy);
  TabularBestResponseMDPInfo br_info = tbr1.NashConv();
  SPIEL_CHECK_FLOAT_NEAR(br_info.nash_conv, 0.0, kFloatTolerance);

  TabularBestResponseMDP tbr2(*game, first_action_policy);
  TabularBestResponseMDPInfo br_info2 = tbr2.NashConv();
  SPIEL_CHECK_FLOAT_NEAR(br_info2.nash_conv, 2.0, kFloatTolerance);
  */
}

void OshiZumoGameTests() {
  UniformPolicy uniform_policy;
  FirstActionPolicy first_action_policy;

  // Numbers verified against algorithms::NashConv using
  // turn_based_simultaneous_game.

  std::shared_ptr<const Game> game = LoadGame(
      "oshi_zumo(coins=10,size=3,min_bid=1)");
  TabularBestResponseMDP tbr1(*game, uniform_policy);
  TabularBestResponseMDPInfo br_info = tbr1.NashConv();
  SPIEL_CHECK_FLOAT_NEAR(br_info.nash_conv, 1.997891313932980, kFloatTolerance);

  TabularBestResponseMDP tbr2(*game, first_action_policy);
  TabularBestResponseMDPInfo br_info2 = tbr2.NashConv();
  SPIEL_CHECK_FLOAT_NEAR(br_info2.nash_conv, 2.0, kFloatTolerance);
}

void ImperfectRecallLiarsDiceGameTests() {
  std::shared_ptr<const Game> ir_game = LoadGame("liars_dice_ir");
  std::shared_ptr<const Game> pr_game = LoadGame("liars_dice");

  std::cout << ir_game->GetType().short_name << std::endl;

  {
    UniformPolicy uniform_policy;
    TabularBestResponseMDP tbr1(*pr_game, uniform_policy);
    TabularBestResponseMDPInfo br_info1 = tbr1.NashConv();
    std::cout << "PR uniform: " <<  br_info1.nash_conv << std::endl;
    SPIEL_CHECK_FLOAT_NEAR(br_info1.nash_conv, 1.561488646384479,
                           kFloatTolerance);

    TabularBestResponseMDP tbr2(*ir_game, uniform_policy);
    TabularBestResponseMDPInfo br_info2 = tbr2.NashConv();
    std::cout << "IR uniform: " << br_info2.nash_conv << std::endl;
  }

  // For a reference, see Figure 1 from Lanctot et al. '12
  // http://mlanctot.info/files/papers/12icml-ir.pdf
  CFRSolver pr_solver(*pr_game);
  CFRSolver ir_solver(*ir_game);
  for (int i = 0; i < 11; i++) {
    pr_solver.EvaluateAndUpdatePolicy();
    ir_solver.EvaluateAndUpdatePolicy();
    if (i % 10 == 0) {
      const std::shared_ptr<Policy> pr_avg_policy = pr_solver.AveragePolicy();
      const std::shared_ptr<Policy> ir_avg_policy = ir_solver.AveragePolicy();

      TabularBestResponseMDP pr_tbr(*pr_game, *pr_avg_policy);
      TabularBestResponseMDPInfo pr_br_info = pr_tbr.NashConv();

      TabularBestResponseMDP ir_tbr(*ir_game, *ir_avg_policy);
      TabularBestResponseMDPInfo ir_br_info = ir_tbr.NashConv();

      printf("%3d %0.15lf %0.15lf\n", i, pr_br_info.nash_conv,
             ir_br_info.nash_conv);
    }
  }
}

}  // namespace
}  // namespace algorithms
}  // namespace open_spiel

int main(int argc, char **argv) {
  open_spiel::algorithms::TicTacToeTests();
  open_spiel::algorithms::KuhnNashConvTests();
  open_spiel::algorithms::LeducNashConvTests();
  open_spiel::algorithms::KuhnLeduc3pTests();
  open_spiel::algorithms::RPSGameTests();
  open_spiel::algorithms::OshiZumoGameTests();
  open_spiel::algorithms::GoofspielGameTests();
  open_spiel::algorithms::ImperfectRecallLiarsDiceGameTests();
}
