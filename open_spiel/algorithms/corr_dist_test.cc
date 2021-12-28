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

#include "open_spiel/algorithms/corr_dist.h"

#include <numeric>
#include <unordered_map>

#include "open_spiel/algorithms/cfr.h"
#include "open_spiel/algorithms/corr_dev_builder.h"
#include "open_spiel/game_transforms/turn_based_simultaneous_game.h"
#include "open_spiel/games/efg_game.h"
#include "open_spiel/games/efg_game_data.h"
#include "open_spiel/games/goofspiel.h"
#include "open_spiel/matrix_game.h"
#include "open_spiel/policy.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/utils/init.h"

namespace open_spiel {
namespace algorithms {
namespace {

inline constexpr double kFloatTolerance = 1e-12;

inline constexpr const char* kGreenwaldSarfatiEg1File =
    "open_spiel/games/efg/greenwald_sarfati_example1.efg";
inline constexpr const char* kGreenwaldSarfatiEg2File =
    "open_spiel/games/efg/greenwald_sarfati_example2.efg";

void TestGibson13MatrixGameExample() {
  // Tests that the example from Sec 2.2 of Gibson 2013, Regret Minimization in
  // Non-Zero-Sum Games with Applications to Building Champion Multiplayer
  // Computer Poker Agents (https://arxiv.org/abs/1305.0034).
  //
  //           a    b
  //       A  1,0  0,0
  //       B  0,0  2,0
  //       C -1,0  1,0
  //
  std::shared_ptr<const matrix_game::MatrixGame> gibson_game =
      matrix_game::CreateMatrixGame({{1, 0}, {0, 2}, {-1, 1}},
                                    {{0, 0}, {0, 0}, {0, 0}});

  NormalFormCorrelationDevice mu = {
      {0.5, {0, 0}},   // (A, a) = 0.5
      {0.25, {1, 1}},  // (B, b) = 0.25
      {0.25, {2, 1}}   // (C, b) = 0.25
  };

  // mu is a CCE.
  SPIEL_CHECK_TRUE(Near(CCEDist(*gibson_game, mu), 0.0));

  // mu is not a CE, because first player gains 1 by deviating to B after
  // receiving the third recommendation, which happens with prob 0.25
  SPIEL_CHECK_TRUE(Near(CEDist(*gibson_game, mu), 0.25));

  // Repeat these tests with a turn-based simultaneous game.
  SPIEL_CHECK_TRUE(Near(CCEDist(*ConvertToTurnBased(*gibson_game), mu), 0.0));
  SPIEL_CHECK_TRUE(Near(CEDist(*ConvertToTurnBased(*gibson_game), mu), 0.25));
}

void TestShapleysGame() {
  // Shapley's game is a general-sum version of Rock, Paper, Scissors.
  // See Fig 7.6 of rown '09, http://www.masfoundations.org/mas.pdf:
  //
  //       R    P    S
  //   R  0,0  0,1  1,0
  //   P  1,0  0,0  0,1
  //   S  0,1  1,0  0,0
  std::shared_ptr<const Game> shapleys_game = LoadGame("matrix_shapleys_game");

  // There is a unique Nash eq at (1/3, 1/3, 1/3). So by Sec 3.4.5 of Shoham
  // and Leyton-Brown there is a CE with 1/9 on all the entries.
  NormalFormCorrelationDevice mu = {
      {1.0 / 9.0, {0, 0}}, {1.0 / 9.0, {0, 1}}, {1.0 / 9.0, {0, 2}},
      {1.0 / 9.0, {1, 0}}, {1.0 / 9.0, {1, 1}}, {1.0 / 9.0, {1, 2}},
      {1.0 / 9.0, {2, 0}}, {1.0 / 9.0, {2, 1}}, {1.0 / 9.0, {2, 2}}};

  SPIEL_CHECK_TRUE(Near(CEDist(*shapleys_game, mu), 0.0));
  std::vector<double> expected_values_full_support =
      ExpectedValues(*shapleys_game, mu);
  SPIEL_CHECK_TRUE(Near(expected_values_full_support[0], 1.0 / 3.0));
  SPIEL_CHECK_TRUE(Near(expected_values_full_support[1], 1.0 / 3.0));

  // There is another CE with 1/6 on the off-diagonals.
  mu = {{1.0 / 6.0, {0, 1}}, {1.0 / 6.0, {0, 2}}, {1.0 / 6.0, {1, 0}},
        {1.0 / 6.0, {1, 2}}, {1.0 / 6.0, {2, 0}}, {1.0 / 6.0, {2, 1}}};

  SPIEL_CHECK_TRUE(Near(CEDist(*shapleys_game, mu), 0.0));
  std::vector<double> expected_values_off_diagonals =
      ExpectedValues(*shapleys_game, mu);
  SPIEL_CHECK_TRUE(Near(expected_values_off_diagonals[0], 0.5));
  SPIEL_CHECK_TRUE(Near(expected_values_off_diagonals[1], 0.5));
}

void TestBoS() {
  // Correlated equilibrium example from Sec 3.4.5 of Shoham & Leyton-Brown '09
  // https://masfoundations.org/mas.pdf
  //
  //       LW    WL
  //   LW  2,1  0,0
  //   WL  0,0  1,2
  std::shared_ptr<const matrix_game::MatrixGame> bos_game =
      matrix_game::CreateMatrixGame({{2, 0}, {0, 1}}, {{1, 0}, {0, 2}});

  NormalFormCorrelationDevice mu = {{0.5, {0, 0}}, {0.5, {1, 1}}};
  SPIEL_CHECK_TRUE(Near(CEDist(*bos_game, mu), 0.0));
}

void TestChicken() {
  // Example from: https://en.wikipedia.org/wiki/Correlated_equilibrium
  std::shared_ptr<const matrix_game::MatrixGame> chicken_game =
      matrix_game::CreateMatrixGame({{0, 7}, {2, 6}}, {{0, 2}, {7, 6}});

  NormalFormCorrelationDevice mu = {
      {0.5, {1, 1}}, {0.25, {1, 0}}, {0.25, {0, 1}}};
  SPIEL_CHECK_TRUE(Near(CEDist(*chicken_game, mu), 0.0));
}

void TestSignalingExampleVonStengelForges2008() {
  // Runs a test based on the signaling game example in Section 2.3 of von
  // Stengel & Forges 2008, Extensive-Form Correlated Equilibrium:
  // Definition and Computational Complexity.

  // First, check the CE of the normal-form version in Figure 2.
  std::shared_ptr<const matrix_game::MatrixGame> signaling_game_nfg =
      matrix_game::CreateMatrixGame(
          {{5, 5, 0, 0}, {5, 2, 3, 0}, {5, 3, 2, 0}, {5, 0, 5, 0}},
          {{5, 5, 6, 6}, {5, 8, 3, 6}, {5, 3, 8, 6}, {5, 6, 5, 6}});

  // Mix equally a'' = b'' = c'' == d'' = 1/4.
  NormalFormCorrelationDevice mu_nfg = {
      {0.25, {0, 3}}, {0.25, {1, 3}}, {0.25, {2, 3}}, {0.25, {3, 3}}};
  SPIEL_CHECK_TRUE(Near(CEDist(*signaling_game_nfg, mu_nfg), 0.0));
  std::vector<double> expected_values =
      ExpectedValues(*signaling_game_nfg, mu_nfg);
  SPIEL_CHECK_TRUE(Near(expected_values[0], 0.0));
  SPIEL_CHECK_TRUE(Near(expected_values[1], 6.0));

  // Now do the extensive-form version. From von Stengel & Forges '08:
  std::shared_ptr<const Game> efg_game =
      efg_game::LoadEFGGame(efg_game::GetSignalingEFGData());
  const efg_game::EFGGame* signaling_game =
      dynamic_cast<const efg_game::EFGGame*>(efg_game.get());
  SPIEL_CHECK_TRUE(signaling_game != nullptr);

  // "However, there is an EFCE with better payoff to both players compared to
  // the outcome with payoff pair (0, 6): A signal X_G or Y_G is chosen with
  // equal probability for type G, and player 2 is told to accept when receiving
  // the chosen signal and to refuse when receiving the other signal (so X_G and
  // lX rY are perfectly correlated, as well as Y_G and r_X l_Y ).
  TabularPolicy XG_XB_policy = efg_game::EFGGameTabularPolicy(efg_game,
      {{{0, "G"}, {{"X_G", 1.0}, {"Y_G", 0.0}}},
       {{0, "B"}, {{"X_B", 1.0}, {"Y_B", 0.0}}},
       {{1, "X"}, {{"l_X", 1.0}, {"r_X", 0.0}}},
       {{1, "Y"}, {{"l_Y", 0.0}, {"r_Y", 1.0}}}});

  TabularPolicy YG_XB_policy = efg_game::EFGGameTabularPolicy(efg_game,
      {{{0, "G"}, {{"X_G", 0.0}, {"Y_G", 1.0}}},
       {{0, "B"}, {{"X_B", 1.0}, {"Y_B", 0.0}}},
       {{1, "X"}, {{"l_X", 0.0}, {"r_X", 1.0}}},
       {{1, "Y"}, {{"l_Y", 1.0}, {"r_Y", 0.0}}}});

  TabularPolicy XG_YB_policy = efg_game::EFGGameTabularPolicy(efg_game,
      {{{0, "G"}, {{"X_G", 1.0}, {"Y_G", 0.0}}},
       {{0, "B"}, {{"X_B", 0.0}, {"Y_B", 1.0}}},
       {{1, "X"}, {{"l_X", 1.0}, {"r_X", 0.0}}},
       {{1, "Y"}, {{"l_Y", 0.0}, {"r_Y", 1.0}}}});

  TabularPolicy YG_YB_policy = efg_game::EFGGameTabularPolicy(efg_game,
      {{{0, "G"}, {{"X_G", 0.0}, {"Y_G", 1.0}}},
       {{0, "B"}, {{"X_B", 0.0}, {"Y_B", 1.0}}},
       {{1, "X"}, {{"l_X", 0.0}, {"r_X", 1.0}}},
       {{1, "Y"}, {{"l_Y", 1.0}, {"r_Y", 0.0}}}});

  // Finally test to see if it's an EFCE.
  CorrelationDevice mu = {
      {0.25, XG_XB_policy},
      {0.25, YG_XB_policy},
      {0.25, XG_YB_policy},
      {0.25, YG_YB_policy},
  };
  expected_values = ExpectedValues(*efg_game, mu);
  SPIEL_CHECK_TRUE(Near(expected_values[0], 3.5));
  SPIEL_CHECK_TRUE(Near(expected_values[1], 6.5));

  CorrDistConfig config;
  SPIEL_CHECK_TRUE(Near(EFCEDist(*efg_game, config, mu), 0.0));

  // EFCEs are contained withing EFCCE (see Section 5 of
  // https://arxiv.org/abs/1908.09893), so mu is also an EFCCE in this game.
  SPIEL_CHECK_TRUE(Near(EFCCEDist(*efg_game, config, mu), 0.0));
}

void Test1PInOutGame() {
  // Example game described in Section 2.4 of von Stengel & Forges,
  // Extensive Form Correlated Equilibrium: Definition and Computational
  // Complexity. CDAM Research Report LSE-CDAM-2006-04.
  // http://www.cdam.lse.ac.uk/Reports/Files/cdam-2006-04.pdf
  //
  // This is a simple example that illustrates the difference between AFCE and
  // EFCE.
  const char* kInOutGameData = R"###(
    EFG 2 R "InOutGame" { "P1" } ""

    p "ROOT" 1 1 "Root Infoset" { "In" "Out" } 0
      p "In" 1 2 "In Infoset" { "In" "Out" } 0
        t "In In" 1 "Outcome In In" { 1.0 }
        t "In Out" 2 "Outcome In Out" { 0.0 }
      p "Out" 1 3 "Out Infoset" { "In" "Out" } 0
        t "Out In" 3 "Outcome Out In" { 0.0 }
        t "Out Out" 4 "Outcome Out Out" { 0.0 }
  )###";
  std::shared_ptr<const Game> efg_game = efg_game::LoadEFGGame(kInOutGameData);

  TabularPolicy single_policy = efg_game::EFGGameTabularPolicy(
      efg_game, {{{0, "Root Infoset"}, {{"In", 0.0}, {"Out", 1.0}}},
                 {{0, "In Infoset"}, {{"In", 0.0}, {"Out", 1.0}}},
                 {{0, "Out Infoset"}, {{"In", 0.0}, {"Out", 1.0}}}});

  CorrelationDevice mu = {{1.0, single_policy}};

  std::vector<double> expected_values = ExpectedValues(*efg_game, mu);
  SPIEL_CHECK_TRUE(Near(expected_values[0], 0.0));

  CorrDistConfig config;
  SPIEL_CHECK_TRUE(Near(AFCEDist(*efg_game, config, mu), 0.0));

  // Player has incentive to switch to In at the first decision and, once having
  // deviated switch to In again, achieving a value of 1. This is 1 more than
  // the correlation device's expected value of 0.
  SPIEL_CHECK_FLOAT_NEAR(EFCEDist(*efg_game, config, mu), 1.0, kFloatTolerance);
}

void TestGreenwaldSarfatiExample1() {
  absl::optional<std::string> file = FindFile(kGreenwaldSarfatiEg1File, 2);
  if (file.has_value()) {
    std::shared_ptr<const Game> efg_game =
        LoadGame(absl::StrCat("efg_game(filename=", file.value(), ")"));
    const efg_game::EFGGame* example_game =
        dynamic_cast<const efg_game::EFGGame*>(efg_game.get());
    SPIEL_CHECK_TRUE(example_game != nullptr);

    TabularPolicy LAl1_policy = efg_game::EFGGameTabularPolicy(efg_game,
        {{{0, "Root infoset"}, {{"L", 1.0}, {"R", 0.0}}},
         {{1, "P2 infoset"}, {{"A", 1.0}, {"B", 0.0}}},
         {{0, "Left P1 infoset"}, {{"l1", 1.0}, {"r1", 0.0}}},
         {{0, "Right P1 infoset"}, {{"l2", 1.0}, {"r2", 0.0}}}});

    TabularPolicy LBl1_policy = efg_game::EFGGameTabularPolicy(efg_game,
        {{{0, "Root infoset"}, {{"L", 1.0}, {"R", 0.0}}},
         {{1, "P2 infoset"}, {{"A", 0.0}, {"B", 1.0}}},
         {{0, "Left P1 infoset"}, {{"l1", 1.0}, {"r1", 0.0}}},
         {{0, "Right P1 infoset"}, {{"l2", 0.0}, {"r2", 1.0}}}});

    CorrelationDevice mu = {{0.5, LAl1_policy}, {0.5, LBl1_policy}};
    CorrDistConfig config;

    // This *is* an AFCE and AFCCE.
    SPIEL_CHECK_FLOAT_NEAR(AFCEDist(*efg_game, config, mu), 0.0,
                           kFloatTolerance);
    SPIEL_CHECK_FLOAT_NEAR(AFCCEDist(*efg_game, config, mu), 0.0,
                           kFloatTolerance);

    // However, *not* an EFCE nor EFCCE.
    SPIEL_CHECK_GT(EFCEDist(*efg_game, config, mu), 0.0);
    SPIEL_CHECK_GT(EFCCEDist(*efg_game, config, mu), 0.0);
  }
}

void TestGreenwaldSarfatiExample2() {
  absl::optional<std::string> file = FindFile(kGreenwaldSarfatiEg2File, 2);
  if (file.has_value()) {
    std::shared_ptr<const Game> efg_game =
        LoadGame(absl::StrCat("efg_game(filename=", file.value(), ")"));
    const efg_game::EFGGame* example_game =
        dynamic_cast<const efg_game::EFGGame*>(efg_game.get());
    SPIEL_CHECK_TRUE(example_game != nullptr);

    TabularPolicy LAl1_policy = efg_game::EFGGameTabularPolicy(efg_game,
        {{{0, "Root infoset"}, {{"L", 1.0}, {"R", 0.0}}},
         {{1, "P2 infoset"}, {{"A", 1.0}, {"B", 0.0}}},
         {{0, "Left P1 infoset"}, {{"l1", 1.0}, {"r1", 0.0}}},
         {{0, "Right P1 infoset"}, {{"l2", 1.0}, {"r2", 0.0}}}});

    TabularPolicy LBl1_policy = efg_game::EFGGameTabularPolicy(efg_game,
        {{{0, "Root infoset"}, {{"L", 1.0}, {"R", 0.0}}},
         {{1, "P2 infoset"}, {{"A", 0.0}, {"B", 1.0}}},
         {{0, "Left P1 infoset"}, {{"l1", 1.0}, {"r1", 0.0}}},
         {{0, "Right P1 infoset"}, {{"l2", 0.0}, {"r2", 1.0}}}});

    TabularPolicy LBr1_policy = efg_game::EFGGameTabularPolicy(efg_game,
        {{{0, "Root infoset"}, {{"L", 1.0}, {"R", 0.0}}},
         {{1, "P2 infoset"}, {{"A", 0.0}, {"B", 1.0}}},
         {{0, "Left P1 infoset"}, {{"l1", 0.0}, {"r1", 1.0}}},
         {{0, "Right P1 infoset"}, {{"l2", 0.0}, {"r2", 1.0}}}});

    CorrelationDevice mu = {{0.5, LAl1_policy},
                            {0.25, LBl1_policy},
                            {0.25, LBr1_policy}};

    CorrDistConfig config;
    SPIEL_CHECK_FLOAT_EQ(EFCEDist(*efg_game, config, mu), 0.0);
  }

  // Matrix game version:
  //
  //              A     B
  //   L,l1,l2   2,2   2,2
  //   L,l1,r2   2,2   2,2
  //   L,r1,l2   0,2   2,2
  //   L,r1,r2   0,2   2,2
  //   R,l1,l2   0,0   0,0
  //   R,l1,r2   0,0   3,0
  //   R,r1,l2   0,0   0,0
  //   R,r1,r2   0,0   3,0
  std::shared_ptr<const matrix_game::MatrixGame> eg2_matrix_game =
      matrix_game::CreateMatrixGame(
          {{2, 2}, {2, 2}, {0, 2}, {0, 2}, {0, 0}, {0, 3}, {0, 0}, {0, 3}},
          {{2, 2}, {2, 2}, {2, 2}, {2, 2}, {0, 0}, {0, 0}, {0, 0}, {0, 0}});

  // To show it's not a CE: match the mu in the EFCE test above
  NormalFormCorrelationDevice mu_nfg = {
      {0.5, {0, 0}},   // L,l1,l2 + A
      {0.25, {1, 1}},  // L,l1,r2 + B
      {0.25, {3, 1}}   // L,r1,r2 + B
  };

  SPIEL_CHECK_GT(CEDist(*eg2_matrix_game, mu_nfg), 0.0);
}

void TestCCECEDistCFRGoofSpiel() {
  std::shared_ptr<const Game> game = LoadGame(
      "turn_based_simultaneous_game(game=goofspiel(num_cards=3,points_order="
      "descending,returns_type=total_points))");
  for (int num_iterations : {1, 10, 100}) {
    std::vector<TabularPolicy> policies;
    policies.reserve(num_iterations);
    CFRSolverBase solver(*game,
                         /*alternating_updates=*/true,
                         /*linear_averaging=*/false,
                         /*regret_matching_plus=*/false,
                         /*random_initial_regrets*/ false);
    for (int i = 0; i < num_iterations; i++) {
      solver.EvaluateAndUpdatePolicy();
      TabularPolicy current_policy =
          static_cast<CFRCurrentPolicy*>(solver.CurrentPolicy().get())
              ->AsTabular();
      policies.push_back(current_policy);
    }

    CorrelationDevice mu = UniformCorrelationDevice(policies);
    CorrDistInfo cce_dist_info = CCEDist(*game, mu);
    std::cout << "num_iterations: " << num_iterations
              << ", cce_dist: " << cce_dist_info.dist_value << std::endl;

    // Disabled in test because it's really slow.
    // double ce_dist = CEDist(*game, DeterminizeCorrDev(mu));
    // std::cout << "num_iterations: " << num_iterations
    //          << ", approximate ce_dist: " << ce_dist << std::endl;
    CorrDistInfo ce_dist_info =
        CEDist(*game, SampledDeterminizeCorrDev(mu, 100));
    std::cout << "num_iterations: " << num_iterations
              << ", approximate ce_dist: " << ce_dist_info.dist_value
              << std::endl;
  }
}
}  // namespace
}  // namespace algorithms
}  // namespace open_spiel

namespace algorithms = open_spiel::algorithms;

int main(int argc, char** argv) {
  open_spiel::Init("", &argc, &argv, true);
  algorithms::TestGibson13MatrixGameExample();
  algorithms::TestShapleysGame();
  algorithms::TestBoS();
  algorithms::TestChicken();
  algorithms::TestSignalingExampleVonStengelForges2008();
  algorithms::Test1PInOutGame();
  algorithms::TestGreenwaldSarfatiExample1();
  algorithms::TestGreenwaldSarfatiExample2();
  algorithms::TestCCECEDistCFRGoofSpiel();
}
