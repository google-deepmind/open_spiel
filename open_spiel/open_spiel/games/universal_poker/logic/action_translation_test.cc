// Copyright 2019 DeepMind Technologies Limited
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

#include "open_spiel/games/universal_poker/logic/action_translation.h"

#include <cmath>
#include <cstdlib>
#include <vector>

#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace universal_poker {
namespace logic {

// As per Table 1 in the paper, "Rand-psHar" holding 'B'=1 and 'x'=0.25, but
// moving 'A' between various different values.
void TestRandPSHarPaperResults() {
  // The paper only calculated out to 3 decimal places.
  double tolerance = 0.001;

  int pot = 2000;
  int untranslated_bet = 500;  // 0.25 * pot

  // Moving A in [0.001, 0.01, 0.05, 0.1]
  RandomizedPsuedoHarmonicActionTranslation result_thousandth =
      CalculatePsuedoHarmonicMapping(untranslated_bet, pot, {2, pot});
  RandomizedPsuedoHarmonicActionTranslation result_hundredth
      = CalculatePsuedoHarmonicMapping(untranslated_bet, pot, {20, pot});
  RandomizedPsuedoHarmonicActionTranslation result_twentieth
      = CalculatePsuedoHarmonicMapping(untranslated_bet, pot, {100, pot});
  RandomizedPsuedoHarmonicActionTranslation result_tenth
      = CalculatePsuedoHarmonicMapping(untranslated_bet, pot, {200, pot});

  // Direct values from the paper.
  SPIEL_CHECK_LT(std::abs(result_thousandth.probability_a - 0.601), tolerance);
  SPIEL_CHECK_LT(std::abs(result_hundredth.probability_a - 0.612), tolerance);
  SPIEL_CHECK_LT(std::abs(result_twentieth.probability_a - 0.663), tolerance);
  SPIEL_CHECK_LT(std::abs(result_tenth.probability_a - 0.733), tolerance);

  // Corresponding percentages calculated by subtracting each above from 1.
  SPIEL_CHECK_LT(std::abs(result_thousandth.probability_b - 0.399), tolerance);
  SPIEL_CHECK_LT(std::abs(result_hundredth.probability_b - 0.388), tolerance);
  SPIEL_CHECK_LT(std::abs(result_twentieth.probability_b - 0.337), tolerance);
  SPIEL_CHECK_LT(std::abs(result_tenth.probability_b - 0.267), tolerance);
}

// Again per the Table 1, Rand-psHar holding B=1 and x=0.25, with A=0.1. But
// now testing scale invariance - ie that when multiplying all three of A, B,
// and x by any constant multiplicative factor k > 0 that it doesn't change the
// results.
void TestRandPSHarPaperResultScaleInvariance() {
  // The paper only calculated out to 3 decimal places.
  const double tolerance = 0.001;

  const Action opponent_bet = 5;
  const int pot = 20;
  const Action small_bet = 2;
  const Action large_bet = 20;  // B = pot
  for (int i = 1; i <= 8; ++i) {
    // [10^1, 10^2, ..., 10^8]
    int scale = pow(10, i);

    Action scaled_opponent_bet = opponent_bet * scale;
    Action scaled_small_bet = small_bet * scale;
    int scaled_pot = pot * scale;
    Action scaled_large_bet = large_bet * scale;

    RandomizedPsuedoHarmonicActionTranslation result =
      CalculatePsuedoHarmonicMapping(
          scaled_opponent_bet,
          scaled_pot,
          {scaled_small_bet, scaled_large_bet});

    SPIEL_CHECK_EQ(result.smaller_bet, scaled_small_bet);
    SPIEL_CHECK_LT(std::abs(result.probability_a - 0.733), tolerance);
    SPIEL_CHECK_EQ(result.larger_bet, scaled_large_bet);
    SPIEL_CHECK_LT(std::abs(result.probability_b - 0.267), tolerance);
  }
}

void TestRandPSHarMappingExactMatch() {
  int untranslated_bet = 200;  // pot sized bet => matches 1.0
  int pot = 200;
  std::vector<Action> action_abstraction = {100, 200, 400, 600, 20000};

  RandomizedPsuedoHarmonicActionTranslation result =
      CalculatePsuedoHarmonicMapping(untranslated_bet, pot, action_abstraction);

  SPIEL_CHECK_EQ(result.smaller_bet, 200);
  SPIEL_CHECK_EQ(result.probability_a, 1.0);
  // We don't care what the larger bet is, just that it's never used.
  SPIEL_CHECK_EQ(result.probability_b, 0.0);
}

void TestCalculatesMedianFiftyFifty() {
  // For f_A,B (x) = (B - x)(1 + A) / (B - A)(1 + x), the "median" of f `x*`
  // where each translated action should be 50% chance is (as per the paper):
  //
  // x* = (A + B + 2AB) / (A + B + 2)
  //
  // Using A=0.2 B=0.5, x* = .9/2.7 = 1/3.
  int pot = 300;
  int untranslated_bet = 100;
  std::vector<Action> action_abstraction = {3, 30, 60, 150, 300};

  // Only imprecision should be that inherent to using doubles. Since in reality
  // this bet size should result in _exactly_ choosing each at 50% frequency.
  double tolerance = 1E-12;

  RandomizedPsuedoHarmonicActionTranslation result =
      CalculatePsuedoHarmonicMapping(untranslated_bet, pot, action_abstraction);

  SPIEL_CHECK_EQ(result.smaller_bet, 60);  // 0.2 * 300 = 60
  SPIEL_CHECK_LT(std::abs(result.probability_a - 0.5), tolerance);
  SPIEL_CHECK_EQ(result.larger_bet, 150);  // 0.5 * 300 = 150
  SPIEL_CHECK_LT(std::abs(result.probability_b - 0.5), tolerance);
}

void TestShortCircuitsBelowMinAbstractionBet() {
  int untranslated_bet = 25;
  int pot = 300;
  std::vector<Action> action_abstraction = {150, 300};

  RandomizedPsuedoHarmonicActionTranslation result =
      CalculatePsuedoHarmonicMapping(untranslated_bet, pot, action_abstraction);

  SPIEL_CHECK_EQ(result.smaller_bet, 150);
  SPIEL_CHECK_EQ(result.probability_a, 1.0);
  // Don't care what the larger bet is, just that it's never used.
  SPIEL_CHECK_EQ(result.probability_b, 0.0);
}

void TestShortCircuitsAboveMaxAbstractionBet() {
  int untranslated_bet = 600;
  int pot = 300;
  std::vector<Action> action_abstraction = {225, 300, 375};

  RandomizedPsuedoHarmonicActionTranslation result =
    CalculatePsuedoHarmonicMapping(untranslated_bet, pot, action_abstraction);

  SPIEL_CHECK_EQ(result.probability_a, 0.0);
  // Don't care what the smaller bet is, just that it's never used.
  SPIEL_CHECK_EQ(result.larger_bet, 375);
  SPIEL_CHECK_EQ(result.probability_b, 1.0);
}

void TestUnsortedNonUniqueActionAbstraction() {
  double tolerance = 0.001;
  int untranslated_bet = 375;
  int pot = 200;
  std::vector<Action> action_abstraction = {400, 300, 150, 200, 150, 300};

  RandomizedPsuedoHarmonicActionTranslation result =
      CalculatePsuedoHarmonicMapping(untranslated_bet, pot, action_abstraction);

  SPIEL_CHECK_EQ(result.smaller_bet, 300);
  SPIEL_CHECK_LT(std::abs(result.probability_a - 0.217), tolerance);
  SPIEL_CHECK_EQ(result.larger_bet, 400);
  SPIEL_CHECK_LT(std::abs(result.probability_b - 0.783), tolerance);
}

}  // namespace logic
}  // namespace universal_poker
}  // namespace open_spiel

int main(int argc, char **argv) {
  // RandPSHar as in the "Rand-psHar" abbreviation used by the paper for the
  // relevant row in the relevant table.
  open_spiel::universal_poker::logic::TestRandPSHarPaperResults();
  open_spiel::universal_poker::logic::TestRandPSHarPaperResultScaleInvariance();
  open_spiel::universal_poker::logic::TestRandPSHarMappingExactMatch();
  open_spiel::universal_poker::logic::TestCalculatesMedianFiftyFifty();
  open_spiel::universal_poker::logic::TestShortCircuitsBelowMinAbstractionBet();
  open_spiel::universal_poker::logic::TestShortCircuitsAboveMaxAbstractionBet();
  open_spiel::universal_poker::logic::TestUnsortedNonUniqueActionAbstraction();
}
