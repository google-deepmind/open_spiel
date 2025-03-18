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

#include <cstdlib>
#include <vector>

#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace universal_poker {
namespace logic {

// As per Table 1 in the paper, "Rand-psHar" holding 'B'=1 and 'x'=0.25, but
// moving 'A' between various different values.
void TestRandPSHarPaperResults() {
  // Scale A B and x since OpenSpiel actions are always integer chips / cannot
  // be < 2.
  // (Assuming scale invariance holds this should give the same results!)
  int scale = 10000;
  int pot = 1 * scale;
  int untranslated_bet = 0.25 * scale;
  // Action 0 maps to fold, 1 maps to check/call, only 2+ are valid bets.
  Action min_bet = 2;
  // In this section the paper only goes as big as pot, so no need to consider
  // larger sizes.
  Action max_bet = pot;

  // The paper only calculated out to 3 decimal places.
  double tolerance = 0.001;

  RandomizedPsuedoHarmonicActionTranslation result =
      CalculatePsuedoHarmonicMapping(untranslated_bet, min_bet, max_bet, pot,
                                     {0.001, 1});
  SPIEL_CHECK_LT(std::abs(result.probability_a - 0.601), tolerance);
  SPIEL_CHECK_LT(std::abs(result.probability_b - 0.399), tolerance);

  result = CalculatePsuedoHarmonicMapping(untranslated_bet, min_bet, max_bet,
                                          pot, {0.01, 1});
  SPIEL_CHECK_LT(std::abs(result.probability_a - 0.612), tolerance);
  SPIEL_CHECK_LT(std::abs(result.probability_b - 0.388), tolerance);

  result = CalculatePsuedoHarmonicMapping(untranslated_bet, min_bet, max_bet,
                                          pot, {0.05, 1});
  SPIEL_CHECK_LT(std::abs(result.probability_a - 0.663), tolerance);
  SPIEL_CHECK_LT(std::abs(result.probability_b - 0.337), tolerance);

  result = CalculatePsuedoHarmonicMapping(untranslated_bet, min_bet, max_bet,
                                          pot, {0.1, 1});
  SPIEL_CHECK_LT(std::abs(result.probability_a - 0.733), tolerance);
  SPIEL_CHECK_LT(std::abs(result.probability_b - 0.267), tolerance);
}

// Again per the Table 1, Rand-psHar holding B=1 and x=0.25, with A=0.1. But
// now testing scale invariance - ie that when multiplying all three of A, B,
// and x by any constant multiplicative factor k > 0 that it doesn't change the
// results.
void TestRandPSHarPaperResultScaleInvariance() {
  for (int multiplicative_factor : {1, 1000, 1000000}) {
    int pot = 20 * multiplicative_factor;
    // 20 chips => 10 BB
    // 200 chips => 100 BB
    // ...
    int untranslated_bet = pot / 4;
    std::vector<double> action_abstraction = {
        0.1,  // 10% of the pot
        1     // 100% of the pot
    };
    // The paper only calculated out to 3 decimal places.
    double tolerance = 0.001;
    // Action 0 maps to fold, 1 maps to check/call, only 2+ are valid bets.
    Action min_bet = 2;
    // In this section the paper only goes as big as pot, so no need to consider
    // larger sizes.
    Action max_bet = pot;

    RandomizedPsuedoHarmonicActionTranslation result =
        CalculatePsuedoHarmonicMapping(untranslated_bet, min_bet, max_bet, pot,
                                       action_abstraction);
    SPIEL_CHECK_EQ(result.smaller_bet, pot * 0.1);
    SPIEL_CHECK_LT(std::abs(result.probability_a - 0.733), tolerance);
    SPIEL_CHECK_EQ(result.larger_bet, pot);
    SPIEL_CHECK_LT(std::abs(result.probability_b - 0.267), tolerance);
  }
}

void TestRandPSHarMappingExactMatch() {
  Action min_bet = 2;
  Action max_bet = 99999;

  std::vector<double> action_abstraction = {0.5, 1.0, 2.0, 3.0, 100.0};
  int pot = 200;
  int untranslated_bet = 200;  // pot sized bet => matches 1.0

  RandomizedPsuedoHarmonicActionTranslation result =
      CalculatePsuedoHarmonicMapping(untranslated_bet, min_bet, max_bet, pot,
                                     action_abstraction);

  SPIEL_CHECK_EQ(result.smaller_bet, 200);
  SPIEL_CHECK_EQ(result.probability_a, 1.0);
  // We don't care what the larger bet is, just that it's never used.
  SPIEL_CHECK_EQ(result.probability_b, 0.0);
}

void TestCalculatesMedianFiftyFifty() {
  // For f_A,B (x) = (B - x)(1 + A) / (B - A)(1 + x), the "median" of f where
  // each translated action should be 50% chance is (as per the paper):
  // x* = (A + B + 2AB) / (A + B + 2)
  //
  // Using A=0.2, B=0.5, x* = .9/2.7 = 1/3.
  std::vector<double> action_abstraction = {0.01, 0.1, 0.2, 0.5, 1.0};
  int untranslated_bet = 100;
  int pot = 300;

  Action min_bet = 2;
  Action max_bet = 9999999;

  // Only imprecision should be that inherent to using doubles. Since in reality
  // this bet size should result in _exactly_ choosing each at 50% frequency.
  double tolerance = 1E-12;

  RandomizedPsuedoHarmonicActionTranslation result =
      CalculatePsuedoHarmonicMapping(untranslated_bet, min_bet, max_bet, pot,
                                     action_abstraction);
  SPIEL_CHECK_EQ(result.smaller_bet, 60);  // 0.2 * 300 = 60
  SPIEL_CHECK_LT(std::abs(result.probability_a - 0.5), tolerance);
  SPIEL_CHECK_EQ(result.larger_bet, 150);  // 0.5 * 300 = 150
  SPIEL_CHECK_LT(std::abs(result.probability_b - 0.5), tolerance);
}

void TestShortCircuitsBelowMinAbstractionBet() {
  int untranslated_bet = 2;
  int pot = 300;
  // Deliberately set so smallest translated bet is 0.5 * pot = 150
  std::vector<double> action_abstraction = {0.5, 1.0};

  Action min_bet = 2;
  Action max_bet = 99999;

  RandomizedPsuedoHarmonicActionTranslation result =
      CalculatePsuedoHarmonicMapping(untranslated_bet, min_bet, max_bet, pot,
                                     action_abstraction);
  SPIEL_CHECK_EQ(result.smaller_bet, 150);
  SPIEL_CHECK_EQ(result.probability_a, 1.0);
  // Don't care what the larger bet is, just that it's never used.
  SPIEL_CHECK_EQ(result.probability_b, 0.0);
}

void TestShortCircuitsAboveMaxAbstractionBet() {
  int untranslated_bet = 600;
  int pot = 300;
  // Deliberately set so largest translated bet is 1.25 * pot = 375
  std::vector<double> action_abstraction = {0.75, 1.0, 1.25};

  Action min_bet = 2;
  Action max_bet = 99999;

  RandomizedPsuedoHarmonicActionTranslation result =
      CalculatePsuedoHarmonicMapping(untranslated_bet, min_bet, max_bet, pot,
                                     action_abstraction);
  // Don't care what the smaller bet is, just that it's never used.
  SPIEL_CHECK_EQ(result.probability_a, 0.0);
  SPIEL_CHECK_EQ(result.larger_bet, 375);
  SPIEL_CHECK_EQ(result.probability_b, 1.0);
}


void TestUnsortedNonUniqueActionAbstraction() {
  Action min_bet = 2;
  Action max_bet = 999;

  std::vector<double> action_abstraction = {2.0, 1.5, 0.5, 1.0, 0.5, 1.5};
  int pot = 200;
  int untranslated_bet = 375;
  double tolerance = 0.001;

  RandomizedPsuedoHarmonicActionTranslation result =
      CalculatePsuedoHarmonicMapping(untranslated_bet, min_bet, max_bet, pot,
                                     action_abstraction);

  SPIEL_CHECK_EQ(result.smaller_bet, 300);  // 1.5 times pot
  SPIEL_CHECK_LT(std::abs(result.probability_a - 0.217), tolerance);
  SPIEL_CHECK_EQ(result.larger_bet, 400);  // 2.0 times pot
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
