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

#ifndef OPEN_SPIEL_GAMES_UNIVERSAL_POKER_LOGIC_ACTION_TRANSLATION_H_
#define OPEN_SPIEL_GAMES_UNIVERSAL_POKER_LOGIC_ACTION_TRANSLATION_H_

#include <vector>

#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace universal_poker {
namespace logic {

struct RandomizedPsuedoHarmonicActionTranslation {
  // Used to calculate "A" for f_A,B (x) as per the paper, ie the smaller bet
  // size (after scaling everything down such that pot=1).
  Action smaller_bet;
  // How frequently to use the above smaller bet size. E.g. 0.5 means 50% of the
  // time. This value + probability_b should sum to 1.0; if this value is set to
  // 1.0 (100%) then the upper bet should never be used.
  double probability_a;

  // Used to calculate "B" for f_A,B (x) as per the paper, ie the larger bet
  // size (after scaling everything down such that pot=1).
  Action larger_bet;
  // How frequently to use the above larger bet size. E.g. 0.5 means 50% of the
  // time. This value + probability_a should sum to 1.0; if this value is set to
  // 1.0 (100%) then the lower bet should never be used.
  double probability_b;
};

// Implementation of the randomized pseudo-harmonic action translation algorithm
// for the universal_poker game.
//
// For more details see:
// - the supplementary materials from the 2019 paper "Superhuman AI for
//   multiplayer poker" by Noam Brown and Tuomas Sandholm
// - the 2013 paper "Action Translation in Extensive-Form Games with Large
//   Action Spaces: Axioms, Paradoxes, and the Pseudo-Harmonic Mapping" by Sam
//   Ganzfried and Tuomas Sandholm.
RandomizedPsuedoHarmonicActionTranslation CalculatePsuedoHarmonicMapping(
    // The original bet size in chips of the opponent that we want to translate.
    // Must be between untranslated_min_bet and untranslated_max_bet.
    int opponent_bet,
    // The smallest bet size in chips that the opponent could have made at this
    // point. Assumed to always be >= 2 (the minimum bet size in all
    // configurations of universal poker), and must be <= untranslated_max_bet
    // and opponent_bet.
    Action untranslated_min_bet,
    // The largest bet size in chips that the opponent could have made at this
    // point. Must be >= untranslated_min_bet and >= opponent_bet.
    Action untranslated_max_bet,
    // Number of_chips currently in the pot.
    // Used to provide "scale invariance" property. Specifically, we use this to
    // scale the untranslated size of the translated actions + the opponent's
    // bet to that everything is calculated relative to a pot size of 1 when
    // doing the math.
    int pot_size,
    // The sorted list of 'buckets' relative to a pot size to choose between.
    // Used to determine which values in valid_actions we should actually
    // translate to.
    // E.g. 1.0 is a whole-pot bet, 0.5 is a half-pot bet, 1.5 is a 150%
    // over-bet, etc.
    // Assumed to be sorted in ascending order.
    std::vector<double> action_abstraction);

}  // namespace logic
}  // namespace universal_poker
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_UNIVERSAL_POKER_LOGIC_ACTION_TRANSLATION_H_
