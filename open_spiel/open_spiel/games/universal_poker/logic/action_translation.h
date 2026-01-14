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
//
// If the opponent's bet is outside the bounds of the action abstraction, or
// exactly equal to one of the translated actions, it will be translated to the
// singular closest such value at 100% frequency. Specifically:
// - 'smaller_bet' will be arbitrarily set to 100% frequency if the opponent's
//    bet is equal to an Action in the action abstraction, or less than the
//    smallest Action in the action abstraction.
// - 'larger_bet' will be arbitrarily set to 100% frequency if the opponent's
//    bet is greater than the larget Action in the action abstraction.
RandomizedPsuedoHarmonicActionTranslation CalculatePsuedoHarmonicMapping(
    // The original bet size in chips of the opponent to be translated. If
    // outside the bounds of the action abstraction, will be translated to the
    // closest value. If equal to one of the translated actions, will be
    // translated to that action at 100% frequency.
    int opponent_bet,
    // Number of_chips currently in the pot.
    // Used to provide "scale invariance" property. Specifically, we scale down
    // everything so that it's calculated relative to a pot size of 1 when doing
    // the math.
    int pot_size,
    // A subset of the valid Action-s for the game.
    // Used to determine which Actions to translate the opponent's bet to.
    // Must contain at least two unique values, and must contain only values
    // that are >=2 (ie no fold or check/call Action).
    std::vector<Action> action_abstraction);

}  // namespace logic
}  // namespace universal_poker
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_UNIVERSAL_POKER_LOGIC_ACTION_TRANSLATION_H_
