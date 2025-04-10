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

#include <algorithm>
#include <cstddef>
#include <vector>

#include "open_spiel/abseil-cpp/absl/algorithm/container.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel::universal_poker::logic {

RandomizedPsuedoHarmonicActionTranslation CalculatePsuedoHarmonicMapping(
    int opponent_bet, int pot_size, std::vector<Action> action_abstraction) {
  // We assume the user may not have sorted + de-duped the action abstraction,
  // which are properties that we will want to be able to rely on below.
  std::vector<Action> sorted_action_abstraction = action_abstraction;
  std::sort(sorted_action_abstraction.begin(), sorted_action_abstraction.end());
  sorted_action_abstraction.erase(
      std::unique(sorted_action_abstraction.begin(),
                  sorted_action_abstraction.end()),
      sorted_action_abstraction.end());

  if (sorted_action_abstraction.size() < 2) {
    SpielFatalError("Action abstraction must have at least two unique values.");
  }
  // Action-s with value 0 and 1 map to fold and check/call, so are not valid
  // bet sizes to translate the opponent's bet to.
  if (sorted_action_abstraction[0] <= 1) {
    SpielFatalError(
        "Action abstraction Action-s must be bets, not folds or check/calls.");
  }

  // If their bet is one of the translated actions, or outside the bounds of our
  // translated actions, we don't need to actually do the math. Since simply
  // using said translated action at 100% frequency will always be the best we
  // can possibly do (and is what they do in the paper!)
  if (absl::c_binary_search(sorted_action_abstraction, opponent_bet)) {
    return RandomizedPsuedoHarmonicActionTranslation{
        .smaller_bet = opponent_bet,
        .probability_a = 1.0,
        // Should never be used, setting to the same value just to be safe
        // though.
        .larger_bet = opponent_bet,
        .probability_b = 0.0,
    };
  }
  Action abstraction_min = sorted_action_abstraction[0];
  if (opponent_bet < abstraction_min) {
    return RandomizedPsuedoHarmonicActionTranslation{
        .smaller_bet = abstraction_min,
        .probability_a = 1.0,
        // Should never be used, setting to the same value just to be safe
        // though.
        .larger_bet = abstraction_min,
        .probability_b = 0.0,
    };
  }
  Action abstraction_max =
      sorted_action_abstraction[sorted_action_abstraction.size() - 1];
  if (opponent_bet > abstraction_max) {
    return RandomizedPsuedoHarmonicActionTranslation{
        // Should never be used, setting to the same value just to be safe
        // though.
        .smaller_bet = abstraction_max,
        .probability_a = 0.0,
        .larger_bet = abstraction_max,
        .probability_b = 1.0,
    };
  }
  // If we reach this point, that means their bet is somewhere in between two of
  // the Action-s in the action abstraction. So we will need to 1. figure out
  // which two Action-s those are, and 2. calculate the randomized
  // pseudo-harmonic mapping to choose between them.

  // i=1 since if the first action in the 0 index was greater, then the checks
  // above would have returned early and we wouldn't have reached this point.
  Action translated_smaller_bet = abstraction_min;
  Action translated_larger_bet = abstraction_max;
  for (size_t i = 1; i < sorted_action_abstraction.size(); ++i) {
    if (sorted_action_abstraction[i] > opponent_bet) {
      translated_smaller_bet = sorted_action_abstraction[i - 1];
      translated_larger_bet = sorted_action_abstraction[i];
      break;
    }
    if (i == sorted_action_abstraction.size() - 1) {
      SpielFatalError("Could not find bounding actions for the opponent's bet "
                      "in the action abstraction.");
    }
  }

  // As per the paper, scaling everything down by the pot size to determine the
  // canonical "A" "B" and "x" values that go into the formula.
  //
  // (Oddly the paper implies that this automatically happens as a consequence
  // of their function, e.g. they explicitly stated that
  // ∀k > 0, x ∈ [A, B], f_kA,kB (kx) = f_A,B (x).
  // But this is clearly not true unless we scale everything such that the pot
  // size is '1' _before_ plugging it in. ... which they themselves also did in
  // the paper.)
  double psuedo_harmonic_a = static_cast<double>(translated_smaller_bet) /
                             static_cast<double>(pot_size);
  double psuedo_harmonic_b = static_cast<double>(translated_larger_bet) /
                             static_cast<double>(pot_size);
  double psuedo_harmonic_x =
      static_cast<double>(opponent_bet) / static_cast<double>(pot_size);

  // As specified in the paper:
  //
  // f_A,B (x) = ((B - x)(1 + A)) / ((B - A)(1 + x))
  //
  // where A is the smaller bet size, B is the larger bet size, and x is the
  // incoming bet size. (Which calculates specifically the probability that the
  // smaller bet size should be chosen).
  double probability_a =
      ((psuedo_harmonic_b - psuedo_harmonic_x) * (1 + psuedo_harmonic_a)) /
      ((psuedo_harmonic_b - psuedo_harmonic_a) * (1 + psuedo_harmonic_x));
  double probability_b = 1.0 - probability_a;

  return RandomizedPsuedoHarmonicActionTranslation{
      .smaller_bet = translated_smaller_bet,
      .probability_a = probability_a,
      .larger_bet = translated_larger_bet,
      .probability_b = probability_b,
  };
}

}  // namespace open_spiel::universal_poker::logic
