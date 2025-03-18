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
#include <cmath>
#include <vector>

#include "open_spiel/abseil-cpp/absl/algorithm/container.h"
#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel::universal_poker::logic {

RandomizedPsuedoHarmonicActionTranslation CalculatePsuedoHarmonicMapping(
    int opponent_bet, Action untranslated_min_bet, Action untranslated_max_bet,
    int pot_size, std::vector<double> action_abstraction) {
  if (untranslated_max_bet < untranslated_min_bet) {
    SpielFatalError(absl::StrCat(
        "untranslated_max_bet must be >= untranslated_min_bet, but got ",
        untranslated_max_bet, " < ", untranslated_min_bet));
  }
  if (opponent_bet < untranslated_min_bet) {
    SpielFatalError(absl::StrCat(
        "The opponent_bet must be >= untranslated_min_bet, but got ",
        opponent_bet, " < ", untranslated_min_bet));
  }
  if (opponent_bet > untranslated_max_bet) {
    SpielFatalError(absl::StrCat(
        "The opponent_bet must be <= untranslated_min_bet, but got ",
        opponent_bet, " > ", untranslated_max_bet));
  }

  // Choose the set of translated (but unscaled) bet sizes we will consider.
  //
  // Start by narrowing down the set of allowed Actions until we find the ones
  // that are closest in size to the scaled values as specified in the action
  // abstraction.
  std::vector<Action> translated_actions;
  for (double abstraction : action_abstraction) {
    // Note: scaling the action abstraction value to be relative to the pot size
    // here rather than the other way around. This way we can work with the
    // unscaled Action-s unless we really need to do the full Pseudoharmonic
    // mapping calculation.
    int abstraction_scaled_up =
        std::round(static_cast<double>(pot_size) * abstraction);
    if (abstraction_scaled_up <= untranslated_min_bet) {
      translated_actions.push_back(untranslated_min_bet);
      continue;
    } else if (abstraction_scaled_up >= untranslated_max_bet) {
      translated_actions.push_back(untranslated_max_bet);
      continue;
    } else {
      translated_actions.push_back(abstraction_scaled_up);
    }
  }

  // Ensures that past this point we can safely assume that the translated
  // actions are unique and sorted.
  std::sort(translated_actions.begin(), translated_actions.end());
  translated_actions.erase(
      std::unique(translated_actions.begin(), translated_actions.end()),
      translated_actions.end());

  // If their bet is one of the translated actions, or outside the bounds of our
  // translated actions, we don't need to actually do the math. Since simply
  // using said translated action at 100% frequency will always be the best we
  // can possibly do (and is what they do in the paper!)
  if (absl::c_binary_search(translated_actions, opponent_bet)) {
    return RandomizedPsuedoHarmonicActionTranslation{
        .smaller_bet = opponent_bet,
        .probability_a = 1.0,
        // Should never be used, setting to the same value just to be safe
        // though.
        .larger_bet = opponent_bet,
        .probability_b = 0.0,
    };
  }
  if (opponent_bet < translated_actions[0]) {
    return RandomizedPsuedoHarmonicActionTranslation{
        .smaller_bet = translated_actions[0],
        .probability_a = 1.0,
        // Should never be used, setting to the same value just to be safe
        // though.
        .larger_bet = translated_actions[0],
        .probability_b = 0.0,
    };
  }
  if (opponent_bet > translated_actions[translated_actions.size() - 1]) {
    return RandomizedPsuedoHarmonicActionTranslation{
        // Should never be used, setting to the same value just to be safe
        // though.
        .smaller_bet = translated_actions[translated_actions.size() - 1],
        .probability_a = 0.0,
        .larger_bet = translated_actions[translated_actions.size() - 1],
        .probability_b = 1.0,
    };
  }

  // Otherwise their bet is somewhere in between two of our translated actions
  // and we need to calculate the randomized pseudo-harmonic mapping to choose
  // between the two.
  //
  // First step: determine the two such actions that bound the opponent's bet.
  int translated_smaller_bet = translated_actions[0];
  int translated_larger_bet = translated_actions[translated_actions.size() - 1];
  for (int translated_bet : translated_actions) {
    if (translated_bet < opponent_bet) {
      // This size *could* be the left-side bounding action, but we don't know
      // for certain until we find the right-side bounding action.
      translated_smaller_bet = translated_bet;
      continue;
    } else {
      // Once we hit this we know for certain that we've found the left-side
      // bound in the prior loop and the right-side bound here.
      translated_larger_bet = translated_bet;
      break;
    }
  }

  // Second step: now that we have the two bounding translated actions, we can
  // finally compute the canonical "A" "B" and "x" values - as per the paper -
  // by scaling everything down by the pot size.
  //
  // (Oddly the paper implies that this automatically happens as a consequence
  // of their function, e.g. they explicitly stated that
  // ∀k > 0, x ∈ [A, B], f_kA,kB (kx) = f_A,B (x).
  // But this is clearly not true unless we scale everything such that the pot
  // size is '1' _before_ plugging it in. ... which fwiw they themselves also
  // did in the paper.)
  double psuedo_harmonic_a = static_cast<double>(translated_smaller_bet) /
                             static_cast<double>(pot_size);
  double psuedo_harmonic_b = static_cast<double>(translated_larger_bet) /
                             static_cast<double>(pot_size);
  double psuedo_harmonic_x =
      static_cast<double>(opponent_bet) / static_cast<double>(pot_size);

  // Third step: perform the math as specified in the paper to calculate the
  // percentage to use the smaller bet size. Specifically:
  //
  // f_A,B (x) = ((B - x)(1 + A)) / ((B - A)(1 + x))
  //
  // where A is the smaller bet size, B is the larger bet size, and x is the
  // incoming bet size.
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
