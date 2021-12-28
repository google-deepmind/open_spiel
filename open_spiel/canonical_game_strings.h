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

#ifndef OPEN_SPIEL_CANONICAL_GAME_STRINGS_H_
#define OPEN_SPIEL_CANONICAL_GAME_STRINGS_H_

#include <string>

// A place to store functions that return canonical game strings. These strings
// can sent to LoadGame to load the game.

namespace open_spiel {

// Returns the "canonical" definition of Heads-up No-limit Texas Hold'em and
// Heads-up Limit Texas Hold'em according to the ACPC:
// http://www.computerpokercompetition.org/.
// Valid values for betting_abstraction are "fc" for fold-call,"fcpa" for
// fold, call, pot, all-in, "fchpa" for fold, call, half-pot, pot, all-in, and
// "fullgame" for the unabstracted game. These indicate the actions that are
// allowed. Note that in limit poker, "fcpa" & "fcpha" are just the full game.
// The string returned can be passed directly to LoadGame.
std::string HunlGameString(const std::string &betting_abstraction);
std::string HulhGameString(const std::string &betting_abstraction);

// Turn based goofspiel w/ imperfect information and descending points order.
std::string TurnBasedGoofspielGameString(int num_cards);

}  // namespace open_spiel

#endif  // OPEN_SPIEL_SPIEL_H_
