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

#ifndef OPEN_SPIEL_GAMES_UNIVERSAL_POKER_LOGIC_GAMEDEF_H_
#define OPEN_SPIEL_GAMES_UNIVERSAL_POKER_LOGIC_GAMEDEF_H_

#include <string>

namespace open_spiel {
namespace universal_poker {
namespace logic {

// Converts an ACPC gamedef into the corresponding string that's compatible with
// OpenSpiel.
std::string GamedefToOpenSpielParameters(const std::string& acpc_gamedef);

}  // namespace logic
}  // namespace universal_poker
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_UNIVERSAL_POKER_LOGIC_GAMEDEF_H_
