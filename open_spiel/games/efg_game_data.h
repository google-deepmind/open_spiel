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

#ifndef OPEN_SPIEL_GAMES_EFG_GAME_DATA_H_
#define OPEN_SPIEL_GAMES_EFG_GAME_DATA_H_

#include <string>

#include "open_spiel/games/efg_game.h"
#include "open_spiel/spiel.h"

namespace open_spiel {
namespace efg_game {

// A few example games used in the tests. These are identical to the contents
// of the files in efg/ but do not need to be loadable from a specific path
// when running tests.
std::string GetSampleEFGData();
std::string GetKuhnPokerEFGData();
std::string GetSignalingEFGData();
std::string GetSimpleForkEFGData();

}  // namespace efg_game
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_EFG_GAME_DATA_H_
