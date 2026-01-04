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

#ifndef OPEN_SPIEL_GAMES_NFG_GAME_H_
#define OPEN_SPIEL_GAMES_NFG_GAME_H_

#include <memory>
#include <string>

#include "open_spiel/spiel.h"

namespace open_spiel {
namespace nfg_game {

// A Gambit .NFG file reader. Currently only the payoff version is supported.
// See http://www.gambit-project.org/gambit13/formats.html for details.
std::shared_ptr<const Game> LoadNFGGame(const std::string& data);

}  // namespace nfg_game
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_NFG_GAME_H_
