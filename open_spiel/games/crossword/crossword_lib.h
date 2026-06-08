// Copyright 2025 DeepMind Technologies Limited
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

#ifndef OPEN_SPIEL_GAMES_CROSSWORD_LIB_H_
#define OPEN_SPIEL_GAMES_CROSSWORD_LIB_H_

#include <memory>

#include "open_spiel/spiel.h"

namespace open_spiel {
namespace crossword {

// Simulate a game that uses the action sampler to choose actions randomly.
void SimulateRandomGame(std::shared_ptr<const open_spiel::Game> game, int seed);

// Simulate a game that uses the answers to the clues to choose actions, in
// random order.
void SimulateWinningGame(std::shared_ptr<const open_spiel::Game> game,
                         int seed);

}  // namespace crossword
}  // namespace open_spiel


#endif  // OPEN_SPIEL_GAMES_CROSSWORD_CROSSWORD_LIB_H_
