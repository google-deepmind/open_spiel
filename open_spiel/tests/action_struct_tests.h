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

#ifndef OPEN_SPIEL_TESTS_ACTION_STRUCT_TESTS_H_
#define OPEN_SPIEL_TESTS_ACTION_STRUCT_TESTS_H_

#include <memory>

#include "open_spiel/spiel.h"

namespace open_spiel {
namespace testing {

// Simulate a game that uses ActionStructSampler. This is mostly intended for
// games that use action structs only as they can't use the core API's way of
// simulating random games.
void SimulateRandomGame(std::shared_ptr<const open_spiel::Game> game,
                        int seed, bool serialize = true);

}  // namespace testing
}  // namespace open_spiel


#endif  // OPEN_SPIEL_TESTS_ACTION_STRUCT_TESTS_H_



