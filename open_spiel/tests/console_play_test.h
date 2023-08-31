// Copyright 2023 DeepMind Technologies Limited
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

#ifndef OPEN_SPIEL_TESTS_CONSOLE_PLAY_TEST_H_
#define OPEN_SPIEL_TESTS_CONSOLE_PLAY_TEST_H_

#include <memory>
#include <unordered_map>

#include "open_spiel/spiel.h"
#include "open_spiel/spiel_bots.h"

namespace open_spiel {
namespace testing {

// Play the game via the console to test its functionality.
//
// If a start_state or start_history is passed, the game starts from the
// specified state or history. If both remain null, the game starts from the
// initial state.
//
// Bots can be specified by passing in a map to a bot per with the player id
// as the key. If the bots map remains null, then there are no bots and play
// is entirely guided by the console.
void ConsolePlayTest(
    const Game& game, const State* start_state = nullptr,
    const std::vector<Action>* start_history = nullptr,
    const std::unordered_map<Player, std::unique_ptr<Bot>>* bots = nullptr);

}  // namespace testing
}  // namespace open_spiel

#endif   // THIRD_PARTY_OPEN_SPIEL_TESTS_CONSOLE_PLAY_TEST_H_

