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

#include "open_spiel/games/go/sgf_game_loader.h"

#include <memory>
#include <string>

#include "open_spiel/game_parameters.h"
#include "open_spiel/games/go/go.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/tests/basic_tests.h"
#include "open_spiel/utils/sgf_reader.h"

namespace open_spiel {
namespace go {
namespace {

void BasicSGFReaderTests() {
  // Example game strings are defined in sgf_reader.h.

  VectorOfGamesAndStates games_and_states =
      LoadGamesFromSGFString(kExampleGoSgfString);
  SPIEL_CHECK_EQ(games_and_states.size(), 1);
  // Serialization should work here since there are no setup moves.
  testing::RandomSimTestWithSpecificInitialState(
      *games_and_states[0].first, 1, games_and_states[0].second.get(),
      /*serialize=*/true);

  // Same here, but for the first game only.
  games_and_states = LoadGamesFromSGFString(kExampleGoSgfString2);
  SPIEL_CHECK_EQ(games_and_states.size(), 2);
  testing::RandomSimTestWithSpecificInitialState(
      *games_and_states[0].first, 1, games_and_states[0].second.get(),
      /*serialize=*/true);
  testing::RandomSimTestWithSpecificInitialState(
      *games_and_states[1].first, 1, games_and_states[1].second.get(),
      /*serialize=*/false);

  // This game has only setup moves, so serialization is not supported.
  games_and_states = LoadGamesFromSGFString(kExampleGoSgfString3);
  SPIEL_CHECK_EQ(games_and_states.size(), 1);
  testing::RandomSimTestWithSpecificInitialState(
      *games_and_states[0].first, 1, games_and_states[0].second.get(),
      /*serialize=*/false);
}

}  // namespace
}  // namespace go
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::go::BasicSGFReaderTests();
}
