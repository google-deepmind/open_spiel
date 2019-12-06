// Copyright 2019 DeepMind Technologies Ltd. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "open_spiel/games/universal_poker.h"

#include "open_spiel/game_parameters.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/tests/basic_tests.h"

namespace open_spiel {
namespace universal_poker {
namespace {

namespace testing = open_spiel::testing;

constexpr absl::string_view kKuhnLimit3P =
    ("GAMEDEF\n"
     "limit\n"
     "numPlayers = 3\n"
     "numRounds = 1\n"
     "blind = 1 1 1\n"
     "raiseSize = 1\n"
     "firstPlayer = 1\n"
     "maxRaises = 1\n"
     "numSuits = 1\n"
     "numRanks = 4\n"
     "numHoleCards = 1\n"
     "numBoardCards = 0\n"
     "END GAMEDEF\n");

constexpr absl::string_view kHoldemNoLimit6P =
    ("GAMEDEF\n"
     "nolimit\n"
     "numPlayers = 6\n"
     "numRounds = 4\n"
     "stack = 20000 20000 20000 20000 20000 20000\n"
     "blind = 50 100 0 0 0 0\n"
     "firstPlayer = 3 1 1 1\n"
     "numSuits = 4\n"
     "numRanks = 13\n"
     "numHoleCards = 2\n"
     "numBoardCards = 0 3 1 1\n"
     "END GAMEDEF\n");

void LoadGameFromGamdeDef() {
  UniversalPokerGame kuhn_limit_3p(
      {{"gamedef", GameParameter(std::string(kKuhnLimit3P))}});
  UniversalPokerGame holdem_no_limit_6p(
      {{"gamedef", GameParameter(std::string(kHoldemNoLimit6P))}});
}

void LoadGameFromDefaultConfig() { LoadGame("universal_poker"); }

void LoadAndRunGameFromGameDef() {
  std::shared_ptr<const Game> kuhn_limit_3p =
      LoadGame("universal_poker",
               {{"gamedef", GameParameter(std::string(kHoldemNoLimit6P))}});
  // TODO(b/145686585): An assert fails in InformationStateTensorShape
  // testing::RandomSimTestNoSerialize(*kuhn_limit_3p, 1);
  // TODO(b/145688976): The serialization is also broken
  // testing::RandomSimTest(*kuhn_limit_3p, 1);
}

void LoadAndRunGameFromDefaultConfig() {
  std::shared_ptr<const Game> game = LoadGame("universal_poker");
  testing::RandomSimTest(*game, 2);
}

void BasicUniversalPokerTests() {
  testing::LoadGameTest("universal_poker");
  testing::ChanceOutcomesTest(*LoadGame("universal_poker"));
  testing::RandomSimTest(*LoadGame("universal_poker"), 100);

  // testing::RandomSimBenchmark("leduc_poker", 10000, false);
  // testing::RandomSimBenchmark("universal_poker", 10000, false);

  std::shared_ptr<const Game> gFivePlayers =
      LoadGame({{"name", GameParameter(std::string("universal_poker"))},
                {"players", GameParameter(5)}});
  testing::RandomSimTest(*gFivePlayers, 100);

  testing::CheckChanceOutcomes(*LoadGame("universal_poker"));
}

}  // namespace
}  // namespace universal_poker
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::universal_poker::LoadGameFromGamdeDef();
  open_spiel::universal_poker::LoadGameFromDefaultConfig();
  open_spiel::universal_poker::LoadAndRunGameFromGameDef();
  open_spiel::universal_poker::LoadAndRunGameFromDefaultConfig();

  open_spiel::universal_poker::BasicUniversalPokerTests();
}
