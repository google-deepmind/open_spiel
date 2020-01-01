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
GameParameters KuhnLimit3PParameters() {
  return {{"betting", GameParameter(std::string("limit"))},
          {"numPlayers", GameParameter(3)},
          {"numRounds", GameParameter(1)},
          {"blind", GameParameter(std::string("1 1 1"))},
          {"raiseSize", GameParameter(std::string("1"))},
          {"firstPlayer", GameParameter(std::string("1"))},
          {"maxRaises", GameParameter(std::string("1"))},
          {"numSuits", GameParameter(1)},
          {"numRanks", GameParameter(4)},
          {"numHoleCards", GameParameter(1)},
          {"numBoardCards", GameParameter(std::string("0"))}};
}

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
GameParameters HoldemNoLimit6PParameters() {
  return {{"betting", GameParameter(std::string("nolimit"))},
          {"numPlayers", GameParameter(6)},
          {"numRounds", GameParameter(4)},
          {"stack",
           GameParameter(std::string("20000 20000 20000 20000 20000 20000"))},
          {"blind", GameParameter(std::string("50 100 0 0 0 0"))},
          {"firstPlayer", GameParameter(std::string("3 1 1 1"))},
          {"numSuits", GameParameter(4)},
          {"numRanks", GameParameter(13)},
          {"numHoleCards", GameParameter(2)},
          {"numBoardCards", GameParameter(std::string("0 3 1 1"))}};
}

void LoadKuhnLimitWithAndWithoutGameDef() {
  UniversalPokerGame kuhn_limit_3p_gamedef(
      {{"gamedef", GameParameter(std::string(kKuhnLimit3P))}});
  UniversalPokerGame kuhn_limit_3p(KuhnLimit3PParameters());

  SPIEL_CHECK_EQ(kuhn_limit_3p_gamedef.GetACPCGame()->ToString(),
                 kuhn_limit_3p.GetACPCGame()->ToString());
  SPIEL_CHECK_TRUE((*(kuhn_limit_3p_gamedef.GetACPCGame())) ==
                   (*(kuhn_limit_3p.GetACPCGame())));
}

void LoadHoldemNoLimit6PWithAndWithoutGameDef() {
  UniversalPokerGame holdem_no_limit_6p_gamedef(
      {{"gamedef", GameParameter(std::string(kHoldemNoLimit6P))}});
  UniversalPokerGame holdem_no_limit_6p(HoldemNoLimit6PParameters());

  SPIEL_CHECK_EQ(holdem_no_limit_6p_gamedef.GetACPCGame()->ToString(),
                 holdem_no_limit_6p.GetACPCGame()->ToString());
  SPIEL_CHECK_TRUE((*(holdem_no_limit_6p_gamedef.GetACPCGame())) ==
                   (*(holdem_no_limit_6p.GetACPCGame())));
}
void LoadGameFromDefaultConfig() { LoadGame("universal_poker"); }

void LoadAndRunGamesFullParameters() {
  std::shared_ptr<const Game> kuhn_limit_3p =
      LoadGame("universal_poker", KuhnLimit3PParameters());
  testing::RandomSimTestNoSerialize(*kuhn_limit_3p, 1);
  // TODO(b/145688976): The serialization is also broken
  // In particular, the firstPlayer string "1" is converted back to an integer
  // when deserializing, which crashes.
  // testing::RandomSimTest(*kuhn_limit_3p, 1);

  std::shared_ptr<const Game> holdem_nolimit_6p =
      LoadGame("universal_poker", HoldemNoLimit6PParameters());
  testing::RandomSimTestNoSerialize(*holdem_nolimit_6p, 1);
  testing::RandomSimTest(*holdem_nolimit_6p, 3);
}

void LoadAndRunGameFromGameDef() {
  std::shared_ptr<const Game> holdem_nolimit_6p =
      LoadGame("universal_poker",
               {{"gamedef", GameParameter(std::string(kHoldemNoLimit6P))}});
  testing::RandomSimTestNoSerialize(*holdem_nolimit_6p, 1);
  // TODO(b/145688976): The serialization is also broken
  // testing::RandomSimTest(*holdem_nolimit_6p, 1);
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

  testing::CheckChanceOutcomes(*LoadGame("universal_poker"));
}

}  // namespace
}  // namespace universal_poker
}  // namespace open_spiel

int main(int argc, char **argv) {
  open_spiel::universal_poker::LoadKuhnLimitWithAndWithoutGameDef();
  open_spiel::universal_poker::LoadHoldemNoLimit6PWithAndWithoutGameDef();

  open_spiel::universal_poker::LoadAndRunGamesFullParameters();

  open_spiel::universal_poker::LoadGameFromDefaultConfig();
  open_spiel::universal_poker::LoadAndRunGameFromGameDef();
  open_spiel::universal_poker::LoadAndRunGameFromDefaultConfig();

  open_spiel::universal_poker::BasicUniversalPokerTests();
}
