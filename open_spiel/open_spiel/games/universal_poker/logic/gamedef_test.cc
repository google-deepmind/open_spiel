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

#include "open_spiel/games/universal_poker/logic/gamedef.h"

#include <iostream>
#include <string>

#include "open_spiel/abseil-cpp/absl/strings/match.h"
#include "open_spiel/abseil-cpp/absl/strings/string_view.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace universal_poker {
namespace logic {

const char kSimpleHeadsupLimitPokerACPCGamedef[] =
    R""""(
GAMEDEF
limit
numPlayers = 2
numRounds = 1
blind = 5 10
raiseSize = 10 10 20
firstPlayer = 1
maxRaises = 2 2 3
numSuits = 4
numRanks = 5
numHoleCards = 1
numBoardCards = 0 2 1
END GAMEDEF)"""";

// Designed to mimic pre-existing code in card_set_test.cc
void TestGamedefToOpenSpielParametersEasyCase() {
  std::cout << "acpc gamedef:\n"
            << kSimpleHeadsupLimitPokerACPCGamedef << "\n"
            << std::endl;
  std::cout << "OpenSpiel gamestate:\n"
            << GamedefToOpenSpielParameters(kSimpleHeadsupLimitPokerACPCGamedef)
            << "\n"
            << std::endl;
}

// By "KeyOnly" we mean 'GAMEDEF', 'limit', 'nolimit', and 'END GAMEDEF' lines
void TestGamedefToOpenSpielParametersNormalizesKeyOnlyLines() {
  std::string open_spiel_game_state =
      GamedefToOpenSpielParameters(kSimpleHeadsupLimitPokerACPCGamedef);

  SPIEL_CHECK_TRUE(absl::StrContains(open_spiel_game_state, "betting=limit,"));
  SPIEL_CHECK_FALSE(
      StrContainsIgnoreCase(open_spiel_game_state, "end gamedef"));
  SPIEL_CHECK_FALSE(
      StrContainsIgnoreCase(open_spiel_game_state, "gamedef"));
  SPIEL_CHECK_FALSE(
      StrContainsIgnoreCase(open_spiel_game_state, "nolimit"));
}

// There's a bug downstream causing a runtime error if we provide it with a
// single value for keys that can have different values on each betting round.
// This function tests our (hacky) fix; whenever a value for these keys has
// only one value in it, we convert it into an equivalent one that will not
// trigger the error.
void TestGamedefToOpenSpielParametersMultiRoundValueEdgeCase() {
  std::string acpc_gamedef = R""""(
GAMEDEF
limit
numPlayers = 1
numRounds = 1
blind = 5
raiseSize = 10
firstPlayer = 1
maxRaises = 2
numSuits = 4
numRanks = 5
numHoleCards = 1
numBoardCards = 2
stack = 100
END GAMEDEF)"""";

  std::string open_spiel_game_state =
      GamedefToOpenSpielParameters(acpc_gamedef);
  SPIEL_CHECK_TRUE(
      absl::StrContains(open_spiel_game_state, ",firstPlayer=1 1,"));
  SPIEL_CHECK_TRUE(
      absl::StrContains(open_spiel_game_state, ",raiseSize=10 10,"));
  SPIEL_CHECK_TRUE(absl::StrContains(open_spiel_game_state, ",maxRaises=2 2,"));
  SPIEL_CHECK_TRUE(absl::StrContains(open_spiel_game_state, ",stack=100 100)"));
}

void TestGamedefToOpenSpielParametersRemovesUnneededLines() {
  std::string acpc_gamedef = R""""(
# COMMENT THAT SHOULD BE IGNORED
gameDEF
limit
numplayers = 2
numrounds = 1
# ANOTHER COMMENT
blind = 5 10
raisesize = 10 10 20

# Empty lines are also ignored!

MAXRAISES = 2 2 3
NUMSUITS = 4
NUMRANKS = 5
nUmHoLeCARds = 1
numBoardCARDS = 0 2 1
end GameDef

# hasta la vista
)"""";

  std::string open_spiel_game_state =
      GamedefToOpenSpielParameters(acpc_gamedef);

  SPIEL_CHECK_FALSE(absl::StrContains(open_spiel_game_state, "COMMENT"));
  SPIEL_CHECK_FALSE(absl::StrContains(open_spiel_game_state, "EMPTY"));
  SPIEL_CHECK_FALSE(absl::StrContains(open_spiel_game_state, "#"));
  SPIEL_CHECK_FALSE(absl::StrContains(open_spiel_game_state, "\n"));
  SPIEL_CHECK_FALSE(
      StrContainsIgnoreCase(open_spiel_game_state, "end gamedef"));
  SPIEL_CHECK_FALSE(
      StrContainsIgnoreCase(open_spiel_game_state, "gamedef"));
}

void TestGamedefToOpenSpielParametersNormalizesCapitalization() {
  std::string acpc_gamedef = R""""(
gameDEF
limit
numplayers = 2
numrounds = 1
blind = 5 10
raisesize = 10 10 20
MAXRAISES = 2 2 3
NUMSUITS = 4
NUMRANKS = 5
nUmHoLeCARds = 1
numBoardCARDS = 0 2 1
end GameDef
)"""";

  std::string open_spiel_game_state =
      GamedefToOpenSpielParameters(acpc_gamedef);

  SPIEL_CHECK_TRUE(absl::StrContains(open_spiel_game_state, ",numPlayers=2,"));
  SPIEL_CHECK_TRUE(absl::StrContains(open_spiel_game_state, ",numRounds=1,"));
  SPIEL_CHECK_TRUE(absl::StrContains(open_spiel_game_state, ",blind=5 10,"));
  SPIEL_CHECK_TRUE(
      absl::StrContains(open_spiel_game_state, ",raiseSize=10 10 20,"));
  SPIEL_CHECK_TRUE(absl::StrContains(open_spiel_game_state, ",numSuits=4,"));
  SPIEL_CHECK_TRUE(absl::StrContains(open_spiel_game_state, ",numRanks=5,"));
  SPIEL_CHECK_TRUE(
      absl::StrContains(open_spiel_game_state, ",numHoleCards=1,"));
}

}  // namespace logic
}  // namespace universal_poker
}  // namespace open_spiel

int main(int argc, char **argv) {
  open_spiel::universal_poker::logic::
      TestGamedefToOpenSpielParametersEasyCase();
  open_spiel::universal_poker::logic::
      TestGamedefToOpenSpielParametersNormalizesKeyOnlyLines();
  open_spiel::universal_poker::logic::
      TestGamedefToOpenSpielParametersMultiRoundValueEdgeCase();
  open_spiel::universal_poker::logic::
      TestGamedefToOpenSpielParametersRemovesUnneededLines();
  open_spiel::universal_poker::logic::
      TestGamedefToOpenSpielParametersNormalizesCapitalization();
}
