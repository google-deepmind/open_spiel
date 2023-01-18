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

#include "open_spiel/games/squadro.h"

#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/tests/basic_tests.h"

namespace open_spiel {
namespace squadro {
namespace {

namespace testing = open_spiel::testing;

void BasicSquadroTests() {
  testing::LoadGameTest("squadro");
  testing::NoChanceOutcomesTest(*LoadGame("squadro"));
}

void InvertDirection() {
  std::shared_ptr<const Game> game = LoadGame("squadro");
  SquadroState state(game,
      "....v..\n"
      ">....v.\n"
      ".^^.<..\n"
      "....>..\n"
      "......<\n"
      "....>..\n"
      "...^...\n"
      "C0\n");
  
  state.ApplyAction(0);
  SPIEL_CHECK_EQ(state.ToString(),
                 ".v..v..\n"
                ">....v.\n"
                "..^.<..\n"
                "....>..\n"
                "......<\n"
                "....>..\n"
                "...^...\n"
                "C1"); 
  state.ApplyAction(2);
  SPIEL_CHECK_EQ(state.ToString(),
                 ".v..v..\n"
                ">....v.\n"
                "..^.<..\n"
                "......<\n"
                "......<\n"
                "....>..\n"
                "...^...\n"
                "C0"); 
}

void JumpOpponentTokens() {
  std::shared_ptr<const Game> game = LoadGame("squadro");
  SquadroState state(game,
                     ".......\n"
                     "....>v.\n"
                     ".v^.<..\n"
                     "....>..\n"
                     "...>^..\n"
                     "....>..\n"
                     "...^...\n"
                     "C1\n");
  state.ApplyAction(4);
  SPIEL_CHECK_EQ(state.ToString(),
                 ".....v.\n"
                 "......<\n"
                 ".v^.<..\n"
                 "....>..\n"
                 "...>^..\n"
                 "....>..\n"
                 "...^...\n"
                 "C0");
    state.ApplyAction(3);
    SPIEL_CHECK_EQ(state.ToString(),
                   ".....v.\n"
                   "....^.<\n"
                   ".v^...<\n"
                   ">......\n"
                   "...>...\n"
                   "....>..\n"
                   "...^...\n"
                   "C1");
    state.ApplyAction(4);
    SPIEL_CHECK_EQ(state.ToString(),
                   ".....v.\n"
                   "...<...\n"
                   ".v^...<\n"
                   ">......\n"
                   "...>...\n"
                   "....>..\n"
                   "...^^..\n"
                   "C0");
    state.ApplyAction(3);
    SPIEL_CHECK_EQ(state.ToString(),
                   ".....v.\n"
                   "...<...\n"
                   ".v^...<\n"
                   ">......\n"
                   "...>^..\n"
                   ">......\n"
                   "...^...\n"
                   "C1");
    state.ApplyAction(3);
    state.ApplyAction(2);
    state.ApplyAction(3);
    state.ApplyAction(4);
    state.ApplyAction(3);
    state.ApplyAction(4);
    state.ApplyAction(3);
    SPIEL_CHECK_EQ(state.ToString(),
                   ".v.....\n"
                   "...<...\n"
                   ".....v.\n"
                   ">..^...\n"
                   ">...^..\n"
                   ">......\n"
                   "..^....\n"
                   "C0");
}

void CheckTerminal() {
  std::shared_ptr<const Game> game = LoadGame("squadro");
  SquadroState state(game,
                     ".......\n"
                     "....>..\n"
                     "...^<..\n"
                     "..<....\n"
                     "......<\n"
                     ".>...v.\n"
                     ".......\n"
                     "C0\n");
  state.ApplyAction(3);
  SPIEL_CHECK_EQ(state.ToString(),
                 ".......\n"
                 "....>..\n"
                 "...^<..\n"
                 "..<....\n"
                 "......<\n"
                 ".>.....\n"
                 ".......\n"
                 "C2");
   SPIEL_CHECK_TRUE(state.IsTerminal());
}

void BasicSerializationTest() {
  std::shared_ptr<const Game> game = LoadGame("squadro");
  std::unique_ptr<State> state = game->NewInitialState();
  std::unique_ptr<State> state2 = game->DeserializeState(state->Serialize());
  SPIEL_CHECK_EQ(state->ToString(), state2->ToString());
}

}  // namespace
}  // namespace squadro
}  // namespace open_spiel

int main(int argc, char **argv) {
  open_spiel::squadro::BasicSquadroTests();
  open_spiel::squadro::InvertDirection();
  open_spiel::squadro::JumpOpponentTokens();
  open_spiel::squadro::BasicSerializationTest();
}