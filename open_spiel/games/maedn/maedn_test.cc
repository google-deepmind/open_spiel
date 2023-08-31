// Copyright 2022 DeepMind Technologies Limited
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

#include "open_spiel/games/maedn/maedn.h"

#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/tests/basic_tests.h"

namespace open_spiel {
namespace maedn {
namespace {

namespace testing = open_spiel::testing;

void BasicMaednTests() {
  testing::LoadGameTest("maedn");

  std::shared_ptr<const Game> game =
      LoadGame("maedn", {{"players", GameParameter(2)},
                         {"twoPlayersOpposite", GameParameter(true)}});

  testing::RandomSimTest(*game, 100);
  testing::RandomSimTestWithUndo(*game, 100);

  for (int players = 2; players <= 4; players++) {
    game = LoadGame("maedn", {{"players", GameParameter(players)},
                              {"twoPlayersOpposite", GameParameter(false)}});

    testing::RandomSimTest(*game, 100);
    testing::RandomSimTestWithUndo(*game, 100);
  }
}

const char* MINIMAL_WINS_EXPECTED_TERMINAL_STATES[] = {
    // 2 players side-by-side, player 1 wins
    ". .     o-o-S     2 2\n"
    ". .     o . o     2 2\n"
    "        o . o        \n"
    "        o . o        \n"
    "S-o-o-o-o . o-o-o-o-o\n"
    "o 1 1 1 1   . . . . o\n"
    "o-o-o-o-o . o-o-o-o-S\n"
    "        o . o        \n"
    "        o . o        \n"
    ". .     o . o     . .\n"
    ". .     S-o-o     . .\n"
    "Turn: *\n"
    "Dice: \n",
    // 2 players side-by-side, player 2 wins
    "1 1     o-o-S     . .\n"
    "1 1     o 2 o     . .\n"
    "        o 2 o        \n"
    "        o 2 o        \n"
    "S-o-o-o-o 2 o-o-o-o-o\n"
    "o . . . .   . . . . o\n"
    "o-o-o-o-o . o-o-o-o-S\n"
    "        o . o        \n"
    "        o . o        \n"
    ". .     o . o     . .\n"
    ". .     S-o-o     . .\n"
    "Turn: *\n"
    "Dice: \n",
    // 2 players opposite sides, player 1 wins
    ". .     o-o-S     . .\n"
    ". .     o . o     . .\n"
    "        o . o        \n"
    "        o . o        \n"
    "S-o-o-o-o . o-o-o-o-o\n"
    "o 1 1 1 1   . . . . o\n"
    "o-o-o-o-o . o-o-o-o-S\n"
    "        o . o        \n"
    "        o . o        \n"
    ". .     o . o     2 2\n"
    ". .     S-o-o     2 2\n"
    "Turn: *\n"
    "Dice: \n",
    // 2 players opposite sides, player 2 wins
    "1 1     o-o-S     . .\n"
    "1 1     o . o     . .\n"
    "        o . o        \n"
    "        o . o        \n"
    "S-o-o-o-o . o-o-o-o-o\n"
    "o . . . .   2 2 2 2 o\n"
    "o-o-o-o-o . o-o-o-o-S\n"
    "        o . o        \n"
    "        o . o        \n"
    ". .     o . o     . .\n"
    ". .     S-o-o     . .\n"
    "Turn: *\n"
    "Dice: \n",
    // 3 players, player 1 wins
    ". .     o-o-S     2 2\n"
    ". .     o . o     2 2\n"
    "        o . o        \n"
    "        o . o        \n"
    "S-o-o-o-o . o-o-o-o-o\n"
    "o 1 1 1 1   . . . . o\n"
    "o-o-o-o-o . o-o-o-o-S\n"
    "        o . o        \n"
    "        o . o        \n"
    ". .     o . o     3 3\n"
    ". .     S-o-o     3 3\n"
    "Turn: *\n"
    "Dice: \n",
    // 3 players, player 2 wins
    "1 1     o-o-S     . .\n"
    "1 1     o 2 o     . .\n"
    "        o 2 o        \n"
    "        o 2 o        \n"
    "S-o-o-o-o 2 o-o-o-o-o\n"
    "o . . . .   . . . . o\n"
    "o-o-o-o-o . o-o-o-o-S\n"
    "        o . o        \n"
    "        o . o        \n"
    ". .     o . o     3 3\n"
    ". .     S-o-o     3 3\n"
    "Turn: *\n"
    "Dice: \n",
    // 3 players, player 3 wins
    "1 1     o-o-S     2 2\n"
    "1 1     o . o     2 2\n"
    "        o . o        \n"
    "        o . o        \n"
    "S-o-o-o-o . o-o-o-o-o\n"
    "o . . . .   3 3 3 3 o\n"
    "o-o-o-o-o . o-o-o-o-S\n"
    "        o . o        \n"
    "        o . o        \n"
    ". .     o . o     . .\n"
    ". .     S-o-o     . .\n"
    "Turn: *\n"
    "Dice: \n",
    // 4 players, player 1 wins
    ". .     o-o-S     2 2\n"
    ". .     o . o     2 2\n"
    "        o . o        \n"
    "        o . o        \n"
    "S-o-o-o-o . o-o-o-o-o\n"
    "o 1 1 1 1   . . . . o\n"
    "o-o-o-o-o . o-o-o-o-S\n"
    "        o . o        \n"
    "        o . o        \n"
    "4 4     o . o     3 3\n"
    "4 4     S-o-o     3 3\n"
    "Turn: *\n"
    "Dice: \n",
    // 4 players, player 2 wins
    "1 1     o-o-S     . .\n"
    "1 1     o 2 o     . .\n"
    "        o 2 o        \n"
    "        o 2 o        \n"
    "S-o-o-o-o 2 o-o-o-o-o\n"
    "o . . . .   . . . . o\n"
    "o-o-o-o-o . o-o-o-o-S\n"
    "        o . o        \n"
    "        o . o        \n"
    "4 4     o . o     3 3\n"
    "4 4     S-o-o     3 3\n"
    "Turn: *\n"
    "Dice: \n",
    // 4 players, player 3 wins
    "1 1     o-o-S     2 2\n"
    "1 1     o . o     2 2\n"
    "        o . o        \n"
    "        o . o        \n"
    "S-o-o-o-o . o-o-o-o-o\n"
    "o . . . .   3 3 3 3 o\n"
    "o-o-o-o-o . o-o-o-o-S\n"
    "        o . o        \n"
    "        o . o        \n"
    "4 4     o . o     . .\n"
    "4 4     S-o-o     . .\n"
    "Turn: *\n"
    "Dice: \n",
    // 4 players, player 4 wins
    "1 1     o-o-S     2 2\n"
    "1 1     o . o     2 2\n"
    "        o . o        \n"
    "        o . o        \n"
    "S-o-o-o-o . o-o-o-o-o\n"
    "o . . . .   . . . . o\n"
    "o-o-o-o-o 4 o-o-o-o-S\n"
    "        o 4 o        \n"
    "        o 4 o        \n"
    ". .     o 4 o     3 3\n"
    ". .     S-o-o     3 3\n"
    "Turn: *\n"
    "Dice: \n",
};

void PlayMinimalGameToWin(int players, bool twoPlayersOpposite, int ply,
                          int terminalStateScenarioNumber) {
  std::shared_ptr<const Game> game = LoadGame(
      "maedn", {{"players", GameParameter(players)},
                {"twoPlayersOpposite", GameParameter(twoPlayersOpposite)}});

  auto state = game->NewInitialState();

  // other players do nothing
  for (int i = 0; i < ply; i++) {
    state->ApplyAction(0);  // dice 1 for other player
    state->ApplyAction(0);  // player passes
  }

  for (int i = 0; i < 4; i++) {
    state->ApplyAction(5);  // dice 6
    state->ApplyAction(1);  // bring in piece
    state->ApplyAction(5);  // dice 6
    state->ApplyAction(2);
    state->ApplyAction(5);  // dice 6
    state->ApplyAction(8);
    state->ApplyAction(5);  // dice 6
    state->ApplyAction(14);
    state->ApplyAction(5);  // dice 6
    state->ApplyAction(20);
    state->ApplyAction(5);  // dice 6
    state->ApplyAction(26);
    state->ApplyAction(5);  // dice 6
    state->ApplyAction(32);
    if (i == 0 || i == 1) {
      state->ApplyAction(5);  // dice 6
      state->ApplyAction(38);
    }
    if (i == 0) {
      state->ApplyAction(0);  // dice 1
      state->ApplyAction(44);

      // other players do nothing
      for (int i = 0; i < players - 1; i++) {
        state->ApplyAction(0);  // dice 1 for other player
        state->ApplyAction(0);  // player passes
      }
    } else if (i == 2) {
      state->ApplyAction(4);  // dice 5
      state->ApplyAction(38);

      // other players do nothing
      for (int i = 0; i < players - 1; i++) {
        state->ApplyAction(0);  // dice 1 for other player
        state->ApplyAction(0);  // player passes
      }
    }
  }

  SPIEL_CHECK_FALSE(state->IsTerminal());
  state->ApplyAction(3);  // dice 4
  state->ApplyAction(38);

  std::cout << "Testing minimal win for " << players << "players, player "
            << ply << "wins" << std::endl
            << "Terminal state:" << std::endl
            << state->ToString() << std::endl;

  SPIEL_CHECK_TRUE(state->IsTerminal());

  std::vector<double> returns = state->Returns();
  for (int i = 0; i < players; i++) {
    double expected = i == ply ? players - 1.0 : -1.0;

    SPIEL_CHECK_EQ(returns[i], expected);
  }

  SPIEL_CHECK_EQ(
      state->ToString(),
      MINIMAL_WINS_EXPECTED_TERMINAL_STATES[terminalStateScenarioNumber]);
}

void MinimalGameToWin() {
  // Test for all constellations whether for any player the
  // minimal winning scenario works as expected.
  // Scenarios: 2p side-by-side, 2p opposite sides, 3p, 4p,
  // for each participating player.

  int terminal_state_scenario_number = 0;
  for (int scenario = 0; scenario < 4; scenario++) {
    int players;
    bool two_players_opposite = false;
    if (scenario == 0) {
      players = 2;
      two_players_opposite = false;
    } else if (scenario == 1) {
      players = 2;
      two_players_opposite = true;
    } else {
      players = scenario + 1;
    }

    for (int ply = 0; ply < players; ply++) {
      PlayMinimalGameToWin(players, two_players_opposite, ply,
                           terminal_state_scenario_number++);
    }
  }
}

void ObservationTensorTest(const State &state) {
  std::shared_ptr<const Game> game = state.GetGame();

  int players = state.NumPlayers();
  for (int ply = 0; ply < players; ply++) {
    std::vector<float> tensor = state.ObservationTensor(ply);

    std::unique_ptr<State> state2_tmp = game->NewInitialState();
    std::unique_ptr<MaednState> state2(
        static_cast<MaednState *>(state2_tmp.release()));

    state2->FromObservationTensor(ply, absl::MakeSpan(tensor), 0, 0);

    // std::cout << "Player: " << ply << std::endl;
    // std::cout << "State:" << std::endl << state.ToString() << std::endl;
    // std::cout << "State2:" << std::endl << state2->ToString() << std::endl;
    // std::cout << "Tensor:" << std::endl << tensor << std::endl;
    SPIEL_CHECK_EQ(state.ToString(), state2->ToString());
  }
}

void CheckObservationTensor() {
  std::shared_ptr<const Game> game =
      LoadGame("maedn", {{"players", GameParameter(2)},
                         {"twoPlayersOpposite", GameParameter(true)}});

  testing::RandomSimTest(*game, 100, true, false, true, &ObservationTensorTest);

  for (int players = 2; players <= 4; players++) {
    std::shared_ptr<const Game> game =
        LoadGame("maedn", {{"players", GameParameter(players)},
                           {"twoPlayersOpposite", GameParameter(false)}});

    testing::RandomSimTest(*game, 100, true, false, true,
                           &ObservationTensorTest);
  }
}

void BasicSerializationTest() {
  std::shared_ptr<const Game> game = LoadGame("maedn");
  std::unique_ptr<State> state = game->NewInitialState();
  std::unique_ptr<State> state2 = game->DeserializeState(state->Serialize());
  SPIEL_CHECK_EQ(state->ToString(), state2->ToString());
}

}  // namespace
}  // namespace maedn
}  // namespace open_spiel

int main(int argc, char **argv) {
  open_spiel::maedn::BasicMaednTests();
  open_spiel::maedn::MinimalGameToWin();
  open_spiel::maedn::BasicSerializationTest();
  open_spiel::maedn::CheckObservationTensor();
}
