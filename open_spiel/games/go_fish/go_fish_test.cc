// Copyright 2026 DeepMind Technologies Limited
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

#include "open_spiel/games/go_fish/go_fish.h"

#include <algorithm>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "open_spiel/abseil-cpp/absl/types/span.h"
#include "open_spiel/game_parameters.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/tests/basic_tests.h"
#include "open_spiel/utils/init.h"

namespace open_spiel {
namespace go_fish {
namespace {

namespace testing = open_spiel::testing;

void BasicGoFishTests() {
  testing::LoadGameTest("go_fish");
  testing::ChanceOutcomesTest(*LoadGame("go_fish"));
  testing::RandomSimTest(*LoadGame("go_fish"), 100);
  for (Player players = 2; players <= 5; players++) {
    testing::RandomSimTest(
        *LoadGame("go_fish", {{"players", GameParameter(players)}}), 100);
  }
}

void SerializationTests() {
  auto game = LoadGame("go_fish");

  // Default board position.
  std::unique_ptr<State> state = game->NewInitialState();
  std::shared_ptr<State> deserialized_state =
      game->DeserializeState(state->Serialize());
  SPIEL_CHECK_EQ(state->ToString(), deserialized_state->ToString());
}

void GoFishCoreRuleTests() {
  std::shared_ptr<const Game> game = LoadGame("go_fish");
  const GoFishGame* gfg = static_cast<const GoFishGame*>(game.get());

  // Test Case 1: Successful Ask and Book Formation
  {
    // Player 0 has three 'a's, Player 1 has one 'a'.
    std::string start_state_str =
        "Ask\n0\na3b1c1d1e1:0\na1f1g1h1i1j1k1l1:0\nb3c3d3e3f3g3h3i3j3k3l3m4";
    std::unique_ptr<State> state = gfg->NewSpecificState(start_state_str);
    const GoFishState* gf_state = static_cast<const GoFishState*>(state.get());
    SPIEL_CHECK_EQ(state->CurrentPlayer(), 0);

    // Player 0 asks Player 1 for 'a'. Action: {player_to_ask=1, rank='a'}
    Action ask_action = gfg->AskStringToAction("1,a");
    const std::vector<Action> legal_actions = state->LegalActions();
    SPIEL_CHECK_TRUE(std::find(legal_actions.begin(), legal_actions.end(),
                               ask_action) != legal_actions.end());
    state->ApplyAction(ask_action);

    // Verify state after the successful ask.
    // Player 0 should have formed a book of 'a'.
    // Player 0's hand size reduced by 3 (from a3) + 1 (a1) = 4, then booked.
    SPIEL_CHECK_EQ(gf_state->PlayerCounts(0), 4);
    SPIEL_CHECK_EQ(gf_state->PlayerCounts(1), 7);
    SPIEL_CHECK_EQ(gf_state->PlayerBooks()[0], 1);
    SPIEL_CHECK_EQ(gf_state->PlayerBooks()[1], 0);
    SPIEL_CHECK_EQ(state->Rewards()[0], 0.0);
    SPIEL_CHECK_EQ(state->Rewards()[1], 0.0);
    // Current player remains Player 0 because cards were received.
    SPIEL_CHECK_EQ(state->CurrentPlayer(), 0);
  }

  // Test Case 2: Unsuccessful Ask (Go Fish)
  {
    // Player 0 has a 'b', asks Player 1 for 'b', but Player 1 has no 'b's.
    std::string start_state_str =
        "Ask\n0\na1b1c1d1e1f1g1:0\na3h1i1j1k1:0\nb3c3d3e3f3g3h3i3j3k3l4m4";
    std::unique_ptr<State> state = gfg->NewSpecificState(start_state_str);
    const GoFishState* gf_state = static_cast<const GoFishState*>(state.get());
    SPIEL_CHECK_EQ(state->CurrentPlayer(), 0);

    // Player 0 asks Player 1 for 'b'. Action: {player_to_ask=1, rank='b'}
    Action ask_action = gfg->AskStringToAction("1,b");
    const std::vector<Action> legal_actions = state->LegalActions();
    SPIEL_CHECK_TRUE(std::find(legal_actions.begin(), legal_actions.end(),
                               ask_action) != legal_actions.end());
    state->ApplyAction(ask_action);

    // State should now be a chance node (fishing).
    SPIEL_CHECK_TRUE(state->IsChanceNode());

    // Player 0 draws a card. We draw 'c' (rank 2) which is not 'b',
    // so turn passes.
    Action draw_action = 2;  // rank 'c'
    state->ApplyAction(draw_action);

    // The turn should pass to Player 1.
    SPIEL_CHECK_EQ(state->CurrentPlayer(), 1);
    SPIEL_CHECK_EQ(gf_state->PoolSize(), 37);  // One card drawn.
  }

  // Test Case 3: Illegal Move - Asking for a card not held
  {
    // Player 0 has no 'c's.
    std::string start_state_str =
        "Ask\n0\na1b1d1e1f1g1h1:0\na3c1i1j1k1:0\nb3c3d3e3f3g3h3i3j3k3l4m4";
    std::unique_ptr<State> state = gfg->NewSpecificState(start_state_str);
    SPIEL_CHECK_EQ(state->CurrentPlayer(), 0);

    // Player 0 tries to ask Player 1 for 'c'.
    Action illegal_action = gfg->AskStringToAction("1,c");
    const std::vector<Action> legal_actions = state->LegalActions();
    SPIEL_CHECK_FALSE(std::find(legal_actions.begin(), legal_actions.end(),
                                illegal_action) != legal_actions.end());
  }

  // Test Case 4: Game Ends When All Books Are Made
  {
    // Simulate a state where only one more book (rank 'm') is needed to end
    // the game. Player 0 has three 'm's, Player 1 has one 'm'. We set the
    // booked counts to 6 for both players, meaning 12 books are already made.
    std::string start_state_str =
        "Ask\n0\nm3:6\nm1:6\n";
    std::unique_ptr<State> state = gfg->NewSpecificState(start_state_str);
    SPIEL_CHECK_EQ(state->CurrentPlayer(), 0);
    SPIEL_CHECK_FALSE(state->IsTerminal());

    // Player 0 asks Player 1 for 'm'.
    Action ask_action = gfg->AskStringToAction("1,m");
    const std::vector<Action> legal_actions = state->LegalActions();
    SPIEL_CHECK_TRUE(std::find(legal_actions.begin(), legal_actions.end(),
                               ask_action) != legal_actions.end());
    state->ApplyAction(ask_action);

    // Verify the game is now terminal.
    SPIEL_CHECK_TRUE(state->IsTerminal());
    // Verify rewards based on total books.
    SPIEL_CHECK_EQ(state->Rewards()[0], 1.0);  // Player 0 wins (7 books vs 6)
    SPIEL_CHECK_EQ(state->Rewards()[1], -1.0);
  }

  // Test Case 5: Player Must Fish When Deck is Empty (3 players)
  {
    std::shared_ptr<const Game> game3 =
        LoadGame("go_fish", {{"players", GameParameter(3)}});
    const GoFishGame* gfg3 = static_cast<const GoFishGame*>(game3.get());
    std::string start_state_str = "Ask\n0\na2:4\nb2:4\na2b2:3\n";
    std::unique_ptr<State> state = gfg3->NewSpecificState(start_state_str);
    const GoFishState* gf_state = static_cast<const GoFishState*>(state.get());
    SPIEL_CHECK_EQ(state->CurrentPlayer(), 0);
    SPIEL_CHECK_EQ(gf_state->PoolSize(), 0);
    SPIEL_CHECK_FALSE(state->IsTerminal());

    // Player 0 asks Player 1 for 'a'. Player 1 does not have 'a'.
    Action ask_action = gfg3->AskStringToAction("1,a");
    const std::vector<Action> legal_actions = state->LegalActions();
    SPIEL_CHECK_TRUE(std::find(legal_actions.begin(), legal_actions.end(),
                               ask_action) != legal_actions.end());
    state->ApplyAction(ask_action);

    // Since the pool is empty and turn passes, game is not terminal.
    SPIEL_CHECK_FALSE(state->IsTerminal());
    SPIEL_CHECK_EQ(state->CurrentPlayer(), 1);
  }
}

template <typename T>
void PrintMatrix(const std::vector<std::vector<T>>& m) {
  for (const auto& row : m) {
    for (const auto& x : row) std::cout << x << " ";
    std::cout << std::endl;
  }
}

void ObservationTensorTests() {
  std::shared_ptr<const Game> game = LoadGame("go_fish");
  std::string start =
      "Ask\n0\nc1d1f1g2h1i1:0\nb2d1g1l2m1:0\na4b2c3d2e4f3g1h3i3j4k4l2m3";
  const GoFishGame* gfg = static_cast<const GoFishGame*>(game.get());
  std::unique_ptr<State> state = gfg->NewSpecificState(start);
  state->ApplyAction(gfg->AskStringToAction("1,g"));
  state->ApplyAction(gfg->AskStringToAction("1,d"));
  state->ApplyAction(gfg->AskStringToAction("1,h"));
  state->ApplyAction(gfg->FishStringToAction("g"));
  state->ApplyAction(gfg->AskStringToAction("0,b"));
  // do some more actions so drawn_since isn;t always 0
  state->ApplyAction(gfg->FishStringToAction("a"));
  state->ApplyAction(gfg->AskStringToAction("1,i"));
  state->ApplyAction(gfg->FishStringToAction("m"));
  state->ApplyAction(gfg->AskStringToAction("0,a"));
  state->ApplyAction(gfg->FishStringToAction("a"));
  state->ApplyAction(gfg->AskStringToAction("0,m"));

  auto shape = game->ObservationTensorShape();
  std::vector<float> v(game->ObservationTensorSize());
  state->ObservationTensor(state->CurrentPlayer(), absl::MakeSpan(v));
  int offset = 0;
  float eps = 1e-6;
  // cards for the player on move
  SPIEL_CHECK_EQ(v[offset++], 2.0 / 4);  // a
  SPIEL_CHECK_EQ(v[offset++], 2.0 / 4);  // b
  SPIEL_CHECK_EQ(v[offset++], 0.0);      // c
  SPIEL_CHECK_EQ(v[offset++], 0.0);      // d
  SPIEL_CHECK_EQ(v[offset++], 0.0);      // e
  SPIEL_CHECK_EQ(v[offset++], 0.0);      // f
  SPIEL_CHECK_EQ(v[offset++], 0.0);      // g
  SPIEL_CHECK_EQ(v[offset++], 0.0);      // h
  SPIEL_CHECK_EQ(v[offset++], 0.0);      // i
  SPIEL_CHECK_EQ(v[offset++], 0.0);      // j
  SPIEL_CHECK_EQ(v[offset++], 0.0);      // k
  SPIEL_CHECK_EQ(v[offset++], 2.0 / 4);  // l
  SPIEL_CHECK_EQ(v[offset++], 2.0 / 4);  // m
  // phase
  SPIEL_CHECK_EQ(v[offset++], 0.0);  // kDeal
  SPIEL_CHECK_EQ(v[offset++], 1.0);  // kAsk
  SPIEL_CHECK_EQ(v[offset++], 0.0);  // Fish
  SPIEL_CHECK_EQ(v[offset++], 0.0);  // kTerminal
  // pool size
  SPIEL_CHECK_FLOAT_NEAR(v[offset++], 34.0 / 52, eps);  // pool size
  // booked is boolean
  SPIEL_CHECK_EQ(v[offset++], 0.0);  // a
  SPIEL_CHECK_EQ(v[offset++], 0.0);  // b
  SPIEL_CHECK_EQ(v[offset++], 0.0);  // c
  SPIEL_CHECK_EQ(v[offset++], 0.0);  // d
  SPIEL_CHECK_EQ(v[offset++], 0.0);  // e
  SPIEL_CHECK_EQ(v[offset++], 0.0);  // f
  SPIEL_CHECK_EQ(v[offset++], 1.0);  // g
  SPIEL_CHECK_EQ(v[offset++], 0.0);  // h
  SPIEL_CHECK_EQ(v[offset++], 0.0);  // i
  SPIEL_CHECK_EQ(v[offset++], 0.0);  // j
  SPIEL_CHECK_EQ(v[offset++], 0.0);  // k
  SPIEL_CHECK_EQ(v[offset++], 0.0);  // l
  SPIEL_CHECK_EQ(v[offset++], 0.0);  // m
  // values for players ordered by pid

  SPIEL_CHECK_EQ(v[offset++], 0.0);                    // p0 on move
  SPIEL_CHECK_FLOAT_NEAR(v[offset++], 1.0 / 13, eps);  // p0 books
  SPIEL_CHECK_FLOAT_NEAR(v[offset++], 6.0 / 52, eps);  // p0 card count

  // did ask is count of times player asked normalized to suits * ranks
  // was asked is 0 or 1
  // drawn since is normalized to cards * ranks
  // min is normalized to suits
  SPIEL_CHECK_EQ(v[offset++], 0.0);  // p0 did ask for a
  SPIEL_CHECK_EQ(v[offset++], 1.0);  // p0 was asked for a
  SPIEL_CHECK_EQ(v[offset++], 0.0);  // drawn since was asked for a
  SPIEL_CHECK_EQ(v[offset++], 0.0);  // player_min a
                                     //
  SPIEL_CHECK_EQ(v[offset++], 0.0);  // p0 did ask for b
  SPIEL_CHECK_EQ(v[offset++], 1.0);  // p0 was asked for b
  // drawn since
  SPIEL_CHECK_FLOAT_NEAR(v[offset++], 1.0 / 52, eps);
  SPIEL_CHECK_EQ(v[offset++], 0.0);  // player_min b

  SPIEL_CHECK_EQ(v[offset++], 0.0);  // p0 did ask for c
  SPIEL_CHECK_EQ(v[offset++], 0.0);  // p0 was asked for c
  SPIEL_CHECK_EQ(v[offset++], 0.0);  // drawn since was asked for c
  SPIEL_CHECK_EQ(v[offset++], 0.0);  // player_min c

  SPIEL_CHECK_FLOAT_NEAR(v[offset++], 1.0 / 52, eps);  // p0 did ask for d
  SPIEL_CHECK_EQ(v[offset++], 0.0);                    // p0 was asked for d
  SPIEL_CHECK_EQ(v[offset++], 0.0);      // drawn since was asked for d
  SPIEL_CHECK_EQ(v[offset++], 2.0 / 4);  // player_min d

  SPIEL_CHECK_EQ(v[offset++], 0.0);  // p0 did ask for e
  SPIEL_CHECK_EQ(v[offset++], 0.0);  // p0 was asked for e
  SPIEL_CHECK_EQ(v[offset++], 0.0);  // drawn since was asked for e
  SPIEL_CHECK_EQ(v[offset++], 0.0);  // player_min e

  SPIEL_CHECK_EQ(v[offset++], 0.0);  // p0 did ask for f
  SPIEL_CHECK_EQ(v[offset++], 0.0);  // p0 was asked for f
  SPIEL_CHECK_EQ(v[offset++], 0.0);  // drawn since was asked for f
  SPIEL_CHECK_EQ(v[offset++], 0.0);  // player_min f

  SPIEL_CHECK_FLOAT_NEAR(v[offset++], 1.0 / 52, eps);  // p0 did ask for g
  SPIEL_CHECK_EQ(v[offset++], 0.0);                    // p0 was asked for g
  SPIEL_CHECK_EQ(v[offset++], 0.0);  // drawn since was asked for g
  SPIEL_CHECK_EQ(v[offset++], 0.0);  // player_min g

  SPIEL_CHECK_FLOAT_NEAR(v[offset++], 1.0 / 52, eps);  // p0 did ask for h
  SPIEL_CHECK_EQ(v[offset++], 0.0);                    // p0 was asked for h
  SPIEL_CHECK_EQ(v[offset++], 0.0);      // drawn since was asked for h
  SPIEL_CHECK_EQ(v[offset++], 1.0 / 4);  // player_min h
}

}  // namespace
}  // namespace go_fish
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::Init(argv[0], &argc, &argv, true);
  open_spiel::go_fish::BasicGoFishTests();
  open_spiel::go_fish::ObservationTensorTests();
  open_spiel::go_fish::SerializationTests();
  open_spiel::go_fish::GoFishCoreRuleTests();
}
