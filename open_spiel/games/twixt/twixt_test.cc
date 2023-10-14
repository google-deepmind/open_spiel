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

#include "open_spiel/spiel.h"
#include "open_spiel/tests/basic_tests.h"
#include "open_spiel/games/twixt/twixt.h"

namespace open_spiel {
namespace twixt {
namespace {

namespace testing = open_spiel::testing;

void BasicTwixTTests() {
  testing::LoadGameTest("twixt");
  testing::NoChanceOutcomesTest(*LoadGame("twixt"));
  testing::RandomSimTest(*LoadGame("twixt"), 100);
}

class TestException : public std::exception {
 public:
    std::string error_msg_ = "";
    char * what() {
        return &error_msg_[0];
    }

    explicit TestException(const std::string& error_msg) {
      error_msg_ = error_msg;
    }
};

void ErrorHandler(const std::string& error_msg) {
  std::cerr << "Twixt Fatal Error: " << error_msg << std::endl << std::flush;
  throw TestException(error_msg);
}



void ParameterTest() {
  std::string game_name = "twixt";
  open_spiel::GameParameters params;
  std::shared_ptr<const open_spiel::Game> game;
  // ok: ansi_color_output=true
  params.insert({"ansi_color_output", open_spiel::GameParameter(true, false)});
  game = open_spiel::LoadGame(game_name, params);
  params.clear();

  // ok: board_size=10
  params.insert({"board_size", open_spiel::GameParameter(10, false)});
  game = open_spiel::LoadGame(game_name, params);
  params.clear();

  // too big: board_size=30
  params.insert({"board_size", open_spiel::GameParameter(30, false)});
  try {
    game = open_spiel::LoadGame(game_name, params);
  } catch (TestException e) {
    std::string expected = "board_size out of range [5..24]: 30";
    SPIEL_CHECK_EQ(expected, std::string(e.what()));
  }
  params.clear();

  // too small: board_size=3
  params.insert({"board_size", open_spiel::GameParameter(3, false)});
  try {
    game = open_spiel::LoadGame(game_name, params);
  } catch (TestException e) {
    std::string expected = "board_size out of range [5..24]: 3";
    SPIEL_CHECK_EQ(expected, std::string(e.what()));
  }

  // invalid param: bad_param
  params.insert({"bad_param", open_spiel::GameParameter(3, false)});
  try {
    game = open_spiel::LoadGame(game_name, params);
  } catch (TestException e) {
    std::string expected = "Unknown parameter 'bad_param'. " \
      "Available parameters are: ansi_color_output, board_size";
    SPIEL_CHECK_EQ(expected, std::string(e.what()));
  }
}

bool IsLegalAction(const std::vector<open_spiel::Action> v,
    open_spiel::Action action) {
  return std::find(v.begin(), v.end(), action) != v.end();
}

void PrintLegalActions(const std::vector<open_spiel::Action> v,
    open_spiel::Player p) {
  std::cout << p << ": ";
  for (int i = 0; i < v.size(); i++) {
    std::cout << v.at(i) << ' ';
  }
  std::cout << std::endl;
}

void SwapTest() {
  std::shared_ptr<const open_spiel::Game> game = open_spiel::LoadGame("twixt");
  auto state = game->NewInitialState();
  // player 0 plays action 19: [2,3] = c5
  SPIEL_CHECK_EQ(0, state->CurrentPlayer());
  SPIEL_CHECK_TRUE(IsLegalAction(state->LegalActions(), 11));
  state->ApplyAction(19);

  // player 1 plays action 19: [2,3] = c5 (SWAP rule)
  SPIEL_CHECK_EQ(1, state->CurrentPlayer());
  state->ApplyAction(19);

  // => [3,5] od3 replaces [2,3] xc5; c5 is empty again and d3 is occupied
  SPIEL_CHECK_TRUE(IsLegalAction(state->LegalActions(), 19));   // c5
  SPIEL_CHECK_FALSE(IsLegalAction(state->LegalActions(), 29));  // d3

  // player 0 plays action 36: [4,4] = e4
  SPIEL_CHECK_EQ(0, state->CurrentPlayer());
  state->ApplyAction(36);

  SPIEL_CHECK_TRUE(IsLegalAction(state->LegalActions(), 19));   // c5
  SPIEL_CHECK_FALSE(IsLegalAction(state->LegalActions(), 29));  // d3
  SPIEL_CHECK_FALSE(IsLegalAction(state->LegalActions(), 36));  // e4
}

void LegalActionsTest() {
  std::shared_ptr<const open_spiel::Game> game = open_spiel::LoadGame("twixt");
  auto state = game->NewInitialState();
  SPIEL_CHECK_FALSE(state->IsTerminal());
  // 48*/48 legal actions
  SPIEL_CHECK_EQ(48, state->LegalActions().size());

  state->ApplyAction(21);   // player 0: xc3
  // 47/48* legal actions; player 1 could play c3 to swap
  SPIEL_CHECK_EQ(48, state->LegalActions().size());

  state->ApplyAction(38);  // player 1: oe2
  // 46*/46 legal actions; player 1 did not swap
  SPIEL_CHECK_EQ(46, state->LegalActions().size());

  state->ApplyAction(15);  // player 0: xb1
  // 45/46* legal actions; player 0 played on his end line
  SPIEL_CHECK_EQ(46, state->LegalActions().size());

  state->ApplyAction(11);  // player 1: ob5
  // 44*/45 legal actions
  SPIEL_CHECK_EQ(44, state->LegalActions().size());

  try {
    state->ApplyAction(11);   // player 0: xb5 NOT LEGAL!
  } catch (TestException e) {
    std::string expected = "Not a legal action: 11";
    SPIEL_CHECK_EQ(expected, std::string(e.what()));
  }

  state->ApplyAction(27);  // player 0: xd5
  // 43/44* legal actions
  SPIEL_CHECK_EQ(44, state->LegalActions().size());

  state->ApplyAction(17);  // player 1: oc7
  // 42*/43 legal actions
  SPIEL_CHECK_EQ(42, state->LegalActions().size());

  state->ApplyAction(42);  // player 0: xf6
  // 41/42* legal actions
  SPIEL_CHECK_EQ(42, state->LegalActions().size());

  state->ApplyAction(45);  // player 1: of3
  // 40*/41 legal actions
  SPIEL_CHECK_EQ(40, state->LegalActions().size());

  state->ApplyAction(48);  // player 0: xg8 wins
  SPIEL_CHECK_TRUE(state->IsTerminal());
  SPIEL_CHECK_EQ(1.0, state->PlayerReturn(0));
  SPIEL_CHECK_EQ(-1.0, state->PlayerReturn(1));
}

void DrawTest() {
  open_spiel::GameParameters params;
  params.insert({"board_size", open_spiel::GameParameter(5, false)});
  std::shared_ptr<const open_spiel::Game> game =
    open_spiel::LoadGame("twixt", params);
  auto state = game->NewInitialState();

  while (!state->IsTerminal()) {
    // this pattern will produce a draw on a 5x5 board
    state->ApplyAction(state->LegalActions().at(0));
    state->ApplyAction(state->LegalActions().at(1));
  }
  SPIEL_CHECK_EQ(0.0, state->PlayerReturn(0));
  SPIEL_CHECK_EQ(0.0, state->PlayerReturn(1));
}

}  // namespace
}  // namespace twixt
}  // namespace open_spiel

int main(int argc, char **argv) {
  open_spiel::twixt::BasicTwixTTests();
  open_spiel::SetErrorHandler(open_spiel::twixt::ErrorHandler);
  open_spiel::twixt::ParameterTest();
  open_spiel::twixt::SwapTest();
  open_spiel::twixt::LegalActionsTest();
  open_spiel::twixt::DrawTest();
}
