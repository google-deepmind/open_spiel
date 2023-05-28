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

#include <iostream>
#include <string>

#include "open_spiel/spiel.h"
#include "open_spiel/tests/basic_tests.h"

#include "abseil-cpp/absl/algorithm/container.h"
#include "abseil-cpp/absl/strings/ascii.h"
#include "abseil-cpp/absl/strings/numbers.h"

namespace open_spiel {
namespace nine_mens_morris {
namespace {

namespace testing = open_spiel::testing;

void BasicNineMensMorrisTests() {
  testing::LoadGameTest("nine_mens_morris");
  testing::NoChanceOutcomesTest(*LoadGame("nine_mens_morris"));
  testing::RandomSimTest(*LoadGame("nine_mens_morris"), 100);
}

void ManualPlaythroughTest() {
  std::shared_ptr<const Game> game = LoadGame("nine_mens_morris");
  std::unique_ptr<State> state = game->NewInitialState();
  std::cout << state << std::endl;
}

void InteractiveTest(bool print_legals) {
  std::shared_ptr<const Game> game = LoadGame("nine_mens_morris");
  std::unique_ptr<State> state = game->NewInitialState();
  
  while (!state->IsTerminal()) {
    std::cout << state->ToString() << std::endl << std::endl;
    std::vector<Action> legal_actions = state->LegalActions();
    Player player = state->CurrentPlayer();
    if (print_legals) {
      std::cout << "Legal actions: " << std::endl;
      for (Action action : legal_actions) {
        std::cout << "  " << action << ": "
                  << state->ActionToString(player, action)
                  << std::endl;
      }
    }
    std::cout << "> ";
    std::string line = "";
    std::getline(std::cin, line);
    absl::StripAsciiWhitespace(&line);
    if (line == "") {
      // TODO: print help screen
      std::cout << "Legal actions: " << std::endl;
      for (Action action : legal_actions) {
        std::cout << "  " << action << ": "
                  << state->ActionToString(player, action)
                  << std::endl;
      }
    } else {
      Action action;
      bool valid = absl::SimpleAtoi(line, &action);
      if (valid) {
        auto iter = absl::c_find(legal_actions, action);
        SPIEL_CHECK_TRUE(iter != legal_actions.end());
        state->ApplyAction(action);
      }
    }
  }

  std::cout << "Terminal state:" << std::endl << std::endl
	    << state->ToString() << std::endl;
  std::cout << "Returns: ";
  std::vector<double> returns = state->Returns();
  for (Player p = 0; p < game->NumPlayers(); ++p) {
    std::cout << returns[p] << " ";
  }
  std::cout << std::endl;
}

}  // namespace
}  // namespace nine_mens_morris
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::nine_mens_morris::BasicNineMensMorrisTests();
  //open_spiel::nine_mens_morris::ManualPlaythroughTest();
  //open_spiel::nine_mens_morris::InteractiveTest(false);
}
