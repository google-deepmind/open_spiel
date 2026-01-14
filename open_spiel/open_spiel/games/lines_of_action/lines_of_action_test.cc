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

#include <algorithm>
#include <array>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "open_spiel/abseil-cpp/absl/random/distributions.h"
#include "open_spiel/abseil-cpp/absl/strings/ascii.h"
#include "open_spiel/abseil-cpp/absl/strings/match.h"
#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/abseil-cpp/absl/strings/str_split.h"
#include "open_spiel/abseil-cpp/absl/types/optional.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_globals.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/tests/basic_tests.h"
#include "open_spiel/utils/file.h"
#include "open_spiel/utils/init.h"

constexpr int kSeed = 1329811;

namespace open_spiel {
namespace lines_of_action {
namespace {

constexpr const char* kFilePathPrefix =
    "third_party/open_spiel/games/lines_of_action/test_games/";

std::array<const char*, 16> kTestGameFileNames = {
    // Appendix A.
    "shortest_game.txt",
    // Appendix E
    "appendix_E_game1.txt",
    "appendix_E_game2.txt",
    "appendix_E_game3.txt",
    "appendix_E_game4.txt",
    "appendix_E_game5.txt",
    "appendix_E_game6.txt",
    "appendix_E_game7.txt",
    // Appendix F
    "appendix_F_game1_incomplete.txt",
    "appendix_F_game2.txt",
    "appendix_F_game3.txt",
    "appendix_F_game4.txt",
    "appendix_F_game5.txt",
    "appendix_F_game6.txt",
    "appendix_F_game7.txt",
    "appendix_F_game8.txt",
};

namespace testing = open_spiel::testing;

void OneRandomGameTest() {
  std::mt19937 rng(kSeed);
  std::shared_ptr<const Game> game = LoadGame("lines_of_action");
  std::unique_ptr<State> state = game->NewInitialState();

  while (!state->IsTerminal()) {
    std::cout << std::endl;
    std::cout << state << std::endl;
    std::cout << "Legal actions: " << std::endl;
    std::vector<Action> legal_actions = state->LegalActions();
    for (Action action : legal_actions) {
      std::cout << "  " << game->ActionToString(state->CurrentPlayer(), action)
                << std::endl;
    }

    int idx = absl::Uniform<int>(rng, 0, legal_actions.size());
    Action action = legal_actions[idx];
    std::cout << "Applying action: "
              << game->ActionToString(state->CurrentPlayer(), action)
              << std::endl;
    state->ApplyAction(action);
  }
  std::cout << "Game is over." << std::endl;
  std::cout << "Terminal state: " << std::endl;
  std::cout << state << std::endl;
  std::cout << "Returns: " << std::endl;
  std::vector<double> returns = state->Returns();
  for (double return_ : returns) {
    std::cout << " " << return_;
  }
  std::cout << std::endl;
}

Action ExtractAction(const State& state,
                     const std::vector<Action>& legal_actions,
                     const std::string& action_string) {
  Player current_player = state.CurrentPlayer();
  for (Action action : legal_actions) {
    if (state.ActionToString(current_player, action) == action_string) {
      return action;
    }
  }

  return kInvalidAction;
}

void PlayThroughTestGames() {
  std::shared_ptr<const Game> game = LoadGame("lines_of_action");

  for (const char* filename : kTestGameFileNames) {
    std::string full_filename = absl::StrCat(kFilePathPrefix, filename);
    absl::optional<std::string> file = FindFile(full_filename, 2);

    if (file != absl::nullopt) {
      std::cout << "Playing through test game " << filename << std::endl;
      std::string game_trace = file::ReadContentsFromFile(*file, "r");
      std::replace(game_trace.begin(), game_trace.end(), '\n', ' ');
      std::vector<std::string> string_history = absl::StrSplit(game_trace, ' ');

      std::unique_ptr<State> state = game->NewInitialState();

      for (std::string action_string : string_history) {
        absl::StripAsciiWhitespace(&action_string);

        if (action_string.empty()) {
          // Skip.
          continue;
        }

        // Skip the move numbers
        if (action_string[action_string.size() - 1] == '.') {
          continue;
        }

        SPIEL_CHECK_FALSE(state->IsTerminal());
        std::vector<Action> legal_actions = state->LegalActions();
        Action action = ExtractAction(*state, legal_actions, action_string);
        if (action == kInvalidAction) {
          std::cout << state->ToString() << std::endl;
          SpielFatalError(absl::StrCat("Invalid action: ", action_string));
        }

        state->ApplyAction(action);
      }

      std::cout << "Final state: " << std::endl;
      std::cout << state->ToString() << std::endl;

      if (!absl::StrContains(full_filename, "incomplete")) {
        SPIEL_CHECK_TRUE(state->IsTerminal());
      }
    }
  }
}

void BasicLinesOfActionTests() {
  testing::LoadGameTest("lines_of_action");
  testing::NoChanceOutcomesTest(*LoadGame("lines_of_action"));
  testing::RandomSimTest(*LoadGame("lines_of_action"), 100);
}

}  // namespace
}  // namespace lines_of_action
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::Init("", &argc, &argv, true);
  open_spiel::lines_of_action::OneRandomGameTest();
  open_spiel::lines_of_action::PlayThroughTestGames();
  open_spiel::lines_of_action::BasicLinesOfActionTests();
}
