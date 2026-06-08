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
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "open_spiel/abseil-cpp/absl/flags/flag.h"
#include "open_spiel/abseil-cpp/absl/flags/parse.h"
#include "open_spiel/games/crossword/crossword.h"
#include "open_spiel/games/crossword/crossword_lib.h"
#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/utils/init.h"
#include "open_spiel/utils/status.h"

ABSL_FLAG(std::string, puzzles_root, "",
          "The root directory of the crossword puzzles. Default is empty, "
          "which will use the default embedded puzzle.");
ABSL_FLAG(std::string, word_list_file, "", "Word list file to use. "
          "Default is empty, which will allow all words.");


namespace open_spiel {
namespace crossword {
namespace {

constexpr int kSeed = 42397719;

void TestCrosswordGameDefaultPuzzleNoWordList() {
  std::shared_ptr<const Game> game = LoadGame("crossword");
  SPIEL_CHECK_TRUE(game != nullptr);
  const CrosswordGame* crossword_game =
      dynamic_cast<const CrosswordGame*>(game.get());
  SPIEL_CHECK_TRUE(crossword_game != nullptr);
  int num_puzzles = crossword_game->num_puzzles();
  int num_words = crossword_game->num_words();
  std::cout << "num_words: " << num_words << "\n";
  std::cout << "num_puzzles: " << num_puzzles << "\n";
  std::unique_ptr<State> state = game->NewInitialState();
  std::cout << "state: \n" << state->ToString() << "\n";

  Status status;
  status = state->ValidateActionStruct(CrosswordActionStruct{"A1", "A"});
  SPIEL_CHECK_FALSE(status.ok());  // Wrong length (too short)
  status = state->ValidateActionStruct(CrosswordActionStruct{"A1",
                                                             "GOAL"});
  SPIEL_CHECK_FALSE(status.ok());  // Too long
  status = state->ApplyActionStruct(CrosswordActionStruct{"A1", "GO"});
  SPIEL_CHECK_TRUE(status.ok());
  status = state->ValidateActionStruct(CrosswordActionStruct{"D1", "GAM"});
  SPIEL_CHECK_FALSE(status.ok());  // Too short
  status = state->ValidateActionStruct(CrosswordActionStruct{"D1", "GLUED"});
  SPIEL_CHECK_FALSE(status.ok());  // Too long
  status = state->ApplyActionStruct(CrosswordActionStruct{"D1", "GOAL"});
  SPIEL_CHECK_TRUE(status.ok());   // Fits (so, legal), but incorrect answer.
  status = state->ApplyActionStruct(CrosswordActionStruct{"D1", "GAME"});
  SPIEL_CHECK_TRUE(status.ok());   // Fits (so, legal), but incorrect answer.

  std::cout << "state: \n" << state->ToString() << "\n";
}

void TestCrosswordGameDefaultRandomSim() {
  std::string word_list_file = absl::GetFlag(FLAGS_word_list_file);
  if (word_list_file.empty()) {
    std::cerr << "No word_list_file specified. Skipping test: "
              << "TestCrosswordGameDefaultRandomSim" << std::endl;
    return;
  }

  std::shared_ptr<const Game> game = LoadGame(
      absl::StrCat("crossword(word_list_file=", word_list_file, ")"));
  SPIEL_CHECK_TRUE(game != nullptr);
  std::unique_ptr<State> state = game->NewInitialState();
  SimulateRandomGame(game, kSeed);
}

void TestCrosswordGameDefaultWinningSim() {
  std::shared_ptr<const Game> game = LoadGame("crossword");
  SPIEL_CHECK_TRUE(game != nullptr);
  std::unique_ptr<State> state = game->NewInitialState();
  SimulateWinningGame(game, kSeed);
}

void TestCrosswordGameWinningSimWithChance(int num_games) {
  std::string puzzles_root = absl::GetFlag(FLAGS_puzzles_root);
  std::string word_list_file = absl::GetFlag(FLAGS_word_list_file);
  if (puzzles_root.empty()) {
    std::cerr << "No puzzles_root specified. Skipping test: "
              << "TestCrosswordGameWinningSimWithChance" << std::endl;
    return;
  }
  std::string game_string = absl::StrCat("crossword(puzzles_root=",
      puzzles_root, ",word_list_file=", word_list_file, ")");
  std::mt19937 rng(kSeed);
  std::shared_ptr<const Game> game =
      LoadGame(game_string);
  SPIEL_CHECK_TRUE(game != nullptr);
  const CrosswordGame* crossword_game =
      dynamic_cast<const CrosswordGame*>(game.get());
  SPIEL_CHECK_TRUE(crossword_game != nullptr);
  int num_puzzles = crossword_game->num_puzzles();
  int num_words = crossword_game->num_words();
  std::cout << "num_words: " << num_words << "\n";
  std::cout << "num_puzzles: " << num_puzzles << "\n";
  for (int i = 0; i < num_games; ++i) {
    std::unique_ptr<State> state = game->NewInitialState();
    std::cout << "state: \n" << state->CurrentPlayer() << "\n";
    SPIEL_CHECK_TRUE(state->IsChanceNode());
    std::vector<std::pair<Action, double>> outcomes =
        state->ChanceOutcomes();
    Action action = SampleAction(outcomes, rng).first;
    state->ApplyAction(action);
    SPIEL_CHECK_FALSE(state->IsChanceNode());
    SimulateWinningGame(game, kSeed + i);
  }
}

}  // namespace
}  // namespace crossword
}  // namespace open_spiel

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  open_spiel::Init(argv[0], &argc, &argv, false);
  open_spiel::crossword::TestCrosswordGameDefaultPuzzleNoWordList();
  open_spiel::crossword::TestCrosswordGameDefaultRandomSim();
  open_spiel::crossword::TestCrosswordGameDefaultWinningSim();
  open_spiel::crossword::TestCrosswordGameWinningSimWithChance(10);
}
