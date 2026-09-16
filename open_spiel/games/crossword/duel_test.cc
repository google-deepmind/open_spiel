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
#include <vector>

#include "open_spiel/abseil-cpp/absl/flags/flag.h"
#include "open_spiel/abseil-cpp/absl/flags/parse.h"
#include "open_spiel/abseil-cpp/absl/random/distributions.h"
#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/games/crossword/duel.h"
#include "open_spiel/games/crossword/duel_default_board.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_globals.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/tests/action_struct_tests.h"
#include "open_spiel/utils/init.h"
#include "open_spiel/utils/status.h"

// Run with --word_list_file=$HOME/wordlist.txt to also exercise the game
// with a real dictionary via random simulation. Without it, the default
// 10-word built-in list is used.
ABSL_FLAG(std::string, word_list_file, "", "Word list file to use. "
          "Default is empty, which will use the small built-in word list.");

namespace open_spiel {
namespace crossword_duel {
namespace {

constexpr int kSeed = 42397719;

void TestCrosswordDuelGameDefaultWordList() {
  std::shared_ptr<const Game> game = LoadGame("crossword_duel");
  SPIEL_CHECK_TRUE(game != nullptr);
  const CrosswordDuelGame* duel_game =
      dynamic_cast<const CrosswordDuelGame*>(game.get());
  SPIEL_CHECK_TRUE(duel_game != nullptr);
  SPIEL_CHECK_EQ(duel_game->num_words(), 10);
  SPIEL_CHECK_EQ(game->NumPlayers(), 2);
}

void TestCreateCrosswordDuelGame() {
  std::shared_ptr<const Game> game =
      CreateCrosswordDuelGame(kDuelDefaultBoard, "");
  SPIEL_CHECK_TRUE(game != nullptr);
  const CrosswordDuelGame* duel_game =
      dynamic_cast<const CrosswordDuelGame*>(game.get());
  SPIEL_CHECK_TRUE(duel_game != nullptr);
  SPIEL_CHECK_EQ(duel_game->num_words(), 10);
  SPIEL_CHECK_EQ(game->NumPlayers(), 2);

  std::unique_ptr<State> state = game->NewInitialState();
  SPIEL_CHECK_TRUE(state != nullptr);
  SPIEL_CHECK_TRUE(state->IsChanceNode());
}

void TestCrosswordDuelGameDealAndFirstMove() {
  std::shared_ptr<const Game> game = LoadGame("crossword_duel");
  std::unique_ptr<State> state = game->NewInitialState();
  auto* duel_state = down_cast<CrosswordDuelState*>(state.get());
  std::mt19937 rng(kSeed);

  // Deal both racks (14 chance nodes).
  while (state->IsChanceNode()) {
    std::vector<std::pair<Action, double>> outcomes = state->ChanceOutcomes();
    Action action = SampleAction(outcomes, rng).first;
    state->ApplyAction(action);
  }
  SPIEL_CHECK_EQ(duel_state->rack(0).size(), kRackSize);
  SPIEL_CHECK_EQ(duel_state->rack(1).size(), kRackSize);
  SPIEL_CHECK_EQ(state->CurrentPlayer(), 0);

  // A word not covering the start cell should be rejected as the first move.
  Status status = state->ValidateActionStruct(
      CrosswordDuelActionStruct(0, 0, "across", "AB"));
  SPIEL_CHECK_FALSE(status.ok());

  // A pass is always legal (direction/word are irrelevant when pass=true).
  CrosswordDuelActionStruct pass_action(0, 0, "across", "X");
  pass_action.pass = true;
  status = state->ValidateActionStruct(pass_action);
  SPIEL_CHECK_TRUE(status.ok());

  std::cout << "Initial state:\n" << state->ToString() << "\n";
}

// Manually drives a short game using CAT/DOG/GOAT from the default word list,
// forcing both racks so the moves are guaranteed to be legal, in order to
// exercise scoring, board mutation, and serialization round-tripping without
// depending on what the random deal produced.
void TestCrosswordDuelGameManualPlayAndSerialization() {
  std::shared_ptr<const Game> game = LoadGame("crossword_duel");
  std::unique_ptr<State> state = game->NewInitialState();
  std::mt19937 rng(kSeed);

  while (state->IsChanceNode()) {
    std::vector<std::pair<Action, double>> outcomes = state->ChanceOutcomes();
    Action action = SampleAction(outcomes, rng).first;
    state->ApplyAction(action);

    std::string serialized_state = state->Serialize();
    std::unique_ptr<State> deserialized_state =
        game->DeserializeState(serialized_state);
    SPIEL_CHECK_EQ(state->Serialize(), deserialized_state->Serialize());
  }

  // Whether any of the (only 10) default words can be spelled from whatever
  // 7 random letters player 0 happened to be dealt is luck of the draw --
  // the first move must be built entirely from the rack. Rather than search
  // for a fit ourselves (which may legitimately not exist), reuse the
  // game's own ActionStructSampler, which already handles the
  // found-nothing-so-pass case the same way real play would.
  std::unique_ptr<ActionStructSampler> sampler =
      state->GetActionStructSampler(kSeed);
  std::unique_ptr<ActionStruct> action = sampler->SampleActionStruct();
  SPIEL_CHECK_TRUE(action != nullptr);
  Status status = state->ApplyActionStruct(*action);
  SPIEL_CHECK_TRUE(status.ok());
  const auto& duel_action =
      down_cast<const CrosswordDuelActionStruct&>(*action);
  if (!duel_action.pass) {
    auto* duel_state_after = down_cast<CrosswordDuelState*>(state.get());
    SPIEL_CHECK_GT(duel_state_after->GetScore(0), 0);
  }

  // Drain any refill chance node triggered by the move.
  while (state->IsChanceNode()) {
    std::vector<std::pair<Action, double>> outcomes = state->ChanceOutcomes();
    Action action = SampleAction(outcomes, rng).first;
    state->ApplyAction(action);
  }
  SPIEL_CHECK_EQ(state->CurrentPlayer(), 1);

  std::string serialized_state = state->Serialize();
  std::unique_ptr<State> deserialized_state =
      game->DeserializeState(serialized_state);
  SPIEL_CHECK_EQ(state->Serialize(), deserialized_state->Serialize());
  SPIEL_CHECK_EQ(state->ToString(), deserialized_state->ToString());
  SPIEL_CHECK_EQ(state->Returns(), deserialized_state->Returns());

  std::cout << "State after first move:\n" << state->ToString() << "\n";
}

void TestCrosswordDuelGameRandomSim(const std::string& word_list_file) {
  std::string game_string = "crossword_duel";
  if (!word_list_file.empty()) {
    game_string =
        absl::StrCat("crossword_duel(word_list_file=", word_list_file, ")");
  }
  std::shared_ptr<const Game> game = LoadGame(game_string);
  SPIEL_CHECK_TRUE(game != nullptr);
  const CrosswordDuelGame* duel_game =
      dynamic_cast<const CrosswordDuelGame*>(game.get());
  SPIEL_CHECK_TRUE(duel_game != nullptr);
  std::cout << "num_words: " << duel_game->num_words() << "\n";
  testing::SimulateRandomGame(game, kSeed);
}

// Same as TestCrosswordDuelGameRandomSim, but builds the game via
// CreateCrosswordDuelGame(kDuelDefaultBoard, word_list_file) instead of
// LoadGame, to exercise that entry point as well.
void TestCreateCrosswordDuelGameRandomSim(const std::string& word_list_file) {
  std::shared_ptr<const Game> game =
      CreateCrosswordDuelGame(kDuelDefaultBoard, word_list_file);
  SPIEL_CHECK_TRUE(game != nullptr);
  const CrosswordDuelGame* duel_game =
      dynamic_cast<const CrosswordDuelGame*>(game.get());
  SPIEL_CHECK_TRUE(duel_game != nullptr);
  std::cout << "num_words: " << duel_game->num_words() << "\n";
  testing::SimulateRandomGame(game, kSeed);
}

}  // namespace
}  // namespace crossword_duel
}  // namespace open_spiel

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  open_spiel::Init(argv[0], &argc, &argv, false);
  open_spiel::crossword_duel::TestCrosswordDuelGameDefaultWordList();
  open_spiel::crossword_duel::TestCreateCrosswordDuelGame();
  open_spiel::crossword_duel::TestCrosswordDuelGameDealAndFirstMove();
  open_spiel::crossword_duel::TestCrosswordDuelGameManualPlayAndSerialization();
  open_spiel::crossword_duel::TestCrosswordDuelGameRandomSim("");

  std::string word_list_file = absl::GetFlag(FLAGS_word_list_file);
  if (!word_list_file.empty()) {
    open_spiel::crossword_duel::TestCrosswordDuelGameRandomSim(word_list_file);
    open_spiel::crossword_duel::TestCreateCrosswordDuelGameRandomSim(
        word_list_file);
  } else {
    std::cerr << "No word_list_file specified. Skipping the random "
              << "simulations with a real dictionary. Pass "
              << "--word_list_file=$HOME/wordlist.txt to also run them."
              << std::endl;
  }
}
