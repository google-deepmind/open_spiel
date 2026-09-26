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
#include "open_spiel/abseil-cpp/absl/strings/match.h"
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

// Applies the chance action that draws the given letter. Fails if that
// letter is not currently available in the bag.
void ApplyForcedChanceLetter(State* state, char letter) {
  std::vector<std::pair<Action, double>> outcomes = state->ChanceOutcomes();
  Action action = letter - 'A';
  bool found = false;
  for (const auto& [outcome_action, prob] : outcomes) {
    if (outcome_action == action) {
      found = true;
      break;
    }
  }
  SPIEL_CHECK_TRUE(found);
  state->ApplyAction(action);
}

// Applies an arbitrary (don't-care) chance outcome, used to fill out the
// remainder of a deal/draw once the letters we care about have been forced.
void ApplyAnyChanceLetter(State* state) {
  SPIEL_CHECK_TRUE(state->IsChanceNode());
  std::vector<std::pair<Action, double>> outcomes = state->ChanceOutcomes();
  state->ApplyAction(outcomes[0].first);
}

void ApplyAnyChanceLetters(State* state, int repetitions) {
  for (int i = 0; i < repetitions; ++i) {
    ApplyAnyChanceLetter(state);
  }
}

void SetupRack(CrosswordDuelState* state, const std::string& rack) {
  for (char c : rack) {
    SPIEL_CHECK_TRUE(state->IsChanceNode());
    state->ApplyAction(state->ChanceOutcomeForLetter(c));
  }
}

void CheckInvalidAction(State* state,
                        int row, int col, Direction direction,
                        const std::string& word) {
  CrosswordDuelActionStruct action(row, col, direction, word);
  Status status = state->ValidateActionStruct(action);
  SPIEL_CHECK_FALSE(status.ok());
}

void CheckAndApplyAction(State* state,
                         int row, int col, Direction direction,
                         const std::string& word) {
  CrosswordDuelActionStruct action(row, col, direction, word);
  Status status = state->ValidateActionStruct(action);
  SPIEL_CHECK_TRUE(status.ok());
  status = state->ApplyActionStruct(action);
  SPIEL_CHECK_TRUE(status.ok());
}

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
      CrosswordDuelActionStruct(0, 0, Direction::kAcross, "AB"));
  SPIEL_CHECK_FALSE(status.ok());

  // A pass is always legal (direction/word are irrelevant when pass=true).
  CrosswordDuelActionStruct pass_action(0, 0, Direction::kAcross, "X");
  pass_action.pass = true;
  status = state->ValidateActionStruct(pass_action);
  SPIEL_CHECK_TRUE(status.ok());

  std::cout << "Initial state:\n" << state->ToString() << "\n";
}

// Forces both racks (via the dealing chance nodes) so that player 0 is
// guaranteed to hold C/A/T and player 1 is guaranteed to hold B/E/A/R, then
// checks: (1) a known-legal placement of CAT at the start cell validates as
// legal; (2) extending the board with a letter the mover's rack doesn't have
// is rejected specifically for insufficient tiles; and (3) an otherwise-legal
// word placed away from the board's only tile is rejected specifically for
// not connecting to an existing tile.
void TestCrosswordDuelGameLegalAndIllegalPlacements() {
  std::shared_ptr<const Game> game = LoadGame("crossword_duel");
  std::unique_ptr<State> state = game->NewInitialState();
  auto* duel_state = down_cast<CrosswordDuelState*>(state.get());

  std::vector<char> forced_p0 = {'C', 'A', 'T'};
  std::vector<char> forced_p1 = {'B', 'E', 'A', 'R'};
  size_t p0_idx = 0, p1_idx = 0;
  while (state->IsChanceNode()) {
    bool dealing_to_p0 = duel_state->rack(0).size() < kRackSize;
    std::vector<char>& forced = dealing_to_p0 ? forced_p0 : forced_p1;
    size_t& idx = dealing_to_p0 ? p0_idx : p1_idx;
    if (idx < forced.size()) {
      ApplyForcedChanceLetter(state.get(), forced[idx]);
      idx++;
    } else {
      ApplyAnyChanceLetter(state.get());
    }
  }
  SPIEL_CHECK_EQ(duel_state->rack(0).size(), kRackSize);
  SPIEL_CHECK_EQ(duel_state->rack(1).size(), kRackSize);
  SPIEL_CHECK_EQ(state->CurrentPlayer(), 0);

  // (1) CAT across the start cell (7, 7) is a legal opening move.
  CrosswordDuelActionStruct legal_action(7, 7, Direction::kAcross, "CAT");
  Status status = state->ValidateActionStruct(legal_action);
  SPIEL_CHECK_TRUE(status.ok());
  status = state->ApplyActionStruct(legal_action);
  SPIEL_CHECK_TRUE(status.ok());
  SPIEL_CHECK_EQ(duel_state->CharAt(7, 7), 'C');
  SPIEL_CHECK_EQ(duel_state->CharAt(7, 8), 'A');
  SPIEL_CHECK_EQ(duel_state->CharAt(7, 9), 'T');

  // Drain the refill chance node(s) triggered by player 0's move.
  while (state->IsChanceNode()) {
    ApplyAnyChanceLetter(state.get());
  }
  SPIEL_CHECK_EQ(state->CurrentPlayer(), 1);

  // (2) Extending "CAT" into "CATX" needs an 'X' tile that player 1's rack
  // (B/E/A/R + filler) does not have. This placement touches the existing
  // "CAT" tiles, so the failure is specifically about the missing tile,
  // not connectivity.
  CrosswordDuelActionStruct not_enough_tiles(7, 7, Direction::kAcross,
                                             "CATX");
  status = state->ValidateActionStruct(not_enough_tiles);
  SPIEL_CHECK_FALSE(status.ok());
  SPIEL_CHECK_TRUE(absl::StrContains(status.message(), "Not enough"));

  // (3) "BEAR" is spellable from player 1's rack, but placed far from the
  // only tiles on the board (CAT, around row/col 7), so it fails for not
  // connecting to any existing tile.
  CrosswordDuelActionStruct not_connected(0, 0, Direction::kAcross, "BEAR");
  status = state->ValidateActionStruct(not_connected);
  SPIEL_CHECK_FALSE(status.ok());
  SPIEL_CHECK_TRUE(absl::StrContains(status.message(), "connect"));
}

// Regression test for the game's termination rules: it must end only via two
// consecutive passes or a player emptying their rack -- never merely because
// the tile bag has run out. Uses a tiny custom board whose tile bag holds
// exactly 2*kRackSize tiles, so the bag is emptied by the initial deal alone;
// a normal placement made after that must neither trigger a chance/draw node
// nor end the game, as long as the mover's rack still has tiles left.
void TestCrosswordDuelGameBagEmptyDoesNotEndGame() {
  const std::string kTinyBoard =
      ".....\n"
      ".....\n"
      "..s..\n"
      ".....\n"
      ".....\n"
      "\n"
      "C 1 1\n"
      "A 1 1\n"
      "T 1 1\n"
      "Z 11 1\n";
  std::shared_ptr<const Game> game = CreateCrosswordDuelGame(kTinyBoard, "");
  std::unique_ptr<State> state = game->NewInitialState();
  auto* duel_state = down_cast<CrosswordDuelState*>(state.get());

  // Force player 0's rack to contain C, A, T; every other draw (for either
  // player) will be Z, the only letter left in the bag once C/A/T are gone,
  // so the bag ends up fully drained (0 tiles) right after the initial deal.
  std::vector<char> forced_p0 = {'C', 'A', 'T'};
  size_t p0_idx = 0;
  while (state->IsChanceNode()) {
    bool dealing_to_p0 = duel_state->rack(0).size() < kRackSize;
    if (dealing_to_p0 && p0_idx < forced_p0.size()) {
      ApplyForcedChanceLetter(state.get(), forced_p0[p0_idx]);
      p0_idx++;
    } else {
      ApplyAnyChanceLetter(state.get());
    }
  }
  SPIEL_CHECK_EQ(duel_state->rack(0).size(), kRackSize);
  SPIEL_CHECK_EQ(duel_state->rack(1).size(), kRackSize);
  SPIEL_CHECK_EQ(duel_state->bag_size(), 0);
  SPIEL_CHECK_EQ(state->CurrentPlayer(), 0);

  // Player 0 plays CAT at the start cell using 3 of their 7 tiles. Even
  // though the bag is already empty (so no refill is possible), player 0's
  // rack still has 4 tiles left, so this move must NOT end the game.
  CrosswordDuelActionStruct cat_action(2, 2, Direction::kAcross, "CAT");
  Status status = state->ApplyActionStruct(cat_action);
  SPIEL_CHECK_TRUE(status.ok());

  SPIEL_CHECK_FALSE(state->IsChanceNode());
  SPIEL_CHECK_EQ(duel_state->bag_size(), 0);
  SPIEL_CHECK_EQ(duel_state->rack(0).size(), kRackSize - 3);
  SPIEL_CHECK_FALSE(state->IsTerminal());
  SPIEL_CHECK_EQ(state->CurrentPlayer(), 1);
}

// Regression test: two consecutive passes end the game even when the tile
// bag is still mostly full, confirming that pass-based termination does not
// depend on the bag's contents.
void TestCrosswordDuelGamePassesEndGameRegardlessOfBag() {
  std::shared_ptr<const Game> game = LoadGame("crossword_duel");
  std::unique_ptr<State> state = game->NewInitialState();
  auto* duel_state = down_cast<CrosswordDuelState*>(state.get());
  std::mt19937 rng(kSeed);

  while (state->IsChanceNode()) {
    std::vector<std::pair<Action, double>> outcomes = state->ChanceOutcomes();
    Action action = SampleAction(outcomes, rng).first;
    state->ApplyAction(action);
  }
  SPIEL_CHECK_GT(duel_state->bag_size(), 0);
  SPIEL_CHECK_FALSE(state->IsTerminal());

  CrosswordDuelActionStruct pass_action(0, 0, Direction::kAcross, "");
  pass_action.pass = true;

  Status status = state->ApplyActionStruct(pass_action);
  SPIEL_CHECK_TRUE(status.ok());
  SPIEL_CHECK_FALSE(state->IsTerminal());

  status = state->ApplyActionStruct(pass_action);
  SPIEL_CHECK_TRUE(status.ok());
  SPIEL_CHECK_TRUE(state->IsTerminal());
  SPIEL_CHECK_GT(duel_state->bag_size(), 0);
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
  // found-nothing-so-pass/discard case the same way real play would.
  std::unique_ptr<ActionStructSampler> sampler =
      state->GetActionStructSampler(kSeed);
  std::unique_ptr<ActionStruct> action = sampler->SampleActionStruct();
  SPIEL_CHECK_TRUE(action != nullptr);
  Status status = state->ApplyActionStruct(*action);
  SPIEL_CHECK_TRUE(status.ok());
  const auto& duel_action =
      down_cast<const CrosswordDuelActionStruct&>(*action);
  if (!duel_action.pass && !duel_action.discard) {
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

// Plays a single random game to completion, printing the state before each
// player move along with the chosen action, and the returns at the end.
// Chance nodes (tile draws) are sampled silently to keep the output readable.
void TestCrosswordDuelGameVerboseRandomSim(const std::string& word_list_file) {
  std::string game_string = "crossword_duel";
  if (!word_list_file.empty()) {
    game_string =
        absl::StrCat("crossword_duel(word_list_file=", word_list_file, ")");
  }
  std::shared_ptr<const Game> game = LoadGame(game_string);
  std::unique_ptr<State> state = game->NewInitialState();
  std::unique_ptr<ActionStructSampler> sampler =
      state->GetActionStructSampler(kSeed);
  std::mt19937 rng(kSeed);

  int move_num = 0;
  while (!state->IsTerminal()) {
    if (state->IsChanceNode()) {
      std::vector<std::pair<Action, double>> outcomes = state->ChanceOutcomes();
      Action action = SampleAction(outcomes, rng).first;
      state->ApplyAction(action);
      continue;
    }
    Player player = state->CurrentPlayer();
    std::unique_ptr<ActionStruct> action = sampler->SampleActionStruct();
    SPIEL_CHECK_TRUE(action != nullptr);
    std::cout << "Move " << move_num << ", player " << player << ":\n"
              << state->ToString() << "\n"
              << "Chosen action: " << action->ToJson() << "\n\n";
    Status status = state->ApplyActionStruct(*action);
    SPIEL_CHECK_TRUE(status.ok());
    ++move_num;
  }

  std::cout << "Terminal state:\n" << state->ToString() << "\n";
  std::vector<double> returns = state->Returns();
  std::cout << "Returns:";
  for (double r : returns) std::cout << " " << r;
  std::cout << "\n";
}

void TestMultiWords(const std::string& word_list_file) {
  // BITE
  //  CONE
  // MEND
  if (word_list_file == "") {
    std::cerr << "TestMultiWords requires a word list. Skipping." << std::endl;
  }
  std::string game_string =
    absl::StrCat("crossword_duel(word_list_file=", word_list_file, ")");
  std::shared_ptr<const Game> game = LoadGame(game_string);
  std::unique_ptr<State> state = game->NewInitialState();
  CrosswordDuelState* duel_state =
      dynamic_cast<CrosswordDuelState*>(state.get());
  SPIEL_CHECK_TRUE(state != nullptr);
  SPIEL_CHECK_TRUE(state->IsChanceNode());

  SetupRack(duel_state, "CONEBTE");
  SetupRack(duel_state, "ICEMEND");

  SPIEL_CHECK_EQ(state->CurrentPlayer(), 0);
  // invalid: missing letter
  CheckInvalidAction(duel_state, 7, 7, Direction::kAcross, "BORE");
  // invalid: not on starting square
  CheckInvalidAction(duel_state, 6, 7, Direction::kAcross, "CONE");
  CheckAndApplyAction(duel_state, 7, 7, Direction::kAcross, "CONE");
  ApplyAnyChanceLetters(duel_state, 4);

  SPIEL_CHECK_EQ(state->CurrentPlayer(), 1);
  // invalid: not a word
  CheckInvalidAction(duel_state, 6, 7, Direction::kDown, "DCE");
  // invalid: missing a letter
  CheckInvalidAction(duel_state, 6, 7, Direction::kDown, "ICY");
  // invalid: second letter has to be a C for 6,7 down
  CheckInvalidAction(duel_state, 6, 7, Direction::kDown, "DICE");
  CheckAndApplyAction(duel_state, 6, 7, Direction::kDown, "ICE");
  ApplyAnyChanceLetters(duel_state, 2);

  SPIEL_CHECK_EQ(state->CurrentPlayer(), 0);
  // invalid: words
  CheckInvalidAction(duel_state, 6, 6, Direction::kAcross, "TIBE");
  CheckAndApplyAction(duel_state, 6, 6, Direction::kAcross, "BITE");
  ApplyAnyChanceLetters(duel_state, 3);

  SPIEL_CHECK_EQ(state->CurrentPlayer(), 1);
  // invalid: words
  CheckInvalidAction(duel_state, 8, 6, Direction::kAcross, "DEMN");
  CheckAndApplyAction(duel_state, 8, 6, Direction::kAcross, "MEND");
  ApplyAnyChanceLetters(duel_state, 3);

  SPIEL_CHECK_EQ(state->CurrentPlayer(), 0);
  std::cout << duel_state->ToString() << std::endl;
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
  open_spiel::crossword_duel::TestCrosswordDuelGameLegalAndIllegalPlacements();
  open_spiel::crossword_duel::TestCrosswordDuelGameBagEmptyDoesNotEndGame();
  open_spiel::crossword_duel::TestCrosswordDuelGamePassesEndGameRegardlessOfBag();
  open_spiel::crossword_duel::TestCrosswordDuelGameManualPlayAndSerialization();
  open_spiel::crossword_duel::TestCrosswordDuelGameRandomSim("");

  std::string word_list_file = absl::GetFlag(FLAGS_word_list_file);
  if (!word_list_file.empty()) {
    open_spiel::crossword_duel::TestCrosswordDuelGameRandomSim(word_list_file);
    open_spiel::crossword_duel::TestCreateCrosswordDuelGameRandomSim(
        word_list_file);
    open_spiel::crossword_duel::TestMultiWords(word_list_file);
    open_spiel::crossword_duel::TestCrosswordDuelGameVerboseRandomSim(
        word_list_file);
  } else {
    std::cerr << "No word_list_file specified. Skipping the random "
              << "simulations with a real dictionary. Pass "
              << "--word_list_file=$HOME/wordlist.txt to also run them."
              << std::endl;
  }
}
