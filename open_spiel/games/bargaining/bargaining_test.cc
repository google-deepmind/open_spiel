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

#include "open_spiel/games/bargaining/bargaining.h"

#include <array>
#include <iostream>
#include <vector>

#include "open_spiel/abseil-cpp/absl/flags/flag.h"
#include "open_spiel/abseil-cpp/absl/flags/parse.h"
#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/spiel.h"
#include "open_spiel/tests/basic_tests.h"
#include "open_spiel/utils/init.h"

// This is set to false by default because it complicates tests on github CI.
ABSL_FLAG(bool, enable_instances_file_test, false,
          "Whether to test loading of an instances file.");

namespace open_spiel {
namespace bargaining {
namespace {

constexpr const char* kInstancesFilename =
    "open_spiel/games/bargaining/bargaining_instances1000.txt";
constexpr int kFileNumInstances = 1000;

namespace testing = open_spiel::testing;

void BasicBargainingTests() {
  testing::LoadGameTest("bargaining");
  testing::RandomSimTest(*LoadGame("bargaining"), 10);
  testing::RandomSimTest(*LoadGame("bargaining(prob_end=0.1)"), 10);
  testing::RandomSimTest(*LoadGame("bargaining(discount=0.9)"), 10);
  testing::RandomSimTest(*LoadGame("bargaining(max_turns=200)"), 10);
}

void BargainingMaxTurnsTest() {
  std::shared_ptr<const Game> game = LoadGame("bargaining(max_turns=200)");
  std::unique_ptr<State> state = game->NewInitialState();
  int num_turns = 200;
  while (num_turns > 0) {
    if (state->IsChanceNode()) {
      ActionsAndProbs chance_outcomes = state->ChanceOutcomes();
      state->ApplyAction(chance_outcomes[0].first);
    } else {
      SPIEL_CHECK_TRUE(!state->IsTerminal());
      num_turns--;
      std::vector<Action> legal_actions = state->LegalActions();
      state->ApplyAction(legal_actions[0]);
    }
  }
  SPIEL_CHECK_TRUE(state->IsTerminal());
}

void BargainingDiscountTest() {
  std::shared_ptr<const Game> game = LoadGame("bargaining(discount=0.9)");
  std::unique_ptr<State> state = game->NewInitialState();
  BargainingState* bargaining_state =
      static_cast<BargainingState*>(state.get());
  ActionsAndProbs chance_outcomes = state->ChanceOutcomes();
  state->ApplyAction(chance_outcomes[0].first);
  std::vector<Action> legal_actions = state->LegalActions();
  state->ApplyAction(legal_actions[0]);
  state->ApplyAction(legal_actions[0]);
  state->ApplyAction(legal_actions[0]);
  state->ApplyAction(legal_actions[0]);
  state->ApplyAction(bargaining_state->AgreeAction());
  // P0 offers [0,0,0] then P1, then P0, then P1, then P0 agrees.
  // P0 would get 10, but it's discounted by 0.9 three times
  SPIEL_CHECK_FLOAT_EQ(state->PlayerReturn(0), 0.9 * 0.9 * 0.9 * 10);
  SPIEL_CHECK_FLOAT_EQ(state->PlayerReturn(1), 0.0);
}

void BargainingProbEndContinueTest() {
  std::shared_ptr<const Game> game = LoadGame("bargaining(prob_end=0.1)");
  std::unique_ptr<State> state = game->NewInitialState();
  state->ApplyAction(state->ChanceOutcomes()[0].first);
  std::vector<Action> legal_actions = state->LegalActions();
  state->ApplyAction(legal_actions[0]);
  state->ApplyAction(legal_actions[0]);
  for (int i = 0; i < (bargaining::kDefaultMaxTurns - 2); ++i) {
    SPIEL_CHECK_TRUE(state->IsChanceNode());
    state->ApplyAction(state->ChanceOutcomes()[0].first);
    SPIEL_CHECK_TRUE(!state->IsChanceNode());
    legal_actions = state->LegalActions();
    state->ApplyAction(legal_actions[0]);
  }
  SPIEL_CHECK_TRUE(state->IsTerminal());
}

void BargainingProbEndEndTest() {
  std::shared_ptr<const Game> game = LoadGame("bargaining(prob_end=0.1)");
  std::unique_ptr<State> state = game->NewInitialState();
  state->ApplyAction(state->ChanceOutcomes()[0].first);
  std::vector<Action> legal_actions = state->LegalActions();
  state->ApplyAction(legal_actions[0]);
  state->ApplyAction(legal_actions[0]);
  for (int i = 0; i < (bargaining::kDefaultMaxTurns - 4); ++i) {
    SPIEL_CHECK_TRUE(state->IsChanceNode());
    state->ApplyAction(state->ChanceOutcomes()[0].first);
    SPIEL_CHECK_TRUE(!state->IsChanceNode());
    legal_actions = state->LegalActions();
    state->ApplyAction(legal_actions[0]);
  }
  SPIEL_CHECK_TRUE(state->IsChanceNode());
  SPIEL_CHECK_TRUE(!state->IsTerminal());
  state->ApplyAction(state->ChanceOutcomes()[1].first);
  SPIEL_CHECK_TRUE(state->IsTerminal());
  SPIEL_CHECK_FLOAT_EQ(state->PlayerReturn(0), 0.0);
  SPIEL_CHECK_FLOAT_EQ(state->PlayerReturn(1), 0.0);
}

void BasicBargainingFromInstancesFileTests() {
  // Game creation and legal actions are fairly heavy, so only run 1 sim.
  std::shared_ptr<const Game> game = LoadGame(
      absl::StrCat("bargaining(instances_file=", kInstancesFilename, ")"));

  const auto* bargaining_game = static_cast<const BargainingGame*>(game.get());
  SPIEL_CHECK_EQ(bargaining_game->AllInstances().size(), kFileNumInstances);

  testing::RandomSimTest(*game, 100);
}

void BasicBargainingFromCCInstancesTests() {
  std::shared_ptr<const Game> game = LoadGame("bargaining");

  const auto* bargaining_game = static_cast<const BargainingGame*>(game.get());
  SPIEL_CHECK_EQ(bargaining_game->AllInstances().size(), kDefaultNumInstances);
}

void BasicBargainingInstanceMapTests() {
  std::shared_ptr<const Game> game = LoadGame("bargaining");
  const auto* bargaining_game = static_cast<const BargainingGame*>(game.get());
  for (int i = 0; i < bargaining_game->AllInstances().size(); ++i) {
    const Instance& instance = bargaining_game->GetInstance(i);
    SPIEL_CHECK_EQ(bargaining_game->GetInstanceIndex(instance), i);
  }
}

void BasicBargainingOfferMapTests() {
  std::shared_ptr<const Game> game = LoadGame("bargaining");
  const auto* bargaining_game = static_cast<const BargainingGame*>(game.get());
  for (int i = 0; i < bargaining_game->AllOffers().size(); ++i) {
    const Offer& offer = bargaining_game->GetOffer(i);
    SPIEL_CHECK_EQ(bargaining_game->GetOfferIndex(offer), i);
  }
}

void BasicBargainingOpponentValuesTests() {
  std::shared_ptr<const Game> game = LoadGame("bargaining");
  const auto* bargaining_game = static_cast<const BargainingGame*>(game.get());
  std::vector<std::vector<int>> expected_values = {
    {4, 0, 2}, {7, 0, 1}, {1, 3, 1}
  };
  std::vector<int> player_values = {1, 2, 3};
  std::vector<int> opponent_values = {8, 1, 0};
  std::vector<std::vector<int>> actual_values =
      bargaining_game->GetPossibleOpponentValues(
          0, player_values, opponent_values);
  SPIEL_CHECK_EQ(actual_values, expected_values);
}

}  // namespace
}  // namespace bargaining
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::Init("", &argc, &argv, false);
  absl::ParseCommandLine(argc, argv);
  open_spiel::bargaining::BasicBargainingTests();
  if (absl::GetFlag(FLAGS_enable_instances_file_test)) {
    open_spiel::bargaining::BasicBargainingFromInstancesFileTests();
  }
  open_spiel::bargaining::BargainingMaxTurnsTest();
  open_spiel::bargaining::BargainingDiscountTest();
  open_spiel::bargaining::BargainingProbEndContinueTest();
  open_spiel::bargaining::BargainingProbEndEndTest();
  open_spiel::bargaining::BasicBargainingFromCCInstancesTests();
  open_spiel::bargaining::BasicBargainingInstanceMapTests();
  open_spiel::bargaining::BasicBargainingOfferMapTests();
  open_spiel::bargaining::BasicBargainingOpponentValuesTests();
}
