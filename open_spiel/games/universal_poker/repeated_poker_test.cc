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

#include "open_spiel/games/universal_poker/repeated_poker.h"

#include <memory>
#include <string>
#include <vector>

#include "open_spiel/abseil-cpp/absl/flags/parse.h"
#include "open_spiel/abseil-cpp/absl/strings/match.h"
#include "open_spiel/canonical_game_strings.h"
#include "open_spiel/game_parameters.h"
#include "open_spiel/observer.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/tests/basic_tests.h"
#include "open_spiel/utils/init.h"

namespace open_spiel {
namespace universal_poker {
namespace repeated_poker {
namespace {

namespace testing = open_spiel::testing;

void BasicRepeatedPokerTest() {
  std::shared_ptr<const Game> game =
      LoadGame("repeated_poker",
               {{"max_num_hands", GameParameter(100)},
                {"reset_stacks", GameParameter(true)},
                {"rotate_dealer", GameParameter(true)},
                {"universal_poker_game_string",
                 GameParameterFromString(open_spiel::HunlGameString(
                     "fullgame"))}});
  std::unique_ptr<State> state = game->NewInitialState();
  testing::RandomSimTest(*game, 5);
}

void BlindScheduleTest() {
  std::shared_ptr<const Game> game =
      LoadGame("repeated_poker",
               {{"max_num_hands", GameParameter(100)},
                {"reset_stacks", GameParameter(false)},
                {"rotate_dealer", GameParameter(true)},
                {"blind_schedule", GameParameter(
                    "10:100/200;20:200/400;50:400/800;")},
                {"universal_poker_game_string",
                 GameParameterFromString(open_spiel::HunlGameString(
                     "fullgame"))}});
  std::unique_ptr<State> state_ = game->NewInitialState();
  RepeatedPokerState* state = dynamic_cast<RepeatedPokerState*>(state_.get());
  SPIEL_CHECK_TRUE(absl::StrContains(state->ToString(), "P0: 200"));
  SPIEL_CHECK_TRUE(absl::StrContains(state->ToString(), "P1: 100"));
  SPIEL_CHECK_EQ(state->SmallBlind(), 100);
  SPIEL_CHECK_EQ(state->BigBlind(), 200);
  std::vector<int> deal_and_fold_actions = {0, 1, 2, 3, 0};
  for (int i = 0; i < 10; ++i) {
    for (const auto& action : deal_and_fold_actions) state->ApplyAction(action);
  }
  SPIEL_CHECK_TRUE(absl::StrContains(state->ToString(), "P0: 400"));
  SPIEL_CHECK_TRUE(absl::StrContains(state->ToString(), "P1: 200"));
  SPIEL_CHECK_EQ(state->SmallBlind(), 200);
  SPIEL_CHECK_EQ(state->BigBlind(), 400);
  for (int i = 0; i < 20; ++i) {
    for (const auto& action : deal_and_fold_actions) state->ApplyAction(action);
  }
  SPIEL_CHECK_TRUE(absl::StrContains(state->ToString(), "P0: 800"));
  SPIEL_CHECK_TRUE(absl::StrContains(state->ToString(), "P1: 400"));
  SPIEL_CHECK_EQ(state->SmallBlind(), 400);
  SPIEL_CHECK_EQ(state->BigBlind(), 800);
  for (int i = 0; i < 60; ++i) {
    for (const auto& action : deal_and_fold_actions) state->ApplyAction(action);
  }
  // Even though we've exceeded the schedule, blinds should still be set to
  // last level.
  SPIEL_CHECK_TRUE(absl::StrContains(state->ToString(), "P0: 800"));
  SPIEL_CHECK_TRUE(absl::StrContains(state->ToString(), "P1: 400"));
  SPIEL_CHECK_EQ(state->SmallBlind(), 400);
  SPIEL_CHECK_EQ(state->BigBlind(), 800);
  // Play to end of game.
  for (int i = 0; i < 10; ++i) {
    for (const auto& action : deal_and_fold_actions) state->ApplyAction(action);
  }
  SPIEL_CHECK_TRUE(state->IsTerminal());
  // Players folded every hand an even number of times so they each have a
  // return of 0.
  SPIEL_CHECK_EQ(state->Returns(), std::vector<double>(2, 0.0));
}

void SerializationTest() {
  std::shared_ptr<const Game> game =
      LoadGame("repeated_poker",
               {{"max_num_hands", GameParameter(100)},
                {"reset_stacks", GameParameter(false)},
                {"rotate_dealer", GameParameter(true)},
                {"blind_schedule", GameParameter(
                    "10:100/200;20:200/400;50:400/800;")},
                {"universal_poker_game_string",
                 GameParameterFromString(open_spiel::HunlGameString(
                     "fullgame"))}});
  std::string game_str = game->ToString();
  std::shared_ptr<const Game> game2 = LoadGame(game_str);
  std::string game_str2 = game->ToString();
  SPIEL_CHECK_EQ(game_str, game_str2);
}

}  // namespace
}  // namespace repeated_poker
}  // namespace universal_poker
}  // namespace open_spiel

int main(int argc, char **argv) {
  open_spiel::Init("", &argc, &argv, true);
  absl::ParseCommandLine(argc, argv);
  open_spiel::universal_poker::repeated_poker::BasicRepeatedPokerTest();
  open_spiel::universal_poker::repeated_poker::BlindScheduleTest();
  open_spiel::universal_poker::repeated_poker::SerializationTest();
}
