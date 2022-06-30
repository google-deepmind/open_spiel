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

#include "open_spiel/games/catch.h"

#include "open_spiel/algorithms/get_all_states.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/tests/basic_tests.h"

namespace open_spiel {
namespace catch_ {
namespace {

namespace testing = open_spiel::testing;

void BasicCatchTests() {
  testing::LoadGameTest("catch");
  testing::ChanceOutcomesTest(*LoadGame("catch"));
  testing::RandomSimTest(*LoadGame("catch"), 100);
}

void GetAllStatesTest() {
  auto catch_game = LoadGame("catch");
  auto states = algorithms::GetAllStates(*catch_game,
                                         /*depth_limit=*/-1,
                                         /*include_terminals=*/true,
                                         /*include_chance_states=*/false);
  SPIEL_CHECK_EQ(
      states.size(),
      kDefaultRows * kDefaultColumns * kDefaultColumns - 6 * kDefaultColumns);

  // Verify number of states that lead to win and loss.
  int num_wins = 0;
  int num_losses = 0;
  for (const auto& pair : states) {
    const auto& state = pair.second;
    if (state->IsTerminal()) {
      if (state->PlayerReturn(0) == 1)
        num_wins++;
      else if (state->PlayerReturn(0) == -1)
        num_losses++;
      else
        SpielFatalError("Unexpected return");
    }
  }
  SPIEL_CHECK_EQ(num_wins, 5);
  SPIEL_CHECK_EQ(num_losses, 20);

  // Verify normalized observation matches string represtation.
  for (const auto& pair : states) {
    std::vector<float> obs(catch_game->ObservationTensorSize());
    pair.second->ObservationTensor(0, absl::MakeSpan(obs));
    const std::string& str = pair.first;
    SPIEL_CHECK_EQ(obs.size(), str.size() - kDefaultRows);
    for (int i = 0; i < obs.size(); i++) {
      SPIEL_CHECK_EQ(obs[i] == 1, str[i + i / kDefaultColumns] != '.');
    }
  }
}

void PlayAndWinTest() {
  auto game = LoadGame("catch");
  auto state = game->NewInitialState();
  SPIEL_CHECK_EQ(state->CurrentPlayer(), kChancePlayerId);
  state->ApplyAction(3);
  SPIEL_CHECK_EQ(state->CurrentPlayer(), 0);
  state->ApplyAction(2);  // Right.
  for (int i = 0; i < kDefaultRows - 2; i++) {
    SPIEL_CHECK_FALSE(state->IsTerminal());
    state->ApplyAction(1);  // Stay.
  }
  SPIEL_CHECK_TRUE(state->IsTerminal());
  SPIEL_CHECK_EQ(state->PlayerReturn(0), 1);
}

void ToStringTest() {
  auto game = LoadGame("catch");
  auto state = game->NewInitialState();
  state->ApplyAction(3);
  SPIEL_CHECK_EQ(state->ToString(),
                 "...o.\n"
                 ".....\n"
                 ".....\n"
                 ".....\n"
                 ".....\n"
                 ".....\n"
                 ".....\n"
                 ".....\n"
                 ".....\n"
                 "..x..\n");
}

}  // namespace
}  // namespace catch_
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::catch_::BasicCatchTests();
  open_spiel::catch_::GetAllStatesTest();
  open_spiel::catch_::PlayAndWinTest();
  open_spiel::catch_::ToStringTest();
}
