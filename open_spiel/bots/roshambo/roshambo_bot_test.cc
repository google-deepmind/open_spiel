// Copyright 2021 DeepMind Technologies Limited
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

#include "open_spiel/bots/roshambo/roshambo_bot.h"

#include <functional>
#include <utility>

#include "open_spiel/abseil-cpp/absl/time/clock.h"
#include "open_spiel/abseil-cpp/absl/time/time.h"
#include "open_spiel/game_transforms/repeated_game.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_bots.h"
#include "open_spiel/bots/roshambo/roshambo/bot_map.h"

namespace open_spiel {
namespace {

uint_fast32_t Seed() { return absl::ToUnixMicros(absl::Now()); }

void MakeAllRoshamboBots() {
  std::vector<std::unique_ptr<Bot>> bots;
  for (std::pair<std::string, std::function<int()>> bot_pair :
       ::roshambo_tournament::bot_map) {
    bots.push_back(roshambo::MakeRoshamboBot(0, bot_pair.first));
  }
  SPIEL_CHECK_EQ(bots.size(), roshambo::kNumBots);
}

// This matchup is deterministic and both bots utilize the match history so
// we can test that the bots are perceiving the game correctly.
void RoshamboBotHistoryTest() {
  GameParameters params;
  params["num_repetitions"] = GameParameter(roshambo::kNumThrows);
  std::shared_ptr<const Game> game = CreateRepeatedGame("matrix_rps", params);

  std::vector<std::unique_ptr<Bot>> bots;
  bots.push_back(roshambo::MakeRoshamboBot(0, "rotatebot"));
  bots.push_back(roshambo::MakeRoshamboBot(1, "copybot"));
  std::unique_ptr<State> state = game->NewInitialState();

  const int num_players = bots.size();
  std::vector<Action> joint_actions(bots.size());
  for (int i = 0; i < roshambo::kNumThrows; ++i) {
    for (Player p = 0; p < num_players; ++p)
      joint_actions[p] = bots[p]->Step(*state);
    state->ApplyActions(joint_actions);
    if (i == 0) {
      // Copybot wins the first round.
      SPIEL_CHECK_EQ(state->PlayerReturn(0), -1);
      SPIEL_CHECK_EQ(state->PlayerReturn(1), 1);
    } else {
      // All subsequent rounds are draws.
      SPIEL_CHECK_EQ(state->PlayerReward(0), 0);
      SPIEL_CHECK_EQ(state->PlayerReward(1), 0);
    }
  }
  SPIEL_CHECK_TRUE(state->IsTerminal());
  SPIEL_CHECK_EQ(state->PlayerReturn(0), -1);
  SPIEL_CHECK_EQ(state->PlayerReturn(1), 1);
}

// Matchup between 1999 and 2000 tournament winners!
void RoshamboBotBasicPlayGame() {
  int num_games = 5;
  std::mt19937 rng(Seed());
  GameParameters params;
  params["num_repetitions"] = GameParameter(roshambo::kNumThrows);
  std::shared_ptr<const Game> game = CreateRepeatedGame("matrix_rps", params);
  std::vector<std::unique_ptr<Bot>> bots;

  bots.push_back(roshambo::MakeRoshamboBot(0, "greenberg"));
  bots.push_back(roshambo::MakeRoshamboBot(1, "iocainebot"));

  for (int i = 0; i < num_games; i++) {
    // Set seed for the underlying C code
    srandom(Seed());
    std::unique_ptr<State> state = game->NewInitialState();

    const int num_players = bots.size();
    std::vector<Action> joint_actions(bots.size());
    for (Player p = 0; p < num_players; ++p) bots[p]->Restart();
    while (!state->IsTerminal()) {
      for (Player p = 0; p < num_players; ++p)
        joint_actions[p] = bots[p]->Step(*state);
      state->ApplyActions(joint_actions);
    }
  }
}

}  // namespace
}  // namespace open_spiel

int main(int argc, char **argv) {
  open_spiel::MakeAllRoshamboBots();
  open_spiel::RoshamboBotHistoryTest();
  open_spiel::RoshamboBotBasicPlayGame();
}
