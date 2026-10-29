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

#include <ctime>
#include <memory>
#include <random>
#include <vector>

#include "absl/algorithm/container.h"
#include "open_spiel/bots/gin_rummy/simple_gin_rummy_bot.h"
#include "open_spiel/games/gin_rummy/gin_rummy.h"
#include "open_spiel/games/gin_rummy/gin_rummy_utils.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_bots.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace gin_rummy {
namespace {

// In each of these positions the bot may knock only after a discard that
// changes the best meld group of its eleven cards. The bot must knock, discard
// that card and lay the melds of the ten cards it keeps.
void SimpleGinRummyBotMandatoryDiscardTest() {
  const std::vector<std::vector<Action>> histories = {
      {29, 48, 7,  47, 17, 31, 16, 41, 24, 44, 2,  14, 3,  13, 5,
       6,  25, 42, 9,  1,  0,  54, 52, 25, 53, 38, 38, 53, 11, 11,
       53, 23, 24, 53, 35, 35, 53, 34, 23, 53, 19, 9,  53, 27, 48,
       53, 28, 19, 53, 20, 47, 53, 22, 22, 53, 12, 12, 53, 37, 37,
       53, 10, 10, 53, 26, 6,  53, 4,  34, 53, 21, 21, 53, 18, 20,
       53, 51, 51, 53, 30, 7,  53, 46, 46, 53, 15},
      {31, 40, 24, 1,  7,  50, 5,  47, 28, 51, 22, 17, 48, 12, 34, 0,  41,
       10, 39, 21, 23, 54, 52, 12, 53, 45, 51, 53, 36, 48, 53, 29, 50, 53,
       3,  22, 53, 33, 24, 53, 14, 34, 53, 4,  47, 53, 32, 21, 53, 18, 33,
       53, 20, 20, 53, 37, 37, 53, 2,  32, 53, 38, 38, 53, 8,  8,  53, 30,
       7,  53, 42, 17, 52, 45, 53, 49, 42, 53, 9,  9,  53, 46, 46, 53, 16},
      {23, 3,  44, 51, 7,  18, 11, 34, 4,  17, 48, 39, 15, 0,
       33, 30, 24, 16, 13, 50, 9,  54, 54, 53, 28, 51, 53, 38,
       38, 53, 42, 11, 52, 48, 53, 2,  23, 53, 5,  33, 53, 43,
       34, 53, 6,  6,  53, 27, 7,  53, 37, 5,  52},
      {16, 45, 11, 29, 18, 23, 14, 31, 48, 26, 50, 4,  39, 8,  40, 1,  21,
       28, 17, 36, 30, 52, 11, 53, 10, 50, 53, 22, 23, 52, 21, 53, 41, 48,
       53, 32, 8,  53, 33, 22, 53, 37, 37, 53, 42, 33, 53, 9,  9,  53, 19,
       45, 53, 15, 32, 52, 19, 53, 44, 44, 52, 32, 53, 13, 17, 52},
  };
  const std::vector<int> expected_deadwood = {10, 9, 10, 10};
  std::shared_ptr<const Game> game = LoadGame("gin_rummy");
  GinRummyUtils utils(kDefaultNumRanks, kDefaultNumSuits, kDefaultHandSize);
  for (int i = 0; i < histories.size(); ++i) {
    std::unique_ptr<State> state = game->NewInitialState();
    for (Action action : histories[i]) {
      SPIEL_CHECK_TRUE(absl::c_linear_search(state->LegalActions(), action));
      state->ApplyAction(action);
    }
    Player player = state->CurrentPlayer();
    SimpleGinRummyBot bot(game->GetParameters(), player);
    SPIEL_CHECK_TRUE(
        absl::c_linear_search(state->LegalActions(), kKnockAction));
    SPIEL_CHECK_EQ(bot.Step(*state), kKnockAction);
    state->ApplyAction(kKnockAction);
    Action discard = bot.Step(*state);
    SPIEL_CHECK_TRUE(absl::c_linear_search(state->LegalActions(), discard));
    state->ApplyAction(discard);
    const auto& gin_state = down_cast<const GinRummyState&>(*state);
    SPIEL_CHECK_EQ(utils.MinDeadwood(gin_state.Hands()[player]),
                   expected_deadwood[i]);
    // The bot lays its melds and passes, all legally.
    while (!state->IsTerminal() && state->CurrentPlayer() == player) {
      Action action = bot.Step(*state);
      SPIEL_CHECK_TRUE(absl::c_linear_search(state->LegalActions(), action));
      state->ApplyAction(action);
    }
  }
}

void SimpleGinRummyBotSelfPlayTest() {
  const int num_games = 3;
  std::mt19937 rng(time(nullptr));
  auto game = LoadGame("gin_rummy");
  std::vector<std::unique_ptr<Bot>> bots;

  for (Player p = 0; p < kNumPlayers; ++p) {
    bots.push_back(
        std::make_unique<SimpleGinRummyBot>(game->GetParameters(), p));
  }

  for (int i = 0; i < num_games; i++) {
    for (Player p = 0; p < gin_rummy::kNumPlayers; ++p) bots[p]->Restart();
    std::unique_ptr<State> state = game->NewInitialState();
    while (!state->IsTerminal()) {
      Player player = state->CurrentPlayer();
      Action action;
      if (state->IsChanceNode()) {
        ActionsAndProbs outcomes = state->ChanceOutcomes();
        action = SampleAction(outcomes,
            std::uniform_real_distribution<double>(0.0, 1.0)(rng)).first;
      } else {
        action = bots[player]->Step(*state);
      }
      state->ApplyAction(action);
    }
  }
}

}  // namespace
}  // namespace gin_rummy
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::gin_rummy::SimpleGinRummyBotMandatoryDiscardTest();
  open_spiel::gin_rummy::SimpleGinRummyBotSelfPlayTest();
}
