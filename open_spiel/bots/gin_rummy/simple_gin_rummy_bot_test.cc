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
#include <vector>

#include "open_spiel/bots/gin_rummy/simple_gin_rummy_bot.h"
#include "open_spiel/games/gin_rummy.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_bots.h"

namespace open_spiel {
namespace gin_rummy {
namespace {

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
  open_spiel::gin_rummy::SimpleGinRummyBotSelfPlayTest();
}
