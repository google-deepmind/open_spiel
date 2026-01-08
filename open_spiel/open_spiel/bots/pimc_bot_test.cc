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

#include "open_spiel/bots/pimc_bot.h"

#include <cstdint>
#include <ctime>
#include <iostream>
#include <memory>
#include <random>
#include <vector>

#include "open_spiel/games/hearts/hearts.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_bots.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace {

constexpr uint32_t kSeed = 18713687;

double hearts_value_function(const State& state, Player p) {
  const auto& hearts_state =
      open_spiel::down_cast<const hearts::HeartsState&>(state);
  return hearts::kTotalPositivePoints - hearts_state.Points(p);
}

void SimpleSelfPlayTest() {
  const int num_games = 3;
  std::mt19937 rng(time(nullptr));
  auto game = LoadGame("hearts");
  std::vector<std::unique_ptr<Bot>> bots;
  const int num_players = game->NumPlayers();

  for (Player p = 0; p < num_players; ++p) {
    bots.push_back(
        std::make_unique<PIMCBot>(hearts_value_function, p, kSeed + p, 10, 2));
  }

  for (int i = 0; i < num_games; i++) {
    int turn = 0;
    std::unique_ptr<State> state = game->NewInitialState();
    while (!state->IsTerminal()) {
      turn += 1;
      std::cout << "Game " << i << ", turn " << turn << std::endl;
      std::cout << "State:" << std::endl << state->ToString() << std::endl;
      Player player = state->CurrentPlayer();
      Action action;
      if (state->IsChanceNode()) {
        ActionsAndProbs outcomes = state->ChanceOutcomes();
        action = SampleAction(outcomes, std::uniform_real_distribution<double>(
                                            0.0, 1.0)(rng))
                     .first;
      } else {
        action = bots[player]->Step(*state);
      }
      std::cout << "Chose action: " << state->ActionToString(action)
                << std::endl;
      state->ApplyAction(action);
    }
  }
}

}  // namespace
}  // namespace open_spiel

int main(int argc, char** argv) { open_spiel::SimpleSelfPlayTest(); }
