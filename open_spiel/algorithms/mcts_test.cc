// Copyright 2019 DeepMind Technologies Ltd. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "open_spiel/algorithms/mcts.h"

#include "open_spiel/algorithms/evaluate_bots.h"
#include "open_spiel/spiel_bots.h"

namespace open_spiel {
namespace {

void BotTest_RandomVsRandom() {
  auto game = LoadGame("kuhn_poker");
  std::vector<std::shared_ptr<Bot>> bots = {
      MakeUniformRandomBot(*game, /*player_id=*/0, /*seed=*/1234),
      MakeUniformRandomBot(*game, /*player_id=*/1, /*seed=*/4321)};
  std::vector<Bot*> bot_ptrs = {bots[0].get(), bots[1].get()};
  constexpr int num_players = 2;
  std::vector<double> average_results(num_players);
  constexpr int num_iters = 100000;
  for (int iteration = 0; iteration < num_iters; ++iteration) {
    auto this_results = EvaluateBots(game->NewInitialState().get(), bot_ptrs,
                                     /*seed=*/iteration);
    for (auto i = Player{0}; i < num_players; ++i)
      average_results[i] += this_results[i];
  }
  for (auto i = Player{0}; i < num_players; ++i)
    average_results[i] /= num_iters;

  SPIEL_CHECK_FLOAT_NEAR(average_results[0], 0.125, 0.01);
  SPIEL_CHECK_FLOAT_NEAR(average_results[1], -0.125, 0.01);
}

}  // namespace
}  // namespace open_spiel

int main(int argc, char** argv) { open_spiel::BotTest_RandomVsRandom(); }
