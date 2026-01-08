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

#include "open_spiel/algorithms/evaluate_bots.h"

#include <memory>

#include "open_spiel/policy.h"
#include "open_spiel/spiel_bots.h"

namespace open_spiel {
namespace {

void BotTest_RandomVsRandom() {
  auto game = LoadGame("kuhn_poker");
  auto bot0 = MakeUniformRandomBot(0, /*seed=*/1234);
  auto bot1 = MakeStatefulRandomBot(*game, 1, /*seed=*/4321);
  constexpr int num_players = 2;
  std::vector<double> average_results(num_players);
  constexpr int num_iters = 100000;
  for (int iteration = 0; iteration < num_iters; ++iteration) {
    auto this_results =
        EvaluateBots(game->NewInitialState().get(), {bot0.get(), bot1.get()},
                     /*seed=*/iteration);
    for (auto p = Player{0}; p < num_players; ++p) {
      average_results[p] += this_results[p];
    }
  }
  for (auto p = Player{0}; p < num_players; ++p) {
    average_results[p] /= num_iters;
  }

  SPIEL_CHECK_FLOAT_NEAR(average_results[0], 0.125, 0.01);
  SPIEL_CHECK_FLOAT_NEAR(average_results[1], -0.125, 0.01);
}

void BotTest_RandomVsRandomPolicy() {
  auto game = LoadGame("kuhn_poker");
  auto bot0 = MakeUniformRandomBot(0, /*seed=*/1234);
  std::unique_ptr<Policy> uniform_policy =
      std::make_unique<TabularPolicy>(GetUniformPolicy(*game));
  auto bot1 =
      MakePolicyBot(*game, Player{1}, /*seed=*/4321, std::move(uniform_policy));
  constexpr int num_players = 2;
  std::vector<double> average_results(num_players);
  constexpr int num_iters = 100000;
  for (int iteration = 0; iteration < num_iters; ++iteration) {
    auto this_results =
        EvaluateBots(game->NewInitialState().get(), {bot0.get(), bot1.get()},
                     /*seed=*/iteration);
    for (auto p = Player{0}; p < num_players; ++p) {
      average_results[p] += this_results[p];
    }
  }
  for (auto p = Player{0}; p < num_players; ++p) {
    average_results[p] /= num_iters;
  }
  SPIEL_CHECK_FLOAT_NEAR(average_results[0], 0.125, 0.01);
  SPIEL_CHECK_FLOAT_NEAR(average_results[1], -0.125, 0.01);
}

}  // namespace
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::BotTest_RandomVsRandom();
  open_spiel::BotTest_RandomVsRandomPolicy();
}
