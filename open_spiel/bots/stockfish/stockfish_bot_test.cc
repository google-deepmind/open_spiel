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

#include "open_spiel/bots/stockfish/stockfish_bot.h"

#include <iostream>
#include <limits>

#include "open_spiel/abseil-cpp/absl/random/distributions.h"
#include "open_spiel/abseil-cpp/absl/time/clock.h"
#include "open_spiel/abseil-cpp/absl/time/time.h"
#include "open_spiel/algorithms/evaluate_bots.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_bots.h"

namespace open_spiel {
namespace {

uint_fast32_t Seed() { return absl::ToUnixMicros(absl::Now()); }

void TestStockfish(Bot *opponent_bot, int argc, char **argv) {
  std::mt19937 rng(Seed());
  auto game = open_spiel::LoadGame("chess");
  std::vector<std::unique_ptr<open_spiel::Bot>> bots;
  std::vector<Bot *> bot_ptrs;

  bots.push_back(stockfish::MakeStockfishBot(argc, argv, 30));
  bot_ptrs.push_back(bots.back().get());

  bot_ptrs.push_back(opponent_bot);

  std::unique_ptr<open_spiel::State> state = game->NewInitialState();
  auto returns = EvaluateBots(state.get(), bot_ptrs,
                              absl::Uniform<int>(rng, 0, std::numeric_limits<int>::max()));
  std::cout << returns << std::endl;
}

void TestStockfishAgainstUniformRandom(int argc, char **argv) {
  auto opponent_bot = MakeUniformRandomBot(1, Seed());
  TestStockfish(opponent_bot.get(), argc, argv);
}

void TestStockfishAgainstStockfish(int argc, char **argv) {
  auto opponent_bot = stockfish::MakeStockfishBot(argc, argv, 30);
  TestStockfish(opponent_bot.get(), argc, argv);
}

}  // namespace
}  // namespace open_spiel

int main(int argc, char **argv) {
  open_spiel::TestStockfishAgainstUniformRandom(argc, argv);
  open_spiel::TestStockfishAgainstStockfish(argc, argv);
}
