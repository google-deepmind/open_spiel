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

#include "open_spiel/bots/uci/uci_bot.h"

#include <memory>

#include "open_spiel/abseil-cpp/absl/flags/flag.h"
#include "open_spiel/abseil-cpp/absl/flags/parse.h"
#include "open_spiel/algorithms/evaluate_bots.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_bots.h"
#include "open_spiel/utils/init.h"

ABSL_FLAG(std::string, binary, "random_uci_bot", "Name of the binary to run.");

namespace open_spiel {
namespace uci {
namespace {

inline constexpr const int kNumGames = 3;
inline constexpr const int kSeed = 12874681;

void RandomUciBotTest() {
  std::string binary = absl::GetFlag(FLAGS_binary);
  std::shared_ptr<const Game> game = LoadGame("chess");
  Options options = {};
  std::unique_ptr<UCIBot> bot1 = std::make_unique<UCIBot>(
      binary, /*move_time*/100, /*ponder*/false, /*options*/options);
  std::unique_ptr<UCIBot> bot2 = std::make_unique<UCIBot>(
      binary, /*move_time*/100, /*ponder*/false, /*options*/options);
  std::vector<Bot*> bots = {bot1.get(), bot2.get()};
  for (int i = 0; i < kNumGames; ++i) {
    std::unique_ptr<State> state = game->NewInitialState();
    EvaluateBots(state.get(), bots, kSeed);
    std::cout << "Game over: " << state->HistoryString() << std::endl;
  }
}

}  // namespace
}  // namespace uci
}  // namespace open_spiel

int main(int argc, char **argv) {
  open_spiel::Init("", &argc, &argv, false);
  absl::ParseCommandLine(argc, argv);
  open_spiel::uci::RandomUciBotTest();
}
