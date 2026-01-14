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

#include "open_spiel/algorithms/is_mcts.h"

#include <random>

#include "open_spiel/abseil-cpp/absl/random/distributions.h"
#include "open_spiel/algorithms/mcts.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_bots.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace {

constexpr const int kSeed = 93879211;

void PlayGame(const Game& game, algorithms::ISMCTSBot* bot, std::mt19937* rng) {
  std::unique_ptr<State> state = game.NewInitialState();
  while (!state->IsTerminal()) {
    std::cout << "State:" << std::endl;
    std::cout << state->ToString() << std::endl;

    Action chosen_action = kInvalidAction;
    if (state->IsChanceNode()) {
      chosen_action =
          SampleAction(state->ChanceOutcomes(), absl::Uniform(*rng, 0.0, 1.0))
              .first;
    } else {
      chosen_action = bot->Step(*state);
    }

    std::cout << "Chosen action: " << state->ActionToString(chosen_action)
              << std::endl;
    state->ApplyAction(chosen_action);
  }

  std::cout << "Terminal state:" << std::endl;
  std::cout << state->ToString() << std::endl;
  std::cout << "Returns: " << absl::StrJoin(state->Returns(), " ") << std::endl;
}

void ISMCTSTest_PlayGame(const std::string& game_name) {
  std::shared_ptr<const Game> game = LoadGame(game_name);
  auto evaluator =
      std::make_shared<algorithms::RandomRolloutEvaluator>(1, kSeed);

  for (algorithms::ISMCTSFinalPolicyType type :
       {algorithms::ISMCTSFinalPolicyType::kNormalizedVisitCount,
        algorithms::ISMCTSFinalPolicyType::kMaxVisitCount,
        algorithms::ISMCTSFinalPolicyType::kMaxValue}) {
    auto bot1 = std::make_unique<algorithms::ISMCTSBot>(
        kSeed, evaluator, 5.0, 1000, algorithms::kUnlimitedNumWorldSamples,
        type, false, false);

    std::mt19937 rng(kSeed);

    std::cout << "Testing " << game_name << ", bot 1" << std::endl;
    PlayGame(*game, bot1.get(), &rng);

    auto bot2 = std::make_unique<algorithms::ISMCTSBot>(
        kSeed, evaluator, 5.0, 1000, 10, type, false, false);
    std::cout << "Testing " << game_name << ", bot 2" << std::endl;
    PlayGame(*game, bot2.get(), &rng);
  }
}

void ISMCTS_BasicPlayGameTest_Kuhn() {
  ISMCTSTest_PlayGame("kuhn_poker");
  ISMCTSTest_PlayGame("kuhn_poker(players=3)");
}

void ISMCTS_BasicPlayGameTest_Leduc() {
  ISMCTSTest_PlayGame("leduc_poker");
  ISMCTSTest_PlayGame("leduc_poker(players=3)");
}

void ISMCTS_LeducObservationTest() {
  std::mt19937 rng(kSeed);
  std::shared_ptr<const Game> game = LoadGame("leduc_poker");
  auto evaluator =
      std::make_shared<algorithms::RandomRolloutEvaluator>(1, kSeed);
  auto bot = std::make_unique<algorithms::ISMCTSBot>(
      kSeed, evaluator, 10.0, 1000, algorithms::kUnlimitedNumWorldSamples,
      algorithms::ISMCTSFinalPolicyType::kNormalizedVisitCount, true, true);
  PlayGame(*game, bot.get(), &rng);
}

}  // namespace
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::ISMCTS_BasicPlayGameTest_Kuhn();
  open_spiel::ISMCTS_BasicPlayGameTest_Leduc();
  open_spiel::ISMCTS_LeducObservationTest();
}
