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

#include <random>

#include "open_spiel/abseil-cpp/absl/random/distributions.h"
#include "open_spiel/algorithms/is_mcts.h"
#include "open_spiel/algorithms/mcts.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_bots.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace {

constexpr const int kSeed = 9492110;  // 93879211;

void PlayGWhist(int human_player, std::mt19937* rng, int num_rollouts) {
  std::shared_ptr<const Game> game = LoadGame("german_whist_foregame");
  std::random_device rd;
  int eval_seed = rd();
  int bot_seed = rd();
  auto evaluator =
      std::make_shared<algorithms::RandomRolloutEvaluator>(1, eval_seed);
  auto bot = std::make_unique<algorithms::ISMCTSBot>(
      bot_seed, evaluator, 0.7 * 13, num_rollouts,
      algorithms::kUnlimitedNumWorldSamples,
      algorithms::ISMCTSFinalPolicyType::kMaxVisitCount, true, false);
  std::unique_ptr<State> state = game->NewInitialState();
  while (!state->IsTerminal()) {
    Action chosen_action = kInvalidAction;
    if (state->IsChanceNode()) {
      chosen_action =
          SampleAction(state->ChanceOutcomes(), absl::Uniform(*rng, 0.0, 1.0))
              .first;
    } else if (state->CurrentPlayer() != human_player) {
      chosen_action = bot->Step(*state);
    } else {
      std::cout << state->InformationStateString(human_player) << std::endl;
      auto legal_actions = state->LegalActions();
      for (int i = 0; i < legal_actions.size(); ++i) {
        std::cout << state->ActionToString(legal_actions[i]) << ",";
      }
      std::cout << std::endl;
      std::cout << "Input action:";
      std::string input;
      std::cin >> input;
      chosen_action = state->StringToAction(input);
      std::cout << std::endl;
    }
    state->ApplyAction(chosen_action);
  }

  std::cout << "Terminal state:" << std::endl;
  std::cout << state->ToString() << std::endl;
  std::cout << "Returns: " << absl::StrJoin(state->Returns(), " ") << std::endl;
}

}  // namespace
}  // namespace open_spiel

int main(int argc, char** argv) {
  std::random_device rd;
  std::mt19937 rng(rd());
  int human_player;
  int num_rollouts;
  std::cout << "human_player:";
  std::cin >> human_player;
  std::cout << "\n";
  std::cout << "num_rollouts:";
  std::cin >> num_rollouts;
  std::cout << "\n";
  open_spiel::PlayGWhist(human_player, &rng, num_rollouts);
}
