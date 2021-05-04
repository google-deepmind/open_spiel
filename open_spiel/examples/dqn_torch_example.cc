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

#include "open_spiel/algorithms/dqn_torch/dqn.h"

#include <memory>

#include "open_spiel/abseil-cpp/absl/flags/flag.h"
#include "open_spiel/abseil-cpp/absl/flags/parse.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_globals.h"
#include "open_spiel/spiel_utils.h"

ABSL_FLAG(int, iter, 1000, "How many learn steps to run.");
ABSL_FLAG(int, eval_every, 300, "How often to evaluate the policy.");

float EvalAgent(std::shared_ptr<const open_spiel::Game> game,
                const std::unique_ptr<open_spiel::algorithms::torch_dqn::DQN>& agent,
                int num_episodes) {
  float rewards = 0.0;
  for (int i = 0; i < num_episodes; i++) {
    std::unique_ptr<open_spiel::State> state = game->NewInitialState();
    float episode_reward = 0.0;
    while (!state->IsTerminal()) {
      open_spiel::Action action = agent->Step(state, true);
      state->ApplyAction(action);
      episode_reward += state->Rewards()[0];
    }
    rewards += episode_reward;
  }
  return rewards / static_cast<float>(num_episodes);
}

void SolveCatch() {
  std::shared_ptr<const open_spiel::Game> game = open_spiel::LoadGame("catch");
  auto dqn = std::make_unique<open_spiel::algorithms::torch_dqn::DQN>(
      /*use_observation*/game->GetType().provides_observation_tensor,
      /*player_id*/0,
      /*state_representation_size*/game->ObservationTensorSize(),
      /*num_actions*/game->NumDistinctActions());
  int iter = absl::GetFlag(FLAGS_iter);
  int eval_every = absl::GetFlag(FLAGS_eval_every);
  int total_reward = 0;
  while (iter-- > 0) {
    std::unique_ptr<open_spiel::State> state = game->NewInitialState();
    while (!state->IsTerminal()) {
      open_spiel::Action action = dqn->Step(state);
      state->ApplyAction(action);
    }
    if (iter % eval_every == 0) {
      float reward = EvalAgent(game, dqn, 100);
      std::cout << reward << std::endl;
    }
  }
}

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  SolveCatch();
  return 0;
}
