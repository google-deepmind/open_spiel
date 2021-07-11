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
#include <random>

#include "open_spiel/abseil-cpp/absl/flags/flag.h"
#include "open_spiel/abseil-cpp/absl/flags/parse.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_globals.h"
#include "open_spiel/spiel_utils.h"

ABSL_FLAG(int, max_iterations, 1000, "How many learn steps to run.");
ABSL_FLAG(int, eval_every, 300, "How often to evaluate the policy.");

float EvalAgent(
    std::mt19937* rng,
    std::shared_ptr<const open_spiel::Game> game,
    const std::unique_ptr<open_spiel::algorithms::torch_dqn::DQN>& agent,
    int num_episodes) {
  double total_returns = 0.0;
  for (int i = 0; i < num_episodes; i++) {
    std::unique_ptr<open_spiel::State> state = game->NewInitialState();
    double episode_return = 0.0;
    while (!state->IsTerminal()) {
      open_spiel::Action action;
      if (state->IsChanceNode()) {
        action = open_spiel::SampleAction(state->ChanceOutcomes(),
                                          absl::Uniform(*rng, 0.0, 1.0)).first;
      } else {
        action = agent->Step(*state, true);
      }
      state->ApplyAction(action);
      episode_return += state->Rewards()[0];
    }
    agent->Step(*state, true);
    SPIEL_CHECK_EQ(episode_return, state->Returns()[0]);
    total_returns += episode_return;
  }
  return total_returns / num_episodes;
}

void SolveCatch() {
  std::mt19937 rng;
  std::shared_ptr<const open_spiel::Game> game = open_spiel::LoadGame("catch");

  // Values copied from: python/examples/single_agent_catch.py
  open_spiel::algorithms::torch_dqn::DQNSettings settings = {
    /*use_observation*/game->GetType().provides_observation_tensor,
    /*player_id*/0,
    /*state_representation_size*/game->ObservationTensorSize(),
    /*num_actions*/game->NumDistinctActions(),
    /*hidden_layers_sizes*/{32, 32},
    /*replay_buffer_capacity*/10000,
    /*batch_size*/128,
    /*learning_rate*/0.1,
    /*update_target_network_every*/250,
    /*learn_every*/10,
    /*discount_factor*/0.99,
    /*min_buffer_size_to_learn*/1000,
    /*epsilon_start*/1.0,
    /*epsilon_end*/0.1,
    /*epsilon_decay_duration*/2000
  };
  auto dqn = std::make_unique<open_spiel::algorithms::torch_dqn::DQN>(settings);
  int max_iterations = absl::GetFlag(FLAGS_max_iterations);
  int eval_every = absl::GetFlag(FLAGS_eval_every);
  int total_reward = 0;
  for (int iter = 0; iter < max_iterations; ++iter) {
    std::unique_ptr<open_spiel::State> state = game->NewInitialState();
    while (!state->IsTerminal()) {
      open_spiel::Action action;
      if (state->IsChanceNode()) {
        action = open_spiel::SampleAction(state->ChanceOutcomes(),
                                          absl::Uniform(rng, 0.0, 1.0)).first;
      } else {
        action = dqn->Step(*state);
      }
      state->ApplyAction(action);
    }
    dqn->Step(*state);
    if (iter % eval_every == 0) {
      float reward = EvalAgent(&rng, game, dqn, 100);
      std::cout << iter << " " << reward << std::endl;
    }
  }
}

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  SolveCatch();
  return 0;
}
