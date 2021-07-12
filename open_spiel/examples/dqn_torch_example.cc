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

ABSL_FLAG(int, seed, 8263487, "Seed to use for random number generation.");

namespace open_spiel {
namespace algorithms {
namespace torch_dqn {
namespace {

void SolveCatch(int seed, int total_episodes, int report_every,
                int num_eval_episodes) {
  std::cout << "Solving catch" << std::endl;
  std::mt19937 rng(seed);
  std::shared_ptr<const open_spiel::Game> game = open_spiel::LoadGame("catch");

  // Values copied from: python/examples/single_agent_catch.py
  DQNSettings settings = {
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
  auto dqn = std::make_unique<DQN>(settings);
  std::vector<Agent*> agents = {dqn.get()};

  for (int num_episodes = 0;
       num_episodes < total_episodes;
       num_episodes += report_every) {
    // Training
    RunEpisodes(&rng, *game, agents,
                /*num_episodes*/report_every, /*is_evaluation*/false);

    std::vector<double> avg_returns = 
        RunEpisodes(&rng, *game, agents,
                    /*num_episodes*/num_eval_episodes, /*is_evaluation*/true);

    std::cout << num_episodes + report_every << " "
              << avg_returns[0] << std::endl;
  }
}

}  // namespace
}  // namespace torch_dqn
}  // namespace algorithms
}  // namespace open_spiel

namespace torch_dqn = open_spiel::algorithms::torch_dqn;

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  torch_dqn::SolveCatch(absl::GetFlag(FLAGS_seed), /*total_episodes*/2000,
                        /*report_every*/250, /*num_eval_episodes*/100);
  return 0;
}
