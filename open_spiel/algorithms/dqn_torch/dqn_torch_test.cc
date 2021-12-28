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

#include "open_spiel/algorithms/dqn_torch/dqn.h"

#include <torch/torch.h>

#include <iostream>
#include <string>
#include <vector>

#include "open_spiel/game_parameters.h"
#include "open_spiel/spiel.h"
#include "open_spiel/games/efg_game.h"
#include "open_spiel/games/efg_game_data.h"


namespace open_spiel {
namespace algorithms {
namespace torch_dqn {
namespace {

constexpr int kSeed = 93879211;

void TestSimpleGame() {
  std::shared_ptr<const Game> game = efg_game::LoadEFGGame(
      efg_game::GetSimpleForkEFGData());
  SPIEL_CHECK_TRUE(game != nullptr);
  DQNSettings settings = {
      /*seed*/ kSeed,
      /*use_observation*/ game->GetType().provides_observation_tensor,
      /*player_id*/ 0,
      /*state_representation_size*/ game->InformationStateTensorSize(),
      /*num_actions*/ game->NumDistinctActions(),
      /*hidden_layers_sizes*/ {16},
      /*replay_buffer_capacity*/ 100,
      /*batch_size*/ 5,
      /*learning_rate*/ 0.01,
      /*update_target_network_every*/ 20,
      /*learn_every*/ 5,
      /*discount_factor*/ 1.0,
      /*min_buffer_size_to_learn*/ 5,
      /*epsilon_start*/ 0.02,
      /*epsilon_end*/ 0.01};
  DQN dqn(settings);
  int total_reward = 0;
  std::unique_ptr<State> state;
  for (int i = 0; i < 150; i++) {
    state = game->NewInitialState();
    while (!state->IsTerminal()) {
      open_spiel::Action action = dqn.Step(*state);
      state->ApplyAction(action);
      total_reward += state->PlayerReward(0);
    }
    dqn.Step(*state);
  }

  SPIEL_CHECK_GE(total_reward, 120);
}

void TestTicTacToe() {
  std::shared_ptr<const Game> game = open_spiel::LoadGame("tic_tac_toe");
  SPIEL_CHECK_TRUE(game != nullptr);
  std::vector<std::unique_ptr<DQN>> agents;
  std::vector<int> hidden_layers = {16};
  for (int i = 0; i < 2; i++) {
    DQNSettings settings = {
        /*seed*/ kSeed + i,
        /*use_observation*/ game->GetType().provides_observation_tensor,
        /*player_id*/ i,
        /*state_representation_size*/ game->ObservationTensorSize(),
        /*num_actions*/ game->NumDistinctActions(),
        /*hidden_layers_sizes*/ hidden_layers,
        /*replay_buffer_capacity*/ 10,
        /*batch_size*/ 5,
        /*learning_rate*/ 0.01,
        /*update_target_network_every*/ 20,
        /*learn_every*/ 5,
        /*discount_factor*/ 1.0,
        /*min_buffer_size_to_learn*/ 5};
    agents.push_back(std::make_unique<DQN>(settings));
  }
  std::unique_ptr<State> state = game->NewInitialState();
  while (!state->IsTerminal()) {
    Player current_player = state->CurrentPlayer();
    open_spiel::Action action = agents[current_player]->Step(*state);
    state->ApplyAction(action);
  }
  for (int i = 0; i < 2; i++) {
    agents[i]->Step(*state);
  }
}

void TestHanabi() {
  std::shared_ptr<const Game> game = open_spiel::LoadGame("tiny_hanabi");
  SPIEL_CHECK_TRUE(game != nullptr);
  std::vector<std::unique_ptr<DQN>> agents;
  std::vector<int> hidden_layers = {16};
  std::mt19937 rng_;
  for (int i = 0; i < 2; i++) {
    DQNSettings settings = {
        /*seed*/ kSeed + i,
        /*use_observation*/ game->GetType().provides_observation_tensor,
        /*player_id*/ i,
        /*state_representation_size*/ game->InformationStateTensorSize(),
        /*num_actions*/ game->NumDistinctActions(),
        /*hidden_layers_sizes*/ hidden_layers,
        /*replay_buffer_capacity*/ 10,
        /*batch_size*/ 5,
        /*learning_rate*/ 0.01,
        /*update_target_network_every*/ 20,
        /*learn_every*/ 5,
        /*discount_factor*/ 1.0,
        /*min_buffer_size_to_learn*/ 5};
    agents.push_back(std::make_unique<DQN>(settings));
  }
  std::unique_ptr<State> state = game->NewInitialState();
  while (!state->IsTerminal()) {
    Player current_player = state->CurrentPlayer();
    open_spiel::Action action;
    if (state->IsChanceNode()) {
      action = open_spiel::SampleAction(state->ChanceOutcomes(),
                                        absl::Uniform(rng_, 0.0, 1.0)).first;
    } else {
      // Simultaneous move game, step both!
      for (int i = 0; i < 2; i++) {
        if (i == current_player) {
          action = agents[i]->Step(*state);
        } else {
          agents[i]->Step(*state);
        }
      }
    }
    state->ApplyAction(action);
  }
  for (int i = 0; i < 2; i++) {
    agents[i]->Step(*state);
  }
}
}  // namespace
}  // namespace torch_dqn
}  // namespace algorithms
}  // namespace open_spiel

namespace torch_dqn = open_spiel::algorithms::torch_dqn;

int main(int args, char** argv) {
  torch::manual_seed(torch_dqn::kSeed);
  torch_dqn::TestSimpleGame();
  torch_dqn::TestTicTacToe();
  torch_dqn::TestHanabi();
  return 0;
}
