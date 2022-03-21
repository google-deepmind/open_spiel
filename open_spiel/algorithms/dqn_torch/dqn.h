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

#ifndef OPEN_SPIEL_ALGORITHMS_DQN_TORCH_DQN_H_
#define OPEN_SPIEL_ALGORITHMS_DQN_TORCH_DQN_H_

#include <torch/torch.h>

#include <memory>
#include <random>
#include <string>
#include <vector>

#include "open_spiel/algorithms/dqn_torch/simple_nets.h"
#include "open_spiel/abseil-cpp/absl/random/random.h"
#include "open_spiel/policy.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/utils/circular_buffer.h"

// Note: This implementation has only been lightly tested on a few small games.

namespace open_spiel {
namespace algorithms {
namespace torch_dqn {

struct Transition {
  std::vector<float> info_state;
  Action action;
  double reward;
  std::vector<float> next_info_state;
  bool is_final_step;
  std::vector<int> legal_actions_mask;
};

struct DQNSettings {
  int seed;
  bool use_observation;
  Player player_id;
  int state_representation_size;
  int num_actions;
  std::vector<int> hidden_layers_sizes = {128};
  int replay_buffer_capacity = 10000;
  int batch_size = 128;
  double learning_rate = 0.01;
  int update_target_network_every = 1000;
  int learn_every = 10;
  double discount_factor = 0.99;
  int min_buffer_size_to_learn = 1000;
  double epsilon_start = 1.0;
  double epsilon_end = 0.1;
  int epsilon_decay_duration = 1000000;
  std::string loss_str = "mse";
};

// TODO(author5): make this into a proper general RL env/agent API as we have
// in the Python API. Then include tabular q-learning and Sarsa as well.

// A general agent class with a Step function.
class Agent {
 public:
  virtual ~Agent() = default;
  virtual Action Step(const State& state, bool is_evaluation = false) = 0;
};

// Run a number of episodes with the given agents and return the average return
// for each agent over the episodes. Set is_evaluation to true when using this
// for evaluation.
std::vector<double> RunEpisodes(std::mt19937* rng, const Game& game,
                                const std::vector<Agent*>& agents,
                                int num_episodes, bool is_evaluation);

class RandomAgent : public Agent {
 public:
  RandomAgent(Player player, int seed) : player_(player), rng_(seed) {}
  virtual ~RandomAgent() = default;
  Action Step(const State& state, bool is_evaluation = false) override;

 private:
  Player player_;
  std::mt19937 rng_;
};

// DQN Agent implementation in LibTorch.
class DQN : public Agent {
 public:
  DQN(const DQNSettings& settings);
  virtual ~DQN() = default;
  Action Step(const State& state, bool is_evaluation = false) override;

  double GetEpsilon(bool is_evaluation, int power = 1.0);
  int seed() const { return seed_; }

 private:
  std::vector<float> GetInfoState(const State& state, Player player_id,
                                  bool use_observation);
  void AddTransition(const State& prev_state, Action prev_action,
                     const State& state);
  Action EpsilonGreedy(std::vector<float> info_state,
                       std::vector<Action> legal_actions, double epsilon);
  void Learn();

  int seed_;
  bool use_observation_;
  int player_id_;
  int num_actions_;
  std::vector<int> hidden_layers_sizes_;
  int update_target_network_every_;
  int learn_every_;
  int min_buffer_size_to_learn_;
  double discount_factor_;
  double epsilon_start_;
  double epsilon_end_;
  double epsilon_decay_duration_;
  CircularBuffer<Transition> replay_buffer_;
  int batch_size_;
  int step_counter_;
  bool exists_prev_;
  std::unique_ptr<State> prev_state_;
  Action prev_action_;
  int input_size_;
  std::string loss_str_;
  MLP q_network_;
  MLP target_q_network_;
  torch::optim::SGD optimizer_;
  std::mt19937 rng_;
};

}  // namespace torch_dqn
}  // namespace algorithms
}  // namespace open_spiel

#endif  // OPEN_SPIEL_ALGORITHMS_DQN_TORCH_DQN_H_
