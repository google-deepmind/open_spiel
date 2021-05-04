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

#ifndef OPEN_SPIEL_ALGORITHMS_DQN_H_
#define OPEN_SPIEL_ALGORITHMS_DQN_H_

#include "open_spiel/algorithms/dqn_torch/simple_nets.h"

#include <torch/torch.h>

#include <memory>
#include <random>
#include <vector>

#include "open_spiel/abseil-cpp/absl/random/random.h"
#include "open_spiel/policy.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/utils/circular_buffer.h"


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

// DQN Agent implementation in LibTorch.
class DQN {
  public: 
    DQN(bool use_observation,
        Player player_id,
        int state_representation_size,
        int num_actions,
        std::vector<int> hidden_layers_sizes={128},
        int replay_buffer_capacity=10000,
        int batch_size=128,
        double learning_rate=0.01,
        int update_target_network_every=1000,
        int learn_every=10,
        double discount_factor=1.0,
        int min_buffer_size_to_learn=1000,
        double epsilon_start=1.0,
        double epsilon_end=0.1,
        int epsilon_decay_duration=1000000,
        std::string optimizer_str="sgd",
        std::string loss_str="mse");
    virtual ~DQN() = default;
    Action Step(const std::unique_ptr<State>& state, bool is_evaluation=false, bool add_transition_record=true);
  private:
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
    int last_loss_value_;
    bool exists_prev_;
    std::unique_ptr<State> prev_state_;
    Action prev_action_;
    int setp_counter_;
    int input_size_;
    std::string loss_str_; 
    MLP q_network_;
    MLP target_q_network_;
    torch::optim::SGD optimizer_;
    std::mt19937 rng_;
    std::vector<float> GetInfoState(const std::unique_ptr<State>& state, Player player_id, bool use_observation);
    void AddTransition(const std::unique_ptr<State>& prev_state, Action prev_action, const std::unique_ptr<State>& state);
    Action EpsilonGreedy(std::vector<float> info_state, std::vector<Action> legal_actions, double epsilon);
    double GetEpsilon(bool is_evaluation, int power=1.0);
    void Learn();
};

}  // namespace torch_dqn
}  // namespace algorithms
}  // namespace open_spiel

#endif  // OPEN_SPIEL_ALGORITHMS_DQN_H_