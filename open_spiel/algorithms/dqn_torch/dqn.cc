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

#include <algorithm>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "open_spiel/abseil-cpp/absl/random/distributions.h"
#include "open_spiel/abseil-cpp/absl/random/random.h"
#include "open_spiel/policy.h"
#include "open_spiel/spiel.h"

namespace open_spiel {
namespace algorithms {
namespace torch_dqn {

constexpr const int kIllegalActionLogitsPenalty = -1e9;

Action RandomAgent::Step(const State& state, bool is_evaluation) {
  if (state.IsTerminal()) {
    return;
  }
  std::vector<Action> legal_actions = state.LegalActions(player_);
  int aidx = absl::Uniform<int>(rng_, 0, legal_actions.size());
  return legal_actions[aidx];
}

DQN::DQN(const DQNSettings& settings)
    : seed_(settings.seed),
      use_observation_(settings.use_observation),
      player_id_(settings.player_id),
      input_size_(settings.state_representation_size),
      num_actions_(settings.num_actions),
      hidden_layers_sizes_(settings.hidden_layers_sizes),
      batch_size_(settings.batch_size),
      update_target_network_every_(settings.update_target_network_every),
      learn_every_(settings.learn_every),
      min_buffer_size_to_learn_(settings.min_buffer_size_to_learn),
      discount_factor_(settings.discount_factor),
      epsilon_start_(settings.epsilon_start),
      epsilon_end_(settings.epsilon_end),
      epsilon_decay_duration_(settings.epsilon_decay_duration),
      replay_buffer_(settings.replay_buffer_capacity),
      q_network_(input_size_, hidden_layers_sizes_, num_actions_),
      target_q_network_(input_size_, hidden_layers_sizes_, num_actions_),
      optimizer_(q_network_->parameters(),
                 torch::optim::SGDOptions(settings.learning_rate)),
      loss_str_(settings.loss_str),
      exists_prev_(false),
      prev_state_(nullptr),
      step_counter_(0),
      rng_(settings.seed) {}

std::vector<float> DQN::GetInfoState(const State& state,
                                     Player player_id,
                                     bool use_observation) {
  if (use_observation) {
    return state.ObservationTensor(player_id);
  } else {
    return state.InformationStateTensor(player_id);
  }
}

Action DQN::Step(const State& state, bool is_evaluation) {
  // Chance nodes should be handled externally to the agent.
  SPIEL_CHECK_TRUE(!state.IsChanceNode());

  Action action;
  if (!state.IsTerminal() &&
      (state.CurrentPlayer() == player_id_ || state.IsSimultaneousNode())) {
    std::vector<float> info_state = GetInfoState(state,
                                                 player_id_,
                                                 use_observation_);
    std::vector<Action> legal_actions = state.LegalActions(player_id_);
    double epsilon = GetEpsilon(is_evaluation);
    action = EpsilonGreedy(info_state, legal_actions, epsilon);
  } else {
    action = 0;
  }

  if (!is_evaluation) {
    step_counter_++;

    if (step_counter_ % learn_every_ == 0) {
      Learn();
    }
    if (step_counter_ % update_target_network_every_ == 0) {
      torch::save(q_network_, "q_network.pt");
      torch::load(target_q_network_, "q_network.pt");
    }
    if (exists_prev_) {
      AddTransition(*prev_state_, prev_action_, state);
    }
  }

  if (state.IsTerminal()) {
    exists_prev_ = false;
    prev_action_ = 0;
    prev_state_ = nullptr;
    return kInvalidAction;
  } else {
    exists_prev_ = true;
    prev_state_ = state.Clone();
    prev_action_ = action;
  }

  return action;
}

void DQN::AddTransition(const State& prev_state,
                        Action prev_action,
                        const State& state) {
  // std::cout << "Adding transition: prev_action = " << prev_action
  //           << ", player id = " << player_id_
  //           << ", reward = " << state.PlayerReward(player_id_) << std::endl;
  Transition transition = {
    /*info_state=*/GetInfoState(prev_state, player_id_, use_observation_),
    /*action=*/prev_action_,
    /*reward=*/state.PlayerReward(player_id_),
    /*next_info_state=*/GetInfoState(state, player_id_, use_observation_),
    /*is_final_step=*/state.IsTerminal(),
    /*legal_actions_mask=*/state.LegalActionsMask()};
  replay_buffer_.Add(transition);
}

Action DQN::EpsilonGreedy(std::vector<float> info_state,
                          std::vector<Action> legal_actions,
                          double epsilon) {
  Action action;
  if (legal_actions.empty()) {
    // In some simultaneous games, some players can have no legal actions.
    return 0;
  } else if (legal_actions.size() == 1) {
    return legal_actions[0];
  }

  if (absl::Uniform(rng_, 0.0, 1.0) < epsilon) {
    ActionsAndProbs actions_probs;
    std::vector<double> probs(legal_actions.size(), 1.0/legal_actions.size());
    for (int i = 0; i < legal_actions.size(); i++) {
      actions_probs.push_back({legal_actions[i], probs[i]});
    }
    action = SampleAction(actions_probs, rng_).first;
  } else {
    torch::Tensor info_state_tensor = torch::from_blob(
        info_state.data(),
        {info_state.size()},
        torch::TensorOptions().dtype(torch::kFloat32)).view({1, -1});
    q_network_->eval();
    torch::Tensor q_value = q_network_->forward(info_state_tensor);
    torch::Tensor legal_actions_mask =
        torch::full({num_actions_}, kIllegalActionLogitsPenalty,
                    torch::TensorOptions().dtype(torch::kFloat32));
    for (Action a : legal_actions) {
      legal_actions_mask[a] = 0;
    }
    action = (q_value.detach() + legal_actions_mask).argmax(1).item().toInt();
  }
  return action;
}

double DQN::GetEpsilon(bool is_evaluation, int power) {
  if (is_evaluation) {
    return 0.0;
  }

  double decay_steps = std::min(
      static_cast<double>(step_counter_), epsilon_decay_duration_);
  double decayed_epsilon = (
    epsilon_end_ + (epsilon_start_ - epsilon_end_) *
    std::pow((1 - decay_steps / epsilon_decay_duration_), power));
  return decayed_epsilon;
}

void DQN::Learn() {
  if (replay_buffer_.Size() < batch_size_
      || replay_buffer_.Size() < min_buffer_size_to_learn_) return;
  std::vector<Transition> transition = replay_buffer_.Sample(&rng_,
                                                             batch_size_);
  std::vector<torch::Tensor> info_states;
  std::vector<torch::Tensor> next_info_states;
  std::vector<torch::Tensor> legal_actions_mask;
  std::vector<Action> actions;
  std::vector<float> rewards;
  std::vector<int> are_final_steps;
  for (auto t : transition) {
    info_states.push_back(
        torch::from_blob(
            t.info_state.data(),
            {1, t.info_state.size()},
            torch::TensorOptions().dtype(torch::kFloat32)).clone());
    next_info_states.push_back(
        torch::from_blob(
            t.next_info_state.data(),
            {1, t.next_info_state.size()},
            torch::TensorOptions().dtype(torch::kFloat32)).clone());
    legal_actions_mask.push_back(
        torch::from_blob(t.legal_actions_mask.data(),
                         {1, t.legal_actions_mask.size()},
                         torch::TensorOptions().dtype(torch::kInt32))
            .to(torch::kInt64)
            .clone());
    actions.push_back(t.action);
    rewards.push_back(t.reward);
    are_final_steps.push_back(t.is_final_step);
  }
  torch::Tensor info_states_tensor = torch::stack(info_states, 0);
  torch::Tensor next_info_states_tensor = torch::stack(next_info_states, 0);
  q_network_->train();
  torch::Tensor q_values = q_network_->forward(info_states_tensor);
  target_q_network_->eval();
  torch::Tensor target_q_values = target_q_network_->forward(
      next_info_states_tensor).detach();

  torch::Tensor legal_action_masks_tensor = torch::stack(legal_actions_mask, 0);
  torch::Tensor illegal_actions = 1.0 - legal_action_masks_tensor;
  torch::Tensor illegal_logits = illegal_actions * kIllegalActionLogitsPenalty;

  torch::Tensor max_next_q = std::get<0>(
      torch::max(target_q_values + illegal_logits, 2));
  torch::Tensor are_final_steps_tensor = torch::from_blob(
      are_final_steps.data(),
      {batch_size_},
      torch::TensorOptions().dtype(torch::kInt32)).to(torch::kFloat32);
  torch::Tensor rewards_tensor = torch::from_blob(
      rewards.data(),
      {batch_size_},
      torch::TensorOptions().dtype(torch::kFloat32));
  torch::Tensor target = rewards_tensor + (
      1.0 - are_final_steps_tensor) * max_next_q.squeeze(1) * discount_factor_;
  torch::Tensor actions_tensor = torch::from_blob(
      actions.data(),
      {batch_size_},
      torch::TensorOptions().dtype(torch::kInt64));
  torch::Tensor predictions = q_values.index(
      {torch::arange(q_values.size(0)),
                     torch::indexing::Slice(),
                     actions_tensor});

  optimizer_.zero_grad();
  torch::Tensor value_loss;
  if (loss_str_ == "mse") {
    torch::nn::MSELoss mse_loss;
    value_loss = mse_loss(predictions.squeeze(1), target);
  } else if (loss_str_ == "huber") {
    torch::nn::SmoothL1Loss l1_loss;
    value_loss = l1_loss(predictions.squeeze(1), target);
  } else {
    SpielFatalError("Not implemented, choose from 'mse', 'huber'.");
  }
  value_loss.backward();
  optimizer_.step();
}

std::vector<double> RunEpisodes(std::mt19937* rng, const Game& game,
                                const std::vector<Agent*>& agents,
                                int num_episodes, bool is_evaluation) {
  SPIEL_CHECK_GE(num_episodes, 1);
  SPIEL_CHECK_EQ(agents.size(), game.NumPlayers());
  std::vector<double> total_returns(game.NumPlayers(), 0.0);
  for (int i = 0; i < num_episodes; i++) {
    std::unique_ptr<open_spiel::State> state = game.NewInitialState();
    while (!state->IsTerminal()) {
      Player player = state->CurrentPlayer();
      open_spiel::Action action;
      if (state->IsChanceNode()) {
        action = open_spiel::SampleAction(state->ChanceOutcomes(),
                                          absl::Uniform(*rng, 0.0, 1.0))
                     .first;
        state->ApplyAction(action);
      } else if (state->IsSimultaneousNode()) {
        std::vector<Action> joint_action(game.NumPlayers());
        for (Player p = 0; p < game.NumPlayers(); ++p) {
          joint_action[p] = agents[p]->Step(*state, is_evaluation);
        }
        state->ApplyActions(joint_action);
      } else {
        action = agents[player]->Step(*state, is_evaluation);
        state->ApplyAction(action);
      }
    }
    std::vector<double> episode_returns = state->Returns();
    for (Player p = 0; p < game.NumPlayers(); ++p) {
      agents[p]->Step(*state, is_evaluation);
      total_returns[p] += episode_returns[p];
    }
  }

  for (Player p = 0; p < game.NumPlayers(); ++p) {
    total_returns[p] /= num_episodes;
  }

  return total_returns;
}

}  // namespace torch_dqn
}  // namespace algorithms
}  // namespace open_spiel
