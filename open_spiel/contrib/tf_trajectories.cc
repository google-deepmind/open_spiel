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

#include "open_spiel/contrib/tf_trajectories.h"

#include <algorithm>
#include <cstring>
#include <memory>
#include <numeric>
#include <random>
#include <vector>

#include "open_spiel/abseil-cpp/absl/random/uniform_int_distribution.h"
#include "open_spiel/abseil-cpp/absl/strings/str_join.h"
#include "unsupported/Eigen/CXX11/Tensor"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace algorithms {

namespace tf = tensorflow;
using Tensor = Eigen::Tensor<float, 2, Eigen::RowMajor>;
using TensorMap = Eigen::TensorMap<Tensor, Eigen::Aligned>;

TFBatchTrajectoryRecorder::TFBatchTrajectoryRecorder(
    const Game& game, const std::string& graph_filename, int batch_size)
    : batch_size_(batch_size),
      states_(),
      terminal_flags_(std::vector<int>(batch_size, 0)),
      num_terminals_(0),
      game_(game.shared_from_this()),
      graph_filename_(graph_filename),
      rng_(),
      dist_(0.0, 1.0),
      flat_input_size_(game_->ObservationTensorSize()),
      num_actions_(game_->NumDistinctActions()) {
  TF_CHECK_OK(
      ReadBinaryProto(tf::Env::Default(), graph_filename_, &graph_def_));
  InitTF();
}

void TFBatchTrajectoryRecorder::Reset() {
  num_terminals_ = 0;
  terminal_flags_.resize(batch_size_);
  std::fill(terminal_flags_.begin(), terminal_flags_.end(), 0);
  ResetInitialStates();
}

void TFBatchTrajectoryRecorder::SampleChance(int idx) {
  while (states_[idx]->IsChanceNode()) {
    std::vector<std::pair<open_spiel::Action, double>> outcomes =
        states_[idx]->ChanceOutcomes();
    Action action = open_spiel::SampleAction(outcomes, dist_(rng_)).first;
    states_[idx]->ApplyAction(action);
  }

  if (states_[idx]->IsTerminal()) {
    num_terminals_++;
    terminal_flags_[idx] = 1;
  }
}

void TFBatchTrajectoryRecorder::ResetInitialStates() {
  states_.resize(batch_size_);
  for (int b = 0; b < batch_size_; ++b) {
    states_[b] = game_->NewInitialState();
    SampleChance(b);
  }
}

void TFBatchTrajectoryRecorder::GetNextStatesUniform() {
  for (int b = 0; b < batch_size_; ++b) {
    if (!terminal_flags_[b]) {
      std::vector<Action> actions = states_[b]->LegalActions();
      absl::uniform_int_distribution<> dist(0, actions.size() - 1);
      Action action = actions[dist(rng_)];
      states_[b]->ApplyAction(action);
      SampleChance(b);
    }
  }
}

void TFBatchTrajectoryRecorder::InitTF() {
  tf_inputs_ = tf::Tensor(tf::DT_FLOAT,
                          tf::TensorShape({batch_size_, flat_input_size_}));
  tf_legal_mask_ =
      tf::Tensor(tf::DT_FLOAT, tf::TensorShape({batch_size_, num_actions_}));

  // Set GPU options
  tf::graph::SetDefaultDevice("/cpu:0", &graph_def_);

  if (tf_session_ != nullptr) {
    TF_CHECK_OK(tf_session_->Close());
  }

  // create a new session
  TF_CHECK_OK(NewSession(tf_opts_, &tf_session_));

  // Load graph into session
  TF_CHECK_OK(tf_session_->Create(graph_def_));

  // Initialize our variables
  TF_CHECK_OK(tf_session_->Run({}, {}, {"init_all_vars_op"}, nullptr));
}

void TFBatchTrajectoryRecorder::FillInputsAndMasks() {
  TensorMap inputs_matrix = tf_inputs_.matrix<float>();
  TensorMap mask_matrix = tf_legal_mask_.matrix<float>();

  std::vector<float> info_state_vector(game_->ObservationTensorSize());
  for (int b = 0; b < batch_size_; ++b) {
    if (!terminal_flags_[b]) {
      std::vector<int> mask = states_[b]->LegalActionsMask();
      // Is there a way to use a vector operation here?
      for (int a = 0; a < mask.size(); ++a) {
        mask_matrix(b, a) = mask[a];
      }

      states_[b]->ObservationTensor(states_[b]->CurrentPlayer(),
                                    absl::MakeSpan(info_state_vector));
      for (int i = 0; i < info_state_vector.size(); ++i) {
        inputs_matrix(b, i) = info_state_vector[i];
      }
    }
  }
}

void TFBatchTrajectoryRecorder::ApplyActions() {
  std::vector<double> prob_dist(num_actions_, 0.0);
  auto sampled_action = tf_outputs_[1].matrix<tensorflow::int64>();
  for (int b = 0; b < batch_size_; ++b) {
    if (!terminal_flags_[b]) {
      Action action = sampled_action(b);
      SPIEL_CHECK_GE(action, 0);
      SPIEL_CHECK_LT(action, num_actions_);
      SPIEL_CHECK_EQ(tf_legal_mask_.matrix<float>()(b, action), 1);
      states_[b]->ApplyAction(action);
      SampleChance(b);
    }
  }
}

void TFBatchTrajectoryRecorder::RunInference() {
  TF_CHECK_OK(tf_session_->Run(
      {{"input", tf_inputs_}, {"legals_mask", tf_legal_mask_}},
      {"policy_softmax", "sampled_actions/Multinomial"}, {}, &tf_outputs_));
}

void TFBatchTrajectoryRecorder::GetNextStatesTF() {
  FillInputsAndMasks();
  RunInference();
  ApplyActions();
}

void TFBatchTrajectoryRecorder::Record() {
  int steps = 0;
  Reset();
  while (num_terminals_ < batch_size_) {
    steps++;
    GetNextStatesTF();
  }
}

}  // namespace algorithms
}  // namespace open_spiel
