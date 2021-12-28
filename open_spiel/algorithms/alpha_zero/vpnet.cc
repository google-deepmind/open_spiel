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

#include "open_spiel/algorithms/alpha_zero/vpnet.h"

#include <algorithm>
#include <cstring>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/abseil-cpp/absl/strings/str_join.h"
#include "unsupported/Eigen/CXX11/Tensor"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/utils/file.h"
#include "open_spiel/utils/run_python.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/protobuf/saver.pb.h"

namespace open_spiel {
namespace algorithms {

namespace tf = tensorflow;
using Tensor = Eigen::Tensor<float, 2, Eigen::RowMajor>;
using TensorMap = Eigen::TensorMap<Tensor, Eigen::Aligned>;
using TensorBool = Eigen::Tensor<bool, 2, Eigen::RowMajor>;
using TensorMapBool = Eigen::TensorMap<TensorBool, Eigen::Aligned>;

bool CreateGraphDef(const Game& game, double learning_rate,
    double weight_decay, const std::string& path, const std::string& filename,
    std::string nn_model, int nn_width, int nn_depth, bool verbose) {
  return RunPython("open_spiel.python.algorithms.alpha_zero.export_model",
                   {
                       "--game", absl::StrCat("'", game.ToString(), "'"),  //
                       "--path", absl::StrCat("'", path, "'"),             //
                       "--graph_def", filename,                            //
                       "--learning_rate", absl::StrCat(learning_rate),     //
                       "--weight_decay", absl::StrCat(weight_decay),       //
                       "--nn_model", nn_model,                             //
                       "--nn_depth", absl::StrCat(nn_depth),               //
                       "--nn_width", absl::StrCat(nn_width),               //
                       absl::StrCat("--verbose=", verbose ? "true" : "false"),
                   });
}

VPNetModel::VPNetModel(const Game& game, const std::string& path,
                       const std::string& file_name, const std::string& device)
    : device_(device),
      path_(path),
      flat_input_size_(game.ObservationTensorSize()),
      num_actions_(game.NumDistinctActions()) {
  // Some assumptions that we can remove eventually. The value net returns
  // a single value in terms of player 0 and the game is assumed to be zero-sum,
  // so player 1 can just be -value.
  SPIEL_CHECK_EQ(game.NumPlayers(), 2);
  SPIEL_CHECK_EQ(game.GetType().utility, GameType::Utility::kZeroSum);

  std::string model_path = absl::StrCat(path, "/", file_name);
  model_meta_graph_contents_ = file::ReadContentsFromFile(model_path, "r");

  TF_CHECK_OK(
      ReadBinaryProto(tf::Env::Default(), model_path, &meta_graph_def_));

  tf::graph::SetDefaultDevice(device, meta_graph_def_.mutable_graph_def());

  if (tf_session_ != nullptr) {
    TF_CHECK_OK(tf_session_->Close());
  }

  // create a new session
  TF_CHECK_OK(NewSession(tf_opts_, &tf_session_));

  // Load graph into session
  TF_CHECK_OK(tf_session_->Create(meta_graph_def_.graph_def()));

  // Initialize our variables
  TF_CHECK_OK(tf_session_->Run({}, {}, {"init_all_vars_op"}, nullptr));
}

std::string VPNetModel::SaveCheckpoint(int step) {
  std::string full_path = absl::StrCat(path_, "/checkpoint-", step);
  tensorflow::Tensor checkpoint_path(tf::DT_STRING, tf::TensorShape());
  checkpoint_path.scalar<tensorflow::tstring>()() = full_path;
  TF_CHECK_OK(tf_session_->Run(
      {{meta_graph_def_.saver_def().filename_tensor_name(), checkpoint_path}},
      {}, {meta_graph_def_.saver_def().save_tensor_name()}, nullptr));
  // Writing a checkpoint from python writes the metagraph file, but c++
  // doesn't, so do it manually to make loading checkpoints easier.
  file::File(absl::StrCat(full_path, ".meta"), "w").Write(
      model_meta_graph_contents_);
  return full_path;
}

void VPNetModel::LoadCheckpoint(const std::string& path) {
  tf::Tensor checkpoint_path(tf::DT_STRING, tf::TensorShape());
  checkpoint_path.scalar<tensorflow::tstring>()() = path;
  TF_CHECK_OK(tf_session_->Run(
      {{meta_graph_def_.saver_def().filename_tensor_name(), checkpoint_path}},
      {}, {meta_graph_def_.saver_def().restore_op_name()}, nullptr));
}

std::vector<VPNetModel::InferenceOutputs> VPNetModel::Inference(
    const std::vector<InferenceInputs>& inputs) {
  int inference_batch_size = inputs.size();

  // Fill the inputs and mask
  tensorflow::Tensor tf_inf_inputs(
      tf::DT_FLOAT, tf::TensorShape({inference_batch_size, flat_input_size_}));
  tensorflow::Tensor tf_inf_legal_mask(
      tf::DT_BOOL, tf::TensorShape({inference_batch_size, num_actions_}));

  TensorMap inputs_matrix = tf_inf_inputs.matrix<float>();
  TensorMapBool mask_matrix = tf_inf_legal_mask.matrix<bool>();

  for (int b = 0; b < inference_batch_size; ++b) {
    // Zero initialize the sparse inputs.
    for (int a = 0; a < num_actions_; ++a) {
      mask_matrix(b, a) = 0;
    }
    for (Action action : inputs[b].legal_actions) {
      mask_matrix(b, action) = 1;
    }
    for (int i = 0; i < inputs[b].observations.size(); ++i) {
      inputs_matrix(b, i) = inputs[b].observations[i];
    }
  }

  // Run the inference
  std::vector<tensorflow::Tensor> tf_outputs;
  TF_CHECK_OK(tf_session_->Run(
      {{"input", tf_inf_inputs}, {"legals_mask", tf_inf_legal_mask},
       {"training", tensorflow::Tensor(false)}},
      {"policy_softmax", "value_out"}, {}, &tf_outputs));

  TensorMap policy_matrix = tf_outputs[0].matrix<float>();
  TensorMap value_matrix = tf_outputs[1].matrix<float>();

  std::vector<InferenceOutputs> out;
  out.reserve(inference_batch_size);
  for (int b = 0; b < inference_batch_size; ++b) {
    double value = value_matrix(b, 0);

    ActionsAndProbs state_policy;
    state_policy.reserve(inputs[b].legal_actions.size());
    for (Action action : inputs[b].legal_actions) {
      state_policy.push_back({action, policy_matrix(b, action)});
    }

    out.push_back({value, state_policy});
  }

  return out;
}

VPNetModel::LossInfo VPNetModel::Learn(const std::vector<TrainInputs>& inputs) {
  int training_batch_size = inputs.size();

  tensorflow::Tensor tf_train_inputs(
      tf::DT_FLOAT, tf::TensorShape({training_batch_size, flat_input_size_}));
  tensorflow::Tensor tf_train_legal_mask(
      tf::DT_BOOL, tf::TensorShape({training_batch_size, num_actions_}));
  tensorflow::Tensor tf_policy_targets(
      tf::DT_FLOAT, tf::TensorShape({training_batch_size, num_actions_}));
  tensorflow::Tensor tf_value_targets(
      tf::DT_FLOAT, tf::TensorShape({training_batch_size, 1}));

  // Fill the inputs and mask
  TensorMap inputs_matrix = tf_train_inputs.matrix<float>();
  TensorMapBool mask_matrix = tf_train_legal_mask.matrix<bool>();
  TensorMap policy_targets_matrix = tf_policy_targets.matrix<float>();
  TensorMap value_targets_matrix = tf_value_targets.matrix<float>();

  for (int b = 0; b < training_batch_size; ++b) {
    // Zero initialize the sparse inputs.
    for (int a = 0; a < num_actions_; ++a) {
      mask_matrix(b, a) = 0;
      policy_targets_matrix(b, a) = 0;
    }

    for (Action action : inputs[b].legal_actions) {
      mask_matrix(b, action) = 1;
    }

    for (int a = 0; a < inputs[b].observations.size(); ++a) {
      inputs_matrix(b, a) = inputs[b].observations[a];
    }

    for (const auto& [action, prob] : inputs[b].policy) {
      policy_targets_matrix(b, action) = prob;
    }

    value_targets_matrix(b, 0) = inputs[b].value;
  }

  // Run a training step and get the losses.
  std::vector<tensorflow::Tensor> tf_outputs;
  TF_CHECK_OK(tf_session_->Run({{"input", tf_train_inputs},
                                {"legals_mask", tf_train_legal_mask},
                                {"policy_targets", tf_policy_targets},
                                {"value_targets", tf_value_targets},
                                {"training", tensorflow::Tensor(true)}},
                               {"policy_loss", "value_loss", "l2_reg_loss"},
                               {"train"}, &tf_outputs));

  return LossInfo(
      tf_outputs[0].scalar<float>()(0),
      tf_outputs[1].scalar<float>()(0),
      tf_outputs[2].scalar<float>()(0));
}

}  // namespace algorithms
}  // namespace open_spiel
