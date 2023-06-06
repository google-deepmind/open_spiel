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

#ifndef OPEN_SPIEL_ALGORITHMS_ALPHA_ZERO_VPNET_H_
#define OPEN_SPIEL_ALGORITHMS_ALPHA_ZERO_VPNET_H_

#include "open_spiel/spiel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow/core/public/session.h"

namespace open_spiel {
namespace algorithms {

// Spawn a python interpreter to call export_model.py.
// There are three options for nn_model: mlp, conv2d and resnet.
// The nn_width is the number of hidden units for the mlp, and filters for
// conv/resnet. The nn_depth is number of layers for all three.
bool CreateGraphDef(
    const Game& game, double learning_rate,
    double weight_decay, const std::string& path, const std::string& filename,
    std::string nn_model, int nn_width, int nn_depth, bool verbose = false);


class VPNetModel {
  // TODO(author7): Save and restore checkpoints:
  // https://stackoverflow.com/questions/37508771/how-to-save-and-restore-a-tensorflow-graph-and-its-state-in-c
  // https://stackoverflow.com/questions/35508866/tensorflow-different-ways-to-export-and-run-graph-in-c/43639305#43639305
  // https://www.tensorflow.org/api_docs/python/tf/compat/v1/train/Saver

 public:
  class LossInfo {
   public:
    LossInfo() {}
    LossInfo(double policy, double value, double l2) :
      policy_(policy), value_(value), l2_(l2), batches_(1) {}

    // Merge another LossInfo into this one.
    LossInfo& operator+=(const LossInfo& other) {
      policy_ += other.policy_;
      value_ += other.value_;
      l2_ += other.l2_;
      batches_ += other.batches_;
      return *this;
    }

    // Return the average losses over all merged into this one.
    double Policy() const { return policy_ / batches_; }
    double Value() const { return value_ / batches_; }
    double L2() const { return l2_ / batches_; }
    double Total() const { return Policy() + Value() + L2(); }

   private:
    double policy_ = 0;
    double value_ = 0;
    double l2_ = 0;
    int batches_ = 0;
  };

  struct InferenceInputs {
    std::vector<Action> legal_actions;
    std::vector<float> observations;

    bool operator==(const InferenceInputs& o) const {
      return legal_actions == o.legal_actions && observations == o.observations;
    }

    template <typename H>
    friend H AbslHashValue(H h, const InferenceInputs& in) {
      return H::combine(std::move(h), in.legal_actions, in.observations);
    }
  };
  struct InferenceOutputs {
    double value;
    ActionsAndProbs policy;
  };

  struct TrainInputs {
    std::vector<Action> legal_actions;
    std::vector<float> observations;
    ActionsAndProbs policy;
    double value;
  };

  VPNetModel(const Game& game, const std::string& path,
             const std::string& file_name,
             const std::string& device = "/cpu:0");

  // Move only, not copyable.
  VPNetModel(VPNetModel&& other) = default;
  VPNetModel& operator=(VPNetModel&& other) = default;
  VPNetModel(const VPNetModel&) = delete;
  VPNetModel& operator=(const VPNetModel&) = delete;

  // Inference: Get both at the same time.
  std::vector<InferenceOutputs> Inference(
    const std::vector<InferenceInputs>& inputs);

  // Training: do one (batch) step of neural net training
  LossInfo Learn(const std::vector<TrainInputs>& inputs);

  std::string SaveCheckpoint(int step);
  void LoadCheckpoint(const std::string& path);

  const std::string Device() const { return device_; }

 private:
  std::string device_;
  std::string path_;

  // Store the full model metagraph file for writing python compatible
  // checkpoints.
  std::string model_meta_graph_contents_;

  int flat_input_size_;
  int num_actions_;

  // Inputs for inference & training separated to have different fixed sizes
  tensorflow::Session* tf_session_ = nullptr;
  tensorflow::MetaGraphDef meta_graph_def_;
  tensorflow::SessionOptions tf_opts_;
};

}  // namespace algorithms
}  // namespace open_spiel

#endif  // OPEN_SPIEL_ALGORITHMS_ALPHA_ZERO_VPNET_H_
