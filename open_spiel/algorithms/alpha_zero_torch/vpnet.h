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

#ifndef OPEN_SPIEL_ALGORITHMS_ALPHA_ZERO_TORCH_VPNET_H_
#define OPEN_SPIEL_ALGORITHMS_ALPHA_ZERO_TORCH_VPNET_H_

#include <torch/torch.h>

#include <nop/structure.h>

#include "open_spiel/algorithms/alpha_zero_torch/model.h"
#include "open_spiel/spiel.h"

namespace open_spiel {
namespace algorithms {
namespace torch_az {

// To avoid having to change function calls and the flow of the AlphaZero setup,
// this function is still called, but rather than create a TensorFlow graph
// definition, it simply creates a struct that the Libtorch model can use to
// initialize from. This struct is then saved and then loaded again when needed.
bool CreateGraphDef(const Game& game, double learning_rate, double weight_decay,
                    const std::string& path, const std::string& filename,
                    std::string nn_model, int nn_width, int nn_depth,
                    bool verbose = false);

class VPNetModel {
 public:
  // A class to handle the network's loss.
  class LossInfo {
   public:
    LossInfo() {}
    LossInfo(double policy, double value, double l2)
        : policy_(policy), value_(value), l2_(l2), batches_(1) {}

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

  // A struct to handle the inputs for inference.
  struct InferenceInputs {
    std::vector<Action> legal_actions;
    std::vector<float> observations;

    bool operator==(const InferenceInputs& other) const {
      return legal_actions == other.legal_actions &&
             observations == other.observations;
    }

    template <typename H>
    friend H AbslHashValue(H h, const InferenceInputs& in) {
      return H::combine(std::move(h), in.legal_actions, in.observations);
    }
  };

  // A struct to hold the outputs of the inference (value and policy).
  struct InferenceOutputs {
    double value;
    ActionsAndProbs policy;
  };

  // A struct to hold the inputs for training.
  struct TrainInputs {
    std::vector<Action> legal_actions;
    std::vector<float> observations;
    ActionsAndProbs policy;
    double value;

    NOP_STRUCTURE(TrainInputs, legal_actions, observations, policy, value);
  };

  enum CheckpointStep {
    kMostRecentCheckpointStep = -1,
    kInvalidCheckpointStep = -2
  };

  VPNetModel(const Game &game, const std::string &path,
             const std::string &file_name,
             const std::string &device = "/cpu:0");

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
  void LoadCheckpoint(int step);
  void LoadCheckpoint(const std::string& path);

  const std::string Device() const { return device_; }

 private:
  std::string device_;
  std::string path_;

  // Store the full model metagraph file
  // for writing python compatible checkpoints.
  std::string model_meta_graph_contents_;

  int flat_input_size_;
  int num_actions_;

  // NOTE:
  // The member model_ takes an already initialized model_config_,
  // and model_optimizer_ takes an already initialized model_
  // parameters and model_config_ learning rate. Therefore, keep the
  // members' (model_config_, model_, model_optimizer_) declaration in
  // the order shown below so the member initialization list works.
  ModelConfig model_config_;
  Model model_;
  torch::optim::Adam model_optimizer_;
  torch::Device torch_device_;
};

}  // namespace torch_az
}  // namespace algorithms
}  // namespace open_spiel

#endif  // OPEN_SPIEL_ALGORITHMS_ALPHA_ZERO_TORCH_VPNET_H_
