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

#ifndef OPEN_SPIEL_ALGORITHMS_DQN_TORCH_SIMPLE_NETS_H_
#define OPEN_SPIEL_ALGORITHMS_DQN_TORCH_SIMPLE_NETS_H_

#include <torch/torch.h>

#include <iostream>
#include <vector>
#include <utility>

#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace algorithms {
namespace torch_dqn {

// Always includes biases and only supports ReLU activations.
class SimpleLinearImpl : public torch::nn::Module {
 public :
  SimpleLinearImpl(int input_size, int output_size, bool activate_relu = true);
  torch::Tensor forward(torch::Tensor x);

 private:
  bool activate_relu_;
  torch::nn::Linear simple_linear_;
};
TORCH_MODULE(SimpleLinear);

// A simple dense network built from linear layers above.
class MLPImpl : public torch::nn::Module {
 public:
  MLPImpl(int input_size, const std::vector<int>& hidden_layers_sizes,
          int output_size, bool activate_final = false);
  torch::Tensor forward(torch::Tensor x);

 private:
  int input_size_;
  std::vector<int> hidden_layers_sizes_;
  int output_size_;
  bool activate_final_;
  torch::nn::ModuleList layers_;
};
TORCH_MODULE(MLP);

}  // namespace torch_dqn
}  // namespace algorithms
}  // namespace open_spiel

#endif  // OPEN_SPIEL_ALGORITHMS_DQN_TORCH_SIMPLE_NETS_H_
