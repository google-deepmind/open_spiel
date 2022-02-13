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

#include "open_spiel/algorithms/alpha_zero_torch/model.h"

#include <torch/torch.h>

#include <iostream>
#include <string>
#include <vector>

#include "open_spiel/abseil-cpp/absl/strings/str_format.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace algorithms {
namespace torch_az {
namespace {

void TestModelCreation() {
  std::cout << "\n~-~-~-~- TestModelCreation -~-~-~-~" << std::endl;

  std::shared_ptr<const Game> game = LoadGame("clobber");

  ModelConfig net_config = {
      /*observation_tensor_shape=*/game->ObservationTensorShape(),
      /*number_of_actions=*/game->NumDistinctActions(),
      /*nn_depth=*/8,
      /*nn_width=*/128,
      /*learning_rate=*/0.001,
      /*weight_decay=*/0.001};
  Model net(net_config, "cpu:0");

  std::cout << "Good! The network looks like:\n" << net << std::endl;
}

void TestModelInference() {
  std::cout << "\n~-~-~-~- TestModelInference -~-~-~-~" << std::endl;

  const int channels = 3;
  const int rows = 8;
  const int columns = 8;
  std::string game_string =
      absl::StrCat("clobber(rows=", std::to_string(rows),
                   ",columns=", std::to_string(columns), ")");

  std::shared_ptr<const Game> game = LoadGame(game_string);
  std::unique_ptr<open_spiel::State> state = game->NewInitialState();

  ModelConfig net_config = {
      /*observation_tensor_shape=*/game->ObservationTensorShape(),
      /*number_of_actions=*/game->NumDistinctActions(),
      /*nn_depth=*/rows + 1,
      /*nn_width=*/128,
      /*learning_rate=*/0.001,
      /*weight_decay=*/0.001};
  Model net(net_config, "cpu:0");

  std::vector<float> observation_vector = state->ObservationTensor();
  torch::Tensor observation_tensor = torch::from_blob(
      observation_vector.data(), {1, channels * rows * columns});
  torch::Tensor mask = torch::full({1, game->NumDistinctActions()}, false,
                                   torch::TensorOptions().dtype(torch::kByte));

  for (Action action : state->LegalActions()) {
    mask[0][action] = true;
  }

  std::cout << "Input:\n"
            << observation_tensor.view({channels, rows, columns}) << std::endl;
  std::cout << "Mask:\n" << mask << std::endl;

  std::vector<torch::Tensor> output = net(observation_tensor, mask);

  std::cout << "Output:\n" << output << std::endl;

  // Check value and policy.
  SPIEL_CHECK_EQ((int)output.size(), 2);
  SPIEL_CHECK_EQ(output[0].numel(), 1);
  SPIEL_CHECK_EQ(output[1].numel(), game->NumDistinctActions());

  // Check mask's influence on the policy.
  for (int i = 0; i < game->NumDistinctActions(); i++) {
    if (mask[0][i].item<bool>()) {
      SPIEL_CHECK_GT(output[1][0][i].item<float>(), 0.0);
    } else {
      SPIEL_CHECK_EQ(output[1][0][i].item<float>(), 0.0);
    }
  }

  std::cout << "Value:\n" << output[0] << std::endl;
  std::cout << "Policy:\n" << output[1] << std::endl;
}

void TestCUDAAVailability() {
  if (torch::cuda::is_available()) {
    std::cout << "CUDA is available!" << std::endl;
  } else {
    std::cout << "CUDA is not available." << std::endl;
  }
}

}  // namespace
}  // namespace torch_az
}  // namespace algorithms
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::algorithms::torch_az::TestModelCreation();
  open_spiel::algorithms::torch_az::TestModelInference();
  open_spiel::algorithms::torch_az::TestCUDAAVailability();
}
