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
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "open_spiel/abseil-cpp/absl/container/flat_hash_map.h"
#include "open_spiel/abseil-cpp/absl/strings/str_format.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/utils/file.h"

namespace open_spiel {
namespace algorithms {
namespace {

double SolveState(
    const State& state,
    absl::flat_hash_map<std::string, int>& cache,
    std::vector<VPNetModel::TrainInputs>& train_inputs) {
  std::string state_str = state.ToString();
  if (cache.find(state_str) != cache.end()) {
    return train_inputs[cache[state_str]].value;
  }
  if (state.IsTerminal()) {
    return state.PlayerReturn(0);
  }

  bool max_player = state.CurrentPlayer() == 0;
  std::vector<float> obs = state.ObservationTensor();
  std::vector<Action> legal_actions = state.LegalActions();

  Action best_action = kInvalidAction;
  double best_value = -2;
  for (Action action : legal_actions) {
    double value = SolveState(*state.Child(action), cache, train_inputs);
    if (best_action == kInvalidAction ||
        (max_player ? value > best_value : value < best_value)) {
      best_action = action;
      best_value = value;
    }
  }
  ActionsAndProbs policy({{best_action, 1}});

  cache[state_str] = train_inputs.size();
  train_inputs.push_back(VPNetModel::TrainInputs{
    legal_actions, obs, policy, best_value});
  return best_value;
}

std::vector<VPNetModel::TrainInputs> SolveGame() {
  std::shared_ptr<const open_spiel::Game> game =
      open_spiel::LoadGame("tic_tac_toe");
  std::unique_ptr<open_spiel::State> state = game->NewInitialState();

  // Store them directly into a vector so they are returned in order so
  // given a static initialization the model trains identically.
  absl::flat_hash_map<std::string, int> cache;
  std::vector<VPNetModel::TrainInputs> train_inputs;
  train_inputs.reserve(4520);
  SolveState(*state, cache, train_inputs);
  return train_inputs;
}

VPNetModel BuildModel(const Game& game, const std::string& nn_model,
                       bool create_graph) {
  std::string tmp_dir = open_spiel::file::GetTmpDir();
  std::string filename = absl::StrCat(
      "open_spiel_vpnet_test_", nn_model, ".pb");

  if (create_graph) {
    SPIEL_CHECK_TRUE(CreateGraphDef(
        game,
        /*learning_rate=*/0.01,
        /*weight_decay=*/0.0001,
        tmp_dir, filename,
        nn_model, /*nn_width=*/32, /*nn_depth=*/2, /*verbose=*/true));
  }

  std::string model_path = absl::StrCat(tmp_dir, "/", filename);
  SPIEL_CHECK_TRUE(file::Exists(model_path));

  VPNetModel model(game, tmp_dir, filename, "/cpu:0");

  return model;
}

void TestModelCreation(const std::string& nn_model) {
  std::cout << "TestModelCreation: " << nn_model << std::endl;
  std::shared_ptr<const Game> game = LoadGame("tic_tac_toe");
  VPNetModel model = BuildModel(*game, nn_model, true);

  std::unique_ptr<open_spiel::State> state = game->NewInitialState();
  std::vector<Action> legal_actions = state->LegalActions();
  std::vector<float> obs = state->ObservationTensor();
  VPNetModel::InferenceInputs inputs = {legal_actions, obs};

  // Check that inference runs at all.
  model.Inference(std::vector{inputs});

  std::vector<VPNetModel::TrainInputs> train_inputs;
  train_inputs.emplace_back(VPNetModel::TrainInputs{
      legal_actions, obs, ActionsAndProbs({{legal_actions[0], 1}}), 0});

  // Check that learning runs at all.
  model.Learn(train_inputs);
}

// Can learn a single trajectory
void TestModelLearnsSimple(const std::string& nn_model) {
  std::cout << "TestModelLearnsSimple: " << nn_model << std::endl;
  std::shared_ptr<const Game> game = LoadGame("tic_tac_toe");
  VPNetModel model = BuildModel(*game, nn_model, false);

  std::vector<VPNetModel::TrainInputs> train_inputs;
  std::unique_ptr<open_spiel::State> state = game->NewInitialState();

  while (!state->IsTerminal()) {
    std::vector<float> obs = state->ObservationTensor();
    std::vector<Action> legal_actions = state->LegalActions();
    Action action = legal_actions[0];
    ActionsAndProbs policy({{action, 1}});

    train_inputs.emplace_back(VPNetModel::TrainInputs{
        legal_actions, obs, policy, 1});

    VPNetModel::InferenceInputs inputs = {legal_actions, obs};
    std::vector<VPNetModel::InferenceOutputs> out =
        model.Inference(std::vector{inputs});
    SPIEL_CHECK_EQ(out.size(), 1);
    SPIEL_CHECK_EQ(out[0].policy.size(), legal_actions.size());

    state->ApplyAction(action);
  }

  std::cout << "states: " << train_inputs.size() << std::endl;
  std::vector<VPNetModel::LossInfo> losses;
  const double policy_loss_goal = 0.05;
  const double value_loss_goal = 0.05;
  for (int i = 0; i < 200; i++) {
    VPNetModel::LossInfo loss = model.Learn(train_inputs);
    std::cout << absl::StrFormat(
        "%d: Losses(total: %.3f, policy: %.3f, value: %.3f, l2: %.3f)\n",
         i, loss.Total(), loss.Policy(), loss.Value(), loss.L2());
    losses.push_back(loss);
    if (loss.Policy() < policy_loss_goal && loss.Value() < value_loss_goal) {
      break;
    }
  }
  SPIEL_CHECK_GT(losses.front().Total(), losses.back().Total());
  SPIEL_CHECK_GT(losses.front().Policy(), losses.back().Policy());
  SPIEL_CHECK_GT(losses.front().Value(), losses.back().Value());
  SPIEL_CHECK_LT(losses.back().Value(), value_loss_goal);
  SPIEL_CHECK_LT(losses.back().Policy(), policy_loss_goal);
}

// Can learn the optimal policy.
void TestModelLearnsOptimal(
    const std::string& nn_model,
    const std::vector<VPNetModel::TrainInputs>& train_inputs) {
  std::cout << "TestModelLearnsOptimal: " << nn_model << std::endl;
  std::shared_ptr<const Game> game = LoadGame("tic_tac_toe");
  VPNetModel model = BuildModel(*game, nn_model, false);

  std::cout << "states: " << train_inputs.size() << std::endl;
  std::vector<VPNetModel::LossInfo> losses;
  const double policy_loss_goal = 0.1;
  const double value_loss_goal = 0.1;
  for (int i = 0; i < 500; i++) {
    VPNetModel::LossInfo loss = model.Learn(train_inputs);
    std::cout << absl::StrFormat(
        "%d: Losses(total: %.3f, policy: %.3f, value: %.3f, l2: %.3f)\n",
         i, loss.Total(), loss.Policy(), loss.Value(), loss.L2());
    losses.push_back(loss);
    if (loss.Policy() < policy_loss_goal && loss.Value() < value_loss_goal) {
      break;
    }
  }
  SPIEL_CHECK_GT(losses.front().Total(), losses.back().Total());
  SPIEL_CHECK_GT(losses.front().Policy(), losses.back().Policy());
  SPIEL_CHECK_GT(losses.front().Value(), losses.back().Value());
  SPIEL_CHECK_LT(losses.back().Value(), value_loss_goal);
  SPIEL_CHECK_LT(losses.back().Policy(), policy_loss_goal);
}

}  // namespace
}  // namespace algorithms
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::algorithms::TestModelCreation("mlp");
  open_spiel::algorithms::TestModelCreation("conv2d");
  open_spiel::algorithms::TestModelCreation("resnet");

  // Tests below here reuse the graphs created above. Graph creation is slow
  // due to calling a separate python process.

  open_spiel::algorithms::TestModelLearnsSimple("mlp");
  open_spiel::algorithms::TestModelLearnsSimple("conv2d");
  open_spiel::algorithms::TestModelLearnsSimple("resnet");

  auto train_inputs = open_spiel::algorithms::SolveGame();
  open_spiel::algorithms::TestModelLearnsOptimal("mlp", train_inputs);
  open_spiel::algorithms::TestModelLearnsOptimal("conv2d", train_inputs);
  open_spiel::algorithms::TestModelLearnsOptimal("resnet", train_inputs);
}
