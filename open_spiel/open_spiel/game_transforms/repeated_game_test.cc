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

#include "open_spiel/game_transforms/repeated_game.h"

#include <string>

#include "open_spiel/abseil-cpp/absl/algorithm/container.h"
#include "open_spiel/spiel.h"
#include "open_spiel/tests/basic_tests.h"

namespace open_spiel {
namespace {

void BasicRepeatedGameTest() {
  std::string game_string =
      "repeated_game(stage_game=matrix_rps(),num_repetitions=10)";
  open_spiel::testing::LoadGameTest(game_string);
  open_spiel::testing::NoChanceOutcomesTest(*LoadGame(game_string));
  open_spiel::testing::RandomSimTest(*LoadGame(game_string), 10);
  // Test loading from a pre-loaded stage game.
  std::shared_ptr<const Game> stage_game = LoadGame("matrix_rps");
  GameParameters params;
  params["num_repetitions"] = GameParameter(10);
  std::shared_ptr<const Game> repeated_game =
      CreateRepeatedGame(*stage_game, params);
  SPIEL_CHECK_TRUE(repeated_game != nullptr);
  // Test loading from a stage game string.
  repeated_game = CreateRepeatedGame("matrix_pd", params);
  SPIEL_CHECK_TRUE(repeated_game != nullptr);
}

void RepeatedRockPaperScissorsTest(std::shared_ptr<const Game> repeated_game) {
  std::unique_ptr<open_spiel::State> state = repeated_game->NewInitialState();
  SPIEL_CHECK_EQ(state->LegalActions(0), state->LegalActions(1));
  SPIEL_CHECK_EQ(state->ActionToString(0, 0), "Rock");
  SPIEL_CHECK_EQ(state->ActionToString(0, 1), "Paper");
  SPIEL_CHECK_EQ(state->ActionToString(0, 2), "Scissors");
  SPIEL_CHECK_EQ(state->ActionToString(1, 0), "Rock");
  SPIEL_CHECK_EQ(state->ActionToString(1, 1), "Paper");
  SPIEL_CHECK_EQ(state->ActionToString(1, 2), "Scissors");

  state->ApplyActions({0, 1});
  SPIEL_CHECK_EQ(state->PlayerReward(0), -1);
  SPIEL_CHECK_EQ(state->PlayerReward(1), 1);
  SPIEL_CHECK_EQ(state->ObservationString(), "Rock Paper ");
  SPIEL_CHECK_TRUE(absl::c_equal(state->ObservationTensor(0),
                                 std::vector<int>{1, 0, 0, 0, 1, 0}));
  state->ApplyActions({1, 0});
  SPIEL_CHECK_EQ(state->PlayerReward(0), 1);
  SPIEL_CHECK_EQ(state->PlayerReward(1), -1);
  SPIEL_CHECK_EQ(state->ObservationString(), "Paper Rock ");
  SPIEL_CHECK_TRUE(absl::c_equal(state->ObservationTensor(0),
                                 std::vector<int>{0, 1, 0, 1, 0, 0}));
  state->ApplyActions({2, 2});
  SPIEL_CHECK_EQ(state->PlayerReward(0), 0);
  SPIEL_CHECK_EQ(state->PlayerReward(1), 0);
  SPIEL_CHECK_EQ(state->ObservationString(), "Scissors Scissors ");
  SPIEL_CHECK_TRUE(absl::c_equal(state->ObservationTensor(0),
                                 std::vector<int>{0, 0, 1, 0, 0, 1}));
  SPIEL_CHECK_TRUE(state->IsTerminal());
}

void RepeatedRockPaperScissorsDefaultsTest() {
  GameParameters params;
  params["num_repetitions"] = GameParameter(3);
  std::shared_ptr<const Game> repeated_game =
      CreateRepeatedGame("matrix_rps", params);
  SPIEL_CHECK_EQ(repeated_game->GetType().max_num_players, 2);
  SPIEL_CHECK_EQ(repeated_game->GetType().min_num_players, 2);
  SPIEL_CHECK_EQ(repeated_game->GetType().utility, GameType::Utility::kZeroSum);
  SPIEL_CHECK_EQ(repeated_game->GetType().reward_model,
                 GameType::RewardModel::kRewards);
  SPIEL_CHECK_TRUE(repeated_game->GetType().provides_observation_tensor);
  SPIEL_CHECK_FALSE(repeated_game->GetType().provides_information_state_tensor);

  // One-hot encoding of each player's previous action.
  SPIEL_CHECK_EQ(repeated_game->ObservationTensorShape()[0], 6);

  RepeatedRockPaperScissorsTest(repeated_game);
}

void RepeatedRockPaperScissorsRecallTwoTest() {
  GameParameters params;
  params["num_repetitions"] = GameParameter(1000);
  params["recall"] = GameParameter(2);
  std::shared_ptr<const Game> repeated_game =
      CreateRepeatedGame("matrix_rps", params);
  SPIEL_CHECK_EQ(repeated_game->GetType().max_num_players, 2);
  SPIEL_CHECK_EQ(repeated_game->GetType().min_num_players, 2);
  SPIEL_CHECK_EQ(repeated_game->GetType().utility, GameType::Utility::kZeroSum);
  SPIEL_CHECK_EQ(repeated_game->GetType().reward_model,
                 GameType::RewardModel::kRewards);
  SPIEL_CHECK_TRUE(repeated_game->GetType().provides_observation_tensor);
  SPIEL_CHECK_FALSE(repeated_game->GetType().provides_information_state_tensor);

  // One-hot encoding of each player's previous action.
  SPIEL_CHECK_EQ(repeated_game->ObservationTensorShape()[0], 12);

  std::vector<std::vector<int>> observation_tensors = {
      {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},  // first
      {1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0},  // second
      {1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0}   // subsequent...
  };
  std::vector<std::string> observation_strings = {
      "",                          // first observation
      "Rock Rock ",                // second
      "Rock Rock Rock Rock "       // subsequent...
  };

  std::unique_ptr<State> state = repeated_game->NewInitialState();
  int step = 0;
  while (!state->IsTerminal()) {
    int obs_idx = std::min(step, 2);
    SPIEL_CHECK_EQ(state->ObservationString(0), observation_strings[obs_idx]);
    SPIEL_CHECK_EQ(state->ObservationString(1), observation_strings[obs_idx]);
    SPIEL_CHECK_TRUE(absl::c_equal(state->ObservationTensor(0),
                                   observation_tensors[obs_idx]));
    SPIEL_CHECK_TRUE(absl::c_equal(state->ObservationTensor(1),
                                   observation_tensors[obs_idx]));
    state->ApplyActions({0, 0});
    step += 1;
  }

  SPIEL_CHECK_EQ(step, 1000);
}

void RepeatedRockPaperScissorsInfoStateEnabledTest() {
  GameParameters params;
  params["num_repetitions"] = GameParameter(3);
  params["enable_infostate"] = GameParameter(true);
  std::shared_ptr<const Game> repeated_game =
      CreateRepeatedGame("matrix_rps", params);
  SPIEL_CHECK_EQ(repeated_game->GetType().max_num_players, 2);
  SPIEL_CHECK_EQ(repeated_game->GetType().min_num_players, 2);
  SPIEL_CHECK_EQ(repeated_game->GetType().utility, GameType::Utility::kZeroSum);
  SPIEL_CHECK_EQ(repeated_game->GetType().reward_model,
                 GameType::RewardModel::kRewards);
  SPIEL_CHECK_TRUE(repeated_game->GetType().provides_observation_tensor);
  SPIEL_CHECK_TRUE(repeated_game->GetType().provides_information_state_tensor);
  SPIEL_CHECK_TRUE(repeated_game->GetType().provides_information_state_string);

  // One-hot encoding of each player's previous action.
  SPIEL_CHECK_EQ(repeated_game->ObservationTensorShape()[0], 6);

  // One-hot encoding of each player's previous action times num_repetitions.
  SPIEL_CHECK_EQ(repeated_game->InformationStateTensorShape()[0], 18);

  // Check information_state_string
  std::unique_ptr<State> state = repeated_game->NewInitialState();
  SPIEL_CHECK_EQ(state->InformationStateString(), "");
  state->ApplyActions({0, 0});
  SPIEL_CHECK_EQ(state->InformationStateString(), "Rock Rock ;");
  state->ApplyActions({1, 2});
  SPIEL_CHECK_EQ(state->InformationStateString(),
                 "Rock Rock ;Paper Scissors ;");

  RepeatedRockPaperScissorsTest(repeated_game);
}


void RepeatedPrisonersDilemaTest() {
  GameParameters params;
  params["num_repetitions"] = GameParameter(2);
  std::shared_ptr<const Game> repeated_game =
      CreateRepeatedGame("matrix_pd", params);
  SPIEL_CHECK_EQ(repeated_game->GetType().max_num_players, 2);
  SPIEL_CHECK_EQ(repeated_game->GetType().min_num_players, 2);
  SPIEL_CHECK_EQ(repeated_game->GetType().utility,
                 GameType::Utility::kGeneralSum);
  // repeated_game->UtilitySum() should raise an error. This is checked in
  // game_transforms_test.py as it's simpler to catch the error from Python.
  SPIEL_CHECK_EQ(repeated_game->GetType().reward_model,
                 GameType::RewardModel::kRewards);

  std::unique_ptr<open_spiel::State> state = repeated_game->NewInitialState();
  SPIEL_CHECK_EQ(state->LegalActions(0), state->LegalActions(1));
  SPIEL_CHECK_EQ(state->ActionToString(0, 0), "Cooperate");
  SPIEL_CHECK_EQ(state->ActionToString(0, 1), "Defect");
  SPIEL_CHECK_EQ(state->ActionToString(1, 0), "Cooperate");
  SPIEL_CHECK_EQ(state->ActionToString(1, 1), "Defect");

  state->ApplyActions({0, 1});
  SPIEL_CHECK_EQ(state->PlayerReward(0), 0);
  SPIEL_CHECK_EQ(state->PlayerReward(1), 10);
  SPIEL_CHECK_EQ(state->ObservationString(), "Cooperate Defect ");
  SPIEL_CHECK_TRUE(
      absl::c_equal(state->ObservationTensor(0), std::vector<int>{1, 0, 0, 1}));
  state->ApplyActions({1, 0});
  SPIEL_CHECK_EQ(state->PlayerReward(0), 10);
  SPIEL_CHECK_EQ(state->PlayerReward(1), 0);
  SPIEL_CHECK_EQ(state->ObservationString(), "Defect Cooperate ");
  SPIEL_CHECK_TRUE(
      absl::c_equal(state->ObservationTensor(1), std::vector<int>{0, 1, 1, 0}));
  SPIEL_CHECK_TRUE(state->IsTerminal());
}

}  // namespace
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::BasicRepeatedGameTest();
  open_spiel::RepeatedRockPaperScissorsDefaultsTest();
  open_spiel::RepeatedRockPaperScissorsRecallTwoTest();
  open_spiel::RepeatedRockPaperScissorsInfoStateEnabledTest();
  open_spiel::RepeatedPrisonersDilemaTest();
}
