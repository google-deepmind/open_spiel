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

#include "open_spiel/tensor_game.h"

#include <algorithm>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/abseil-cpp/absl/strings/str_join.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace tensor_game {
namespace {
// Check the utilities to see if the game is constant-sum or identical
// (cooperative).
GameType::Utility GetUtilityType(
    const std::vector<std::vector<double>>& utils) {
  double util_sum = 0;
  // Assume both are true until proven otherwise.
  bool constant_sum = true;
  bool identical = true;
  for (int i = 0; i < utils[0].size(); ++i) {
    double util_sum_i = 0;
    for (int player = 0; player < utils.size(); ++player) {
      util_sum_i += utils[player][i];
    }

    if (i == 0) {
      util_sum = util_sum_i;
    } else {
      if (constant_sum && !Near(util_sum_i, util_sum)) {
        constant_sum = false;
      }
    }

    if (identical) {
      for (int player = 1; player < utils.size(); ++player) {
        if (utils[0][i] != utils[player][i]) {
          identical = false;
          break;
        }
      }
    }
  }

  if (constant_sum && Near(util_sum, 0.0)) {
    return GameType::Utility::kZeroSum;
  } else if (constant_sum) {
    return GameType::Utility::kConstantSum;
  } else if (identical) {
    return GameType::Utility::kIdentical;
  } else {
    return GameType::Utility::kGeneralSum;
  }
}
}  // namespace

TensorState::TensorState(std::shared_ptr<const Game> game)
    : NFGState(game),
      tensor_game_(static_cast<const TensorGame*>(game.get())) {}

std::string TensorState::ToString() const {
  std::string result = "";
  absl::StrAppend(&result, "Terminal? ", IsTerminal() ? "true" : "false", "\n");
  if (IsTerminal()) {
    absl::StrAppend(&result, "History: ", HistoryString(), "\n");
    absl::StrAppend(&result, "Returns: ", absl::StrJoin(Returns(), ","), "\n");
  }

  return result;
}

std::unique_ptr<State> TensorGame::NewInitialState() const {
  return std::unique_ptr<State>(new TensorState(shared_from_this()));
}

std::shared_ptr<const TensorGame> CreateTensorGame(
    const std::vector<std::vector<double>>& utils,
    const std::vector<int>& shape) {
  std::vector<std::vector<std::string>> action_names(shape.size());
  for (Player player = 0; player < shape.size(); ++player) {
    for (int i = 0; i < shape[player]; ++i) {
      action_names[player].push_back(absl::StrCat("action", player, "_", i));
    }
  }
  return CreateTensorGame("short_name", "Long Name", action_names, utils);
}

// Create a matrix game with the specified utilities and row/column names.
// Utilities must be in row-major form.

std::shared_ptr<const TensorGame> CreateTensorGame(
    const std::string& short_name, const std::string& long_name,
    const std::vector<std::vector<std::string>>& action_names,
    const std::vector<std::vector<double>>& utils) {
  const int size =
      std::accumulate(action_names.begin(), action_names.end(), 1,
                      [](const int s, auto names) { return s * names.size(); });
  SPIEL_CHECK_TRUE(
      std::all_of(utils.begin(), utils.end(), [size](const auto& player_utils) {
        return player_utils.size() == size;
      }));

  // Detect the utility type from the utilities.
  const GameType::Utility utility = GetUtilityType(utils);

  const GameType game_type{
      /*short_name=*/short_name,
      /*long_name=*/long_name,
      GameType::Dynamics::kSimultaneous,
      GameType::ChanceMode::kDeterministic,
      GameType::Information::kOneShot,
      utility,
      GameType::RewardModel::kTerminal,
      /*max_num_players=*/static_cast<int>(utils.size()),
      /*min_num_players=*/static_cast<int>(utils.size()),
      /*provides_information_state_string=*/true,
      /*provides_information_state_tensor=*/true,
      /*provides_observation_string=*/false,
      /*provides_observation_tensor=*/false,
      /*parameter_specification=*/{}  // no parameters
  };

  return std::shared_ptr<const TensorGame>(
      new TensorGame(game_type, {}, action_names, utils));
}

}  // namespace tensor_game
}  // namespace open_spiel
