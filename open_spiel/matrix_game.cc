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

#include "open_spiel/matrix_game.h"

#include <memory>
#include <string>
#include <vector>

#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/abseil-cpp/absl/strings/str_join.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace matrix_game {
namespace {
// Check the utilities to see if the game is constant-sum or identical
// (cooperative).
GameType::Utility GetUtilityType(const std::vector<double>& row_player_utils,
                                 const std::vector<double>& col_player_utils) {
  double util_sum = 0;
  // Assume both are true until proven otherwise.
  bool constant_sum = true;
  bool identical = true;
  for (int i = 0; i < row_player_utils.size(); ++i) {
    if (i == 0) {
      util_sum = row_player_utils[i] + col_player_utils[i];
    } else {
      if (constant_sum &&
          !Near(row_player_utils[i] + col_player_utils[i], util_sum)) {
        constant_sum = false;
      }
    }

    if (identical && row_player_utils[i] != col_player_utils[i]) {
      identical = false;
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

MatrixState::MatrixState(std::shared_ptr<const Game> game)
    : NFGState(game),
      matrix_game_(static_cast<const MatrixGame*>(game.get())) {}

std::string MatrixState::ToString() const {
  std::string result = "";
  absl::StrAppend(&result, "Terminal? ", IsTerminal() ? "true" : "false", "\n");
  if (IsTerminal()) {
    absl::StrAppend(&result, "History: ", HistoryString(), "\n");
    absl::StrAppend(&result, "Returns: ", absl::StrJoin(Returns(), ","), "\n");
  }
  absl::StrAppend(&result, "Row actions: ");
  for (auto move : LegalActions(0)) {
    absl::StrAppend(&result, ActionToString(0, move), " ");
  }
  absl::StrAppend(&result, "\nCol actions: ");
  for (auto move : LegalActions(1)) {
    absl::StrAppend(&result, ActionToString(1, move), " ");
  }
  absl::StrAppend(&result, "\nUtility matrix:\n");
  for (int r = 0; r < matrix_game_->NumRows(); r++) {
    for (int c = 0; c < matrix_game_->NumCols(); c++) {
      absl::StrAppend(&result, matrix_game_->RowUtility(r, c), ",",
                      matrix_game_->ColUtility(r, c), " ");
    }
    absl::StrAppend(&result, "\n");
  }
  return result;
}

std::unique_ptr<State> MatrixGame::NewInitialState() const {
  return std::unique_ptr<State>(new MatrixState(shared_from_this()));
}

std::vector<double> FlattenMatrix(
    const std::vector<std::vector<double>>& matrix_rows) {
  std::vector<double> utilities;
  int total_size = 0;
  int row_size = -1;
  int i = 0;

  for (int r = 0; r < matrix_rows.size(); ++r) {
    if (row_size < 0) {
      row_size = matrix_rows[r].size();
    }
    SPIEL_CHECK_GT(row_size, 0);
    SPIEL_CHECK_EQ(row_size, matrix_rows[r].size());
    total_size += row_size;
    utilities.resize(total_size, 0);

    for (int c = 0; c < matrix_rows[r].size(); ++c) {
      utilities[i] = matrix_rows[r][c];
      ++i;
    }
  }

  return utilities;
}

std::shared_ptr<const MatrixGame> CreateMatrixGame(
    const std::vector<std::vector<double>>& row_player_utils,
    const std::vector<std::vector<double>>& col_player_utils) {
  SPIEL_CHECK_GT(row_player_utils.size(), 0);
  int num_rows = row_player_utils.size();
  int num_columns = row_player_utils[0].size();
  std::vector<std::string> row_names(num_rows);
  std::vector<std::string> col_names(num_columns);
  for (int i = 0; i < num_rows; ++i) {
    row_names[i] = absl::StrCat("row", i);
  }
  for (int i = 0; i < num_columns; ++i) {
    col_names[i] = absl::StrCat("col", i);
  }
  return CreateMatrixGame("short_name", "Long Name", row_names, col_names,
                          row_player_utils, col_player_utils);
}

// Create a matrix game with the specified utilities and row/column names.
// Utilities must be in row-major form.
std::shared_ptr<const MatrixGame> CreateMatrixGame(
    const std::string& short_name, const std::string& long_name,
    const std::vector<std::string>& row_names,
    const std::vector<std::string>& col_names,
    const std::vector<std::vector<double>>& row_player_utils,
    const std::vector<std::vector<double>>& col_player_utils) {
  int rows = row_names.size();
  int columns = col_names.size();
  std::vector<double> flat_row_utils = FlattenMatrix(row_player_utils);
  std::vector<double> flat_col_utils = FlattenMatrix(col_player_utils);
  SPIEL_CHECK_EQ(flat_row_utils.size(), rows * columns);
  SPIEL_CHECK_EQ(flat_col_utils.size(), rows * columns);
  return CreateMatrixGame(short_name, long_name, row_names, col_names,
                          flat_row_utils, flat_col_utils);
}

std::shared_ptr<const MatrixGame> CreateMatrixGame(
    const std::string& short_name, const std::string& long_name,
    const std::vector<std::string>& row_names,
    const std::vector<std::string>& col_names,
    const std::vector<double>& flat_row_utils,
    const std::vector<double>& flat_col_utils) {
  // Detect the utility type from the utilities.
  GameType::Utility utility = GetUtilityType(flat_row_utils, flat_col_utils);

  GameType game_type{
      /*short_name=*/short_name,
      /*long_name=*/long_name,
      GameType::Dynamics::kSimultaneous,
      GameType::ChanceMode::kDeterministic,
      GameType::Information::kOneShot,
      utility,
      GameType::RewardModel::kTerminal,
      /*max_num_players=*/2,
      /*min_num_players=*/2,
      /*provides_information_state_string=*/true,
      /*provides_information_state_tensor=*/true,
      /*provides_observation_string=*/true,
      /*provides_observation_tensor=*/true,
      /*parameter_specification=*/{}  // no parameters
  };

  return std::shared_ptr<const MatrixGame>(new MatrixGame(
      game_type, {}, row_names, col_names, flat_row_utils, flat_col_utils));
}

}  // namespace matrix_game
}  // namespace open_spiel
