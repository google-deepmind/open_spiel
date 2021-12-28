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

#ifndef OPEN_SPIEL_MATRIX_GAME_H_
#define OPEN_SPIEL_MATRIX_GAME_H_

#include <algorithm>
#include <iterator>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include "open_spiel/normal_form_game.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

// A matrix game is an example of a 2-player normal-form game.

namespace open_spiel {
namespace matrix_game {

inline constexpr int kRowPlayer = 0;
inline constexpr int kColPlayer = 1;

// Return a flattened version of these vector of rows. This simply scans each
// row in turn, appending each elements onto the end of a 1D vector. The rows
// must have the same size.
std::vector<double> FlattenMatrix(
    const std::vector<std::vector<double>>& matrix_rows);

class MatrixGame : public NormalFormGame {
 public:
  MatrixGame(GameType game_type, GameParameters game_parameters,
             std::vector<std::string> row_action_names,
             std::vector<std::string> col_action_names,
             std::vector<double> row_utilities,
             std::vector<double> col_utilities)
      : NormalFormGame(game_type, game_parameters),
        row_action_names_(row_action_names),
        col_action_names_(col_action_names),
        row_utilities_(row_utilities),
        col_utilities_(col_utilities) {}

  MatrixGame(GameType game_type, GameParameters game_parameters,
             std::vector<std::string> row_action_names,
             std::vector<std::string> col_action_names,
             const std::vector<std::vector<double>> row_utilities,
             const std::vector<std::vector<double>> col_utilities)
      : NormalFormGame(game_type, game_parameters),
        row_action_names_(row_action_names),
        col_action_names_(col_action_names),
        row_utilities_(FlattenMatrix(row_utilities)),
        col_utilities_(FlattenMatrix(col_utilities)) {}

  // Implementation of Game interface
  int NumDistinctActions() const override {
    return std::max(NumRows(), NumCols());
  }

  std::unique_ptr<State> NewInitialState() const override;

  int NumPlayers() const override { return 2; }

  double MinUtility() const override {
    return std::min(
        *std::min_element(begin(row_utilities_), end(row_utilities_)),
        *std::min_element(begin(col_utilities_), end(col_utilities_)));
  }

  double MaxUtility() const override {
    return std::max(
        *std::max_element(begin(row_utilities_), end(row_utilities_)),
        *std::max_element(begin(col_utilities_), end(col_utilities_)));
  }

  // Methods for MatrixState to call.
  int NumRows() const { return row_action_names_.size(); }
  int NumCols() const { return col_action_names_.size(); }
  double RowUtility(int row, int col) const {
    return row_utilities_[Index(row, col)];
  }
  double ColUtility(int row, int col) const {
    return col_utilities_[Index(row, col)];
  }
  double PlayerUtility(Player player, int row, int col) const {
    SPIEL_CHECK_TRUE(player == Player{0} || player == Player{1});
    return (player == Player{0} ? row_utilities_[Index(row, col)]
                                : col_utilities_[Index(row, col)]);
  }
  const std::vector<double>& RowUtilities() const { return row_utilities_; }
  const std::vector<double>& ColUtilities() const { return col_utilities_; }
  const std::vector<double>& PlayerUtilities(
      const Player player) const {
    SPIEL_CHECK_TRUE(player == Player{0} || player == Player{1});
    return (player == Player{0} ? row_utilities_ : col_utilities_);
  }
  const std::string& RowActionName(int row) const {
    return row_action_names_[row];
  }
  const std::string& ColActionName(int col) const {
    return col_action_names_[col];
  }

  std::vector<double> GetUtilities(const std::vector<Action>& joint_action)
      const override {
    int index = Index(joint_action[0], joint_action[1]);
    return {row_utilities_[index], col_utilities_[index]};
  }

  double GetUtility(Player player, const std::vector<Action>& joint_action)
      const override {
    return PlayerUtility(player, joint_action[0], joint_action[1]);
  }

  bool operator==(const Game& other_game) const override {
    const auto& other = down_cast<const MatrixGame&>(other_game);
    return (row_action_names_.size() == other.row_action_names_.size() &&
            col_action_names_.size() == other.col_action_names_.size() &&
            row_utilities_ == other.row_utilities_ &&
            col_utilities_ == other.col_utilities_);
  }

  bool ApproxEqual(const Game& other_game, double tolerance) const {
    const auto& other = down_cast<const MatrixGame&>(other_game);
    return (row_action_names_.size() == other.row_action_names_.size() &&
            col_action_names_.size() == other.col_action_names_.size() &&
            AllNear(row_utilities_, other.row_utilities_, tolerance) &&
            AllNear(col_utilities_, other.col_utilities_, tolerance));
  }

 private:
  int Index(int row, int col) const { return row * NumCols() + col; }
  std::vector<std::string> row_action_names_;
  std::vector<std::string> col_action_names_;
  std::vector<double> row_utilities_;
  std::vector<double> col_utilities_;
};

class MatrixState : public NFGState {
 public:
  explicit MatrixState(std::shared_ptr<const Game> game);
  explicit MatrixState(const MatrixState&) = default;

  std::vector<Action> LegalActions(Player player) const override {
    if (IsTerminal()) return {};
    if (player == kSimultaneousPlayerId) {
      return LegalFlatJointActions();
    } else {
      std::vector<Action> moves(player == kRowPlayer ? matrix_game_->NumRows()
                                                     : matrix_game_->NumCols());
      std::iota(moves.begin(), moves.end(), 0);  // fill with values 0...n-1
      return moves;
    }
  }

  std::string ToString() const override;

  std::string ActionToString(Player player, Action action_id) const override {
    if (player == kSimultaneousPlayerId)
      return FlatJointActionToString(action_id);
    else if (player == kRowPlayer)
      return matrix_game_->RowActionName(action_id);
    else
      return matrix_game_->ColActionName(action_id);
  }

  bool IsTerminal() const override { return !joint_move_.empty(); }

  std::vector<double> Returns() const override {
    if (IsTerminal()) {
      return {matrix_game_->RowUtility(joint_move_[0], joint_move_[1]),
              matrix_game_->ColUtility(joint_move_[0], joint_move_[1])};
    } else {
      return {0, 0};
    }
  }

  std::unique_ptr<State> Clone() const override {
    return std::unique_ptr<State>(new MatrixState(*this));
  }

 protected:
  void DoApplyActions(const std::vector<Action>& moves) override {
    SPIEL_CHECK_EQ(moves.size(), 2);
    SPIEL_CHECK_GE(moves[kRowPlayer], 0);
    SPIEL_CHECK_LT(moves[kRowPlayer], matrix_game_->NumRows());
    SPIEL_CHECK_GE(moves[kColPlayer], 0);
    SPIEL_CHECK_LT(moves[kColPlayer], matrix_game_->NumCols());
    joint_move_ = moves;
  }

 private:
  std::vector<Action> joint_move_{};  // joint move that was chosen
  const MatrixGame* matrix_game_;
};

// Create a matrix game with the specified utilities and row/column names.
// Utilities must be in row-major form.
std::shared_ptr<const MatrixGame> CreateMatrixGame(
    const std::string& short_name, const std::string& long_name,
    const std::vector<std::string>& row_names,
    const std::vector<std::string>& col_names,
    const std::vector<std::vector<double>>& row_player_utils,
    const std::vector<std::vector<double>>& col_player_utils);

// Create a matrix game with the specified utilities, with default names
// ("short_name", "Long Name", row player utilities, col player utilities).
// Utilities must be in row-major order.
std::shared_ptr<const MatrixGame> CreateMatrixGame(
    const std::string& short_name, const std::string& long_name,
    const std::vector<std::string>& row_names,
    const std::vector<std::string>& col_names,
    const std::vector<double>& flat_row_utils,
    const std::vector<double>& flat_col_utils);

// Create a matrix game with the specified utilities, with default names
// ("short_name", "Long Name", row0, row1.., col0, col1, ...).
// Utilities must be in row-major form.
std::shared_ptr<const MatrixGame> CreateMatrixGame(
    const std::vector<std::vector<double>>& row_player_utils,
    const std::vector<std::vector<double>>& col_player_utils);

}  // namespace matrix_game
}  // namespace open_spiel

#endif  // OPEN_SPIEL_MATRIX_GAME_H_
