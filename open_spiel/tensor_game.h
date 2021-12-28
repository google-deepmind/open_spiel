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

#ifndef OPEN_SPIEL_TENSOR_GAME_H_
#define OPEN_SPIEL_TENSOR_GAME_H_

#include <algorithm>
#include <iterator>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "open_spiel/normal_form_game.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

// A tensor game is an example of an n-player normal-form game.

namespace open_spiel {
namespace tensor_game {

class TensorGame : public NormalFormGame {
 public:
  // action_names[player] is the list of action names for player.
  // utilities[player] is a flattened tensor of utilities for player, in
  // row-major/C-style/lexicographic order of all players' actions.
  TensorGame(GameType game_type, GameParameters game_parameters,
             std::vector<std::vector<std::string>> action_names,
             std::vector<std::vector<double>> utilities)
      : NormalFormGame(std::move(game_type), std::move(game_parameters)),
        action_names_(std::move(action_names)),
        utilities_(std::move(utilities)),
        shape_(utilities_.size()) {
    int size = 1;
    for (Player player = 0; player < action_names_.size(); ++player) {
      size *= action_names_[player].size();
      shape_[player] = action_names_[player].size();
    }
    ComputeMinMaxUtility();
    SPIEL_CHECK_TRUE(std::all_of(utilities_.begin(), utilities_.end(),
                                 [size](const auto& player_utils) {
                                   return player_utils.size() == size;
                                 }));
  }

  // Implementation of Game interface
  int NumDistinctActions() const override {
    return *std::max_element(begin(shape_), end(shape_));
  }

  std::unique_ptr<State> NewInitialState() const override;

  int NumPlayers() const override { return utilities_.size(); }

  double MinUtility() const override { return min_utility_; }

  double MaxUtility() const override { return max_utility_; }

  const std::vector<int>& Shape() const { return shape_; }
  const double PlayerUtility(const Player player,
                             const std::vector<Action>& actions) const {
    SPIEL_CHECK_GE(player, 0);
    SPIEL_CHECK_LT(player, NumPlayers());
    return utilities_[player][index(actions)];
  }
  const std::vector<double>& PlayerUtilities(const Player player) const {
    SPIEL_CHECK_GE(player, 0);
    SPIEL_CHECK_LT(player, NumPlayers());
    return utilities_[player];
  }
  const std::string& ActionName(const Player player,
                                const Action& action) const {
    SPIEL_CHECK_GE(player, 0);
    SPIEL_CHECK_LT(player, NumPlayers());
    return action_names_[player][action];
  }

  bool operator==(const Game& other_game) const override {
    const auto& other = down_cast<const TensorGame&>(other_game);
    return (shape_ == other.shape_ && utilities_ == other.utilities_);
  }

  bool ApproxEqual(const Game& other_game, double tolerance) const {
    const auto& other = down_cast<const TensorGame&>(other_game);
    if (shape_ != other.shape_) {
      return false;
    }
    for (Player p = 0; p < NumPlayers(); ++p) {
      if (!AllNear(utilities_[p], other.utilities_[p], tolerance)) {
        return false;
      }
    }
    return true;
  }

  std::vector<double> GetUtilities(const std::vector<Action>& joint_action)
      const override {
    int idx = index(joint_action);
    std::vector<double> utilities;
    utilities.reserve(NumPlayers());
    for (Player p = 0; p < NumPlayers(); ++p) {
      utilities.push_back(utilities_[p][idx]);
    }
    return utilities;
  }

  double GetUtility(Player player, const std::vector<Action>& joint_action)
      const override {
    return PlayerUtility(player, joint_action);
  }

 private:
  const int index(const std::vector<Action>& args) const {
    int ind = 0;
    for (int i = 0; i < NumPlayers(); ++i) {
      ind = ind * shape_[i] + args[i];
    }
    return ind;
  }

  void ComputeMinMaxUtility() {
    min_utility_ = *std::min_element(begin(utilities_[0]), end(utilities_[0]));
    for (Player player = 1; player < NumPlayers(); ++player) {
      min_utility_ =
          std::min(min_utility_, *std::min_element(begin(utilities_[player]),
                                                   end(utilities_[player])));
    }

    max_utility_ = *std::max_element(begin(utilities_[0]), end(utilities_[0]));
    for (Player player = 1; player < NumPlayers(); ++player) {
      max_utility_ =
          std::max(max_utility_, *std::max_element(begin(utilities_[player]),
                                                   end(utilities_[player])));
    }
  }

  // action_names_[player] is the list of action names for player.
  const std::vector<std::vector<std::string>> action_names_;
  // utilities_[player] is a flattened tensor of utilities for player, in
  // row-major/C-style/lexicographic order of all players' actions.
  const std::vector<std::vector<double>> utilities_;
  std::vector<int> shape_;
  double min_utility_;
  double max_utility_;
};

class TensorState : public NFGState {
 public:
  explicit TensorState(std::shared_ptr<const Game> game);
  explicit TensorState(const TensorState&) = default;

  std::vector<Action> LegalActions(Player player) const override {
    if (IsTerminal()) return {};
    if (player == kSimultaneousPlayerId) {
      return LegalFlatJointActions();
    } else {
      std::vector<Action> moves(tensor_game_->Shape()[player]);
      std::iota(moves.begin(), moves.end(), 0);  // fill with values 0...n-1
      return moves;
    }
  }

  std::string ToString() const override;

  std::string ActionToString(Player player, Action action_id) const override {
    if (player == kSimultaneousPlayerId)
      return FlatJointActionToString(action_id);
    else
      return tensor_game_->ActionName(player, action_id);
  }

  bool IsTerminal() const override { return !joint_move_.empty(); }

  std::vector<double> Returns() const override {
    std::vector<double> returns(NumPlayers());
    if (IsTerminal()) {
      for (Player player = 0; player < returns.size(); player++) {
        returns[player] = tensor_game_->PlayerUtility(player, joint_move_);
      }
    }
    return returns;
  }

  std::unique_ptr<State> Clone() const override {
    return std::unique_ptr<State>(new TensorState(*this));
  }

 protected:
  void DoApplyActions(const std::vector<Action>& moves) override {
    SPIEL_CHECK_EQ(moves.size(), NumPlayers());
    for (Player player = 0; player < NumPlayers(); player++) {
      SPIEL_CHECK_GE(moves[player], 0);
      SPIEL_CHECK_LT(moves[player], tensor_game_->Shape()[player]);
    }
    joint_move_ = moves;
  }

 private:
  std::vector<Action> joint_move_{};  // joint move that was chosen
  const TensorGame* tensor_game_;
};

// Create a tensor game with the specified utilities and action names.
// utils[player] is a flattened tensor of utilities for player, in
// row-major/C-style/lexicographic order of all players' actions.


std::shared_ptr<const TensorGame> CreateTensorGame(
    const std::string& short_name, const std::string& long_name,
    const std::vector<std::vector<std::string>>& action_names,
    const std::vector<std::vector<double>>& utils);

// Create a tensor game with the specified utilities, with names
// "short_name", "Long Name" and action names
// action0_0, action0_1.. for player 0, and so forth for other players.
// utils[player] is a flattened tensor of utilities for player, in
// row-major/C-style/lexicographic order of all players' actions.

std::shared_ptr<const TensorGame> CreateTensorGame(
    const std::vector<std::vector<double>>& utils,
    const std::vector<int>& shape);

}  // namespace tensor_game
}  // namespace open_spiel

#endif  // OPEN_SPIEL_TENSOR_GAME_H_
