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

#ifndef OPEN_SPIEL_NORMAL_FORM_GAME_H_
#define OPEN_SPIEL_NORMAL_FORM_GAME_H_

#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/abseil-cpp/absl/strings/str_join.h"
#include "open_spiel/simultaneous_move_game.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

// This class describes an n-player normal-form game. A normal-form game is
// also known as a one-shot game or strategic-form game. Essentially, all
// players act simultaneously and the game ends after a single joint action
// taken by all players. E.g. a matrix game is an example of a 2-player normal
// form game.

namespace open_spiel {

class NFGState : public SimMoveState {
 public:
  NFGState(std::shared_ptr<const Game> game) : SimMoveState(game) {}

  // There are no chance nodes in a normal-form game (there is only one state),
  Player CurrentPlayer() const final {
    return IsTerminal() ? kTerminalPlayerId : kSimultaneousPlayerId;
  }

  // Since there's only one state, we can implement the representations here.
  std::string InformationStateString(Player player) const override {
    SPIEL_CHECK_GE(player, 0);
    SPIEL_CHECK_LT(player, num_players_);
    std::string info_state = absl::StrCat("Observing player: ", player, ". ");
    if (!IsTerminal()) {
      absl::StrAppend(&info_state, "Non-terminal");
    } else {
      absl::StrAppend(&info_state,
                      "Terminal. History string: ", HistoryString());
    }
    return info_state;
  }

  std::string ObservationString(Player player) const override {
    SPIEL_CHECK_GE(player, 0);
    SPIEL_CHECK_LT(player, num_players_);
    std::string obs_str;
    if (!IsTerminal()) {
      absl::StrAppend(&obs_str, "Non-terminal");
    } else {
      absl::StrAppend(&obs_str, "Terminal. History string: ", HistoryString());
    }
    return obs_str;
  }

  std::string ToString() const override {
    std::string result = "Normal form game default NFGState::ToString. ";
    if (IsTerminal()) {
      absl::StrAppend(&result, "Terminal, history: ", HistoryString(),
                      ", returns: ", absl::StrJoin(Returns(), ","));
    } else {
      absl::StrAppend(&result, "Non-terminal");
    }
    return result;
  }

  void InformationStateTensor(Player player,
                              absl::Span<float> values) const override {
    SPIEL_CHECK_GE(player, 0);
    SPIEL_CHECK_LT(player, num_players_);
    SPIEL_CHECK_EQ(values.size(), 1);
    if (IsTerminal()) {
      values[0] = 1;
    } else {
      values[0] = 0;
    }
  }

  void ObservationTensor(Player player,
                         absl::Span<float> values) const override {
    SPIEL_CHECK_GE(player, 0);
    SPIEL_CHECK_LT(player, num_players_);
    SPIEL_CHECK_EQ(values.size(), 1);
    if (IsTerminal()) {
      values[0] = 1;
    } else {
      values[0] = 0;
    }
  }
};

class NormalFormGame : public SimMoveGame {
 public:
  // Game has one state.
  std::vector<int> InformationStateTensorShape() const override {
    return {1};
  }
  std::vector<int> ObservationTensorShape() const override { return {1}; }

  // Game lasts one turn.
  int MaxGameLength() const override { return 1; }
  // There aren't chance nodes in these games.
  int MaxChanceNodesInHistory() const override { return 0; }

  // Direct access to utility. This is just a default implementation, which is
  // overridden in subclasses for faster access.
  virtual std::vector<double> GetUtilities(
      const std::vector<Action>& joint_action) const {
    std::unique_ptr<State> state = NewInitialState();
    state->ApplyActions(joint_action);
    return state->Returns();
  }

  virtual double GetUtility(Player player,
                            const std::vector<Action>& joint_action) const {
    return GetUtilities(joint_action)[player];
  }

  double UtilitySum() const override {
    if (game_type_.utility == GameType::Utility::kZeroSum) {
      return 0.0;
    } else if (game_type_.utility == GameType::Utility::kConstantSum) {
      std::vector<Action> joint_action(NumPlayers(), 0);
      std::vector<double> utilities = GetUtilities(joint_action);
      return std::accumulate(utilities.begin(), utilities.end(), 0.0);
    }
    SpielFatalError(absl::StrCat("No appropriate UtilitySum value for ",
                                 "general-sum or identical utility games."));
  }

 protected:
  NormalFormGame(GameType game_type, GameParameters game_parameters)
      : SimMoveGame(game_type, game_parameters) {}
};

}  // namespace open_spiel

#endif  // OPEN_SPIEL_NORMAL_FORM_GAME_H_
