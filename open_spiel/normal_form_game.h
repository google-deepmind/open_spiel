// Copyright 2019 DeepMind Technologies Ltd. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef THIRD_PARTY_OPEN_SPIEL_NORMAL_FORM_GAME_H_
#define THIRD_PARTY_OPEN_SPIEL_NORMAL_FORM_GAME_H_

#include <memory>
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
    std::string info_state = absl::StrCat("Observing player: ", player, ". ");
    if (!IsTerminal()) {
      absl::StrAppend(&info_state, "Non-terminal");
    } else {
      absl::StrAppend(&info_state,
                      "Terminal. History string: ", HistoryString());
    }
    return info_state;
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
                              std::vector<double>* values) const override {
    values->resize(1);
    if (IsTerminal()) {
      (*values)[0] = 1;
    } else {
      (*values)[0] = 0;
    }
  }
};

class NormalFormGame : public SimMoveGame {
 public:
  // Game has one state.
  std::vector<int> InformationStateTensorShape() const override {
    return {1};
  }

  // Game lasts one turn.
  int MaxGameLength() const override { return 1; }

 protected:
  NormalFormGame(GameType game_type, GameParameters game_parameters)
      : SimMoveGame(game_type, game_parameters) {}
};

}  // namespace open_spiel

#endif  // THIRD_PARTY_OPEN_SPIEL_NORMAL_FORM_GAME_H_
