// Copyright 2019 DeepMind Technologies Limited
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

#ifndef OPEN_SPIEL_GAMES_TWIXT_TWIXT_H_
#define OPEN_SPIEL_GAMES_TWIXT_TWIXT_H_

#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "open_spiel/games/twixt/twixtboard.h"
#include "open_spiel/games/twixt/twixtcell.h"

// https://en.wikipedia.org/wiki/TwixT

namespace open_spiel {
namespace twixt {

class TwixTState : public State {
 public:
  explicit TwixTState(std::shared_ptr<const Game> game);

  TwixTState(const TwixTState &) = default;
  TwixTState &operator=(const TwixTState &) = default;

  open_spiel::Player CurrentPlayer() const override { return current_player_; };

  std::string ActionToString(open_spiel::Player player,
                             Action action) const override;

  std::string ToString() const override { return board_.ToString(); };

  bool IsTerminal() const override {
    int result = board_.result();
    return (result == kRedWin || result == kBlueWin || result == kDraw);
  };

  std::vector<double> Returns() const override {
    double reward;
    int result = board_.result();
    if (result == kOpen || result == kDraw) {
      return {0.0, 0.0};
    } else {
      reward = 1.0;
      if (result == kRedWin) {
        return {reward, -reward};
      } else {
        return {-reward, reward};
      }
    }
  };

  std::string InformationStateString(open_spiel::Player player) const override {
    SPIEL_CHECK_GE(player, 0);
    SPIEL_CHECK_LT(player, kNumPlayers);
    return ToString();
  };

  std::string ObservationString(open_spiel::Player player) const override {
    SPIEL_CHECK_GE(player, 0);
    SPIEL_CHECK_LT(player, kNumPlayers);
    return ToString();
  };

  void ObservationTensor(open_spiel::Player player,
                         absl::Span<float> values) const override;

  std::unique_ptr<State> Clone() const override {
    return std::unique_ptr<State>(new TwixTState(*this));
  };

  void UndoAction(open_spiel::Player, Action) override{};

  std::vector<Action> LegalActions() const override {
    if (IsTerminal()) return {};
    return board_.GetLegalActions(current_player_);
  };

 protected:
  void DoApplyAction(Action action) override {
    const std::vector<Action> &v = LegalActions();
    if (std::find(v.begin(), v.end(), action) == v.end()) {
      SpielFatalError("Not a legal action: " + std::to_string(action));
    }
    board_.ApplyAction(CurrentPlayer(), action);
    if (board_.result() == kOpen) {
      set_current_player(1 - CurrentPlayer());
    } else {
      set_current_player(kTerminalPlayerId);
    }
  };

 private:
  Player current_player_ = kRedPlayer;
  Board board_;
  void set_current_player(Player player) { current_player_ = player; }
  void SetPegAndLinksOnTensor(absl::Span<float>, const Cell &, int, bool,
                              Position) const;
};

class TwixTGame : public Game {
 public:
  explicit TwixTGame(const GameParameters &params);

  std::unique_ptr<State> NewInitialState() const override {
    return std::unique_ptr<State>(new TwixTState(shared_from_this()));
  };

  int NumDistinctActions() const override { return board_size_ * board_size_; };

  int NumPlayers() const override { return kNumPlayers; };
  double MinUtility() const override { return -1.0; };
  absl::optional<double> UtilitySum() const override { return 0.0; };
  double MaxUtility() const override { return 1.0; };

  std::vector<int> ObservationTensorShape() const override {
    static std::vector<int> shape{kNumPlanes, board_size_, board_size_ - 2};
    return shape;
  }

  int MaxGameLength() const {
    // square - 4 corners + swap move
    return board_size_ * board_size_ - 4 + 1;
  }
  bool ansi_color_output() const { return ansi_color_output_; }
  int board_size() const { return board_size_; }

 private:
  bool ansi_color_output_;
  int board_size_;
};

}  // namespace twixt
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_TWIXT_TWIXT_H_
