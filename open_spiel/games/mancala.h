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

#ifndef OPEN_SPIEL_GAMES_MANCALA_H_
#define OPEN_SPIEL_GAMES_MANCALA_H_

#include <array>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "open_spiel/spiel.h"

// Mancala
// https://en.wikipedia.org/wiki/Mancala.
//
// Note that this implements the Kalah rule set, see
// https://en.wikipedia.org/wiki/Kalah. Oware is another game from the Mancala
// family of games implemented in oware.{h,cc}.
//
// Parameters: none

namespace open_spiel {
namespace mancala {

// Constants.
inline constexpr int kNumPlayers = 2;
inline constexpr int kNumPits = 6;
inline constexpr int kTotalPits = (kNumPits + 1) * 2;

// State of an in-play game.
class MancalaState : public State {
 public:
  MancalaState(std::shared_ptr<const Game> game);

  MancalaState(const MancalaState&) = default;
  MancalaState& operator=(const MancalaState&) = default;

  void SetBoard(const std::array<int, kTotalPits>& board);
  int BoardAt(int position) const { return board_[position]; }
  Player CurrentPlayer() const override {
    return IsTerminal() ? kTerminalPlayerId : current_player_;
  }
  std::string ActionToString(Player player, Action action_id) const override;
  std::string ToString() const override;
  bool IsTerminal() const override;
  std::vector<double> Returns() const override;
  std::string ObservationString(Player player) const override;
  void ObservationTensor(Player player,
                         absl::Span<float> values) const override;
  std::unique_ptr<State> Clone() const override;
  std::vector<Action> LegalActions() const override;

 protected:
  std::array<int, kTotalPits> board_;
  void DoApplyAction(Action move) override;

 private:
  void InitBoard();
  Player current_player_ = 0;  // Player zero goes first
};

// Game object.
class MancalaGame : public Game {
 public:
  explicit MancalaGame(const GameParameters& params);
  int NumDistinctActions() const override { return kTotalPits; }
  std::unique_ptr<State> NewInitialState() const override {
    return std::unique_ptr<State>(new MancalaState(shared_from_this()));
  }
  int NumPlayers() const override { return kNumPlayers; }
  double MinUtility() const override { return -1; }
  absl::optional<double> UtilitySum() const override { return 0; }
  double MaxUtility() const override { return 1; }
  std::vector<int> ObservationTensorShape() const override {
    return {kTotalPits};
  }
  // There is arbitrarily chosen number to ensure the game is finite.
  int MaxGameLength() const override { return 1000; }
};

}  // namespace mancala
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_MANCALA_H_
