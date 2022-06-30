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

#ifndef OPEN_SPIEL_GAMES_PENTAGO_H_
#define OPEN_SPIEL_GAMES_PENTAGO_H_

#include <array>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "open_spiel/spiel.h"

// https://en.wikipedia.org/wiki/Pentago
// Does not implement pie rule to balance the game
//
// Parameters:
//   "ansi_color_output" bool    Whether to color the output for a terminal.

namespace open_spiel {
namespace pentago {

inline constexpr int kNumPlayers = 2;
inline constexpr int kBoardSize = 6;
inline constexpr int kBoardPositions = kBoardSize * kBoardSize;
inline constexpr int kPossibleRotations = 8;
inline constexpr int kPossibleActions = kBoardPositions * kPossibleRotations;
inline constexpr int kPossibleWinConditions = 32;
inline constexpr int kCellStates = 1 + kNumPlayers;

enum PentagoPlayer {
  kPlayer1,
  kPlayer2,
  kPlayerNone,
  kPlayerDraw,
};

// State of an in-play game.
class PentagoState : public State {
 public:
  PentagoState(std::shared_ptr<const Game> game,
               bool ansi_color_output = false);

  PentagoState(const PentagoState&) = default;

  Player CurrentPlayer() const override {
    return IsTerminal() ? kTerminalPlayerId : static_cast<int>(current_player_);
  }
  std::string ActionToString(Player player, Action action_id) const override;
  std::string ToString() const override;
  bool IsTerminal() const override { return outcome_ != kPlayerNone; }
  std::vector<double> Returns() const override;
  std::string InformationStateString(Player player) const override;
  std::string ObservationString(Player player) const override;

  // A 3d tensor, 3 player-relative one-hot 2d planes. The layers are: the
  // specified player, the other player, and empty.
  void ObservationTensor(Player player,
                         absl::Span<float> values) const override;
  std::unique_ptr<State> Clone() const override;
  std::vector<Action> LegalActions() const override;

 protected:
  void DoApplyAction(Action action) override;

  PentagoPlayer get(int x, int y) const { return get(x + y * kBoardSize); }
  PentagoPlayer get(int i) const;

 private:
  std::array<uint64_t, kNumPlayers> board_;
  PentagoPlayer current_player_ = kPlayer1;
  PentagoPlayer outcome_ = kPlayerNone;
  int moves_made_ = 0;
  const bool ansi_color_output_;
};

// Game object.
class PentagoGame : public Game {
 public:
  explicit PentagoGame(const GameParameters& params);

  int NumDistinctActions() const override { return kPossibleActions; }
  std::unique_ptr<State> NewInitialState() const override {
    return std::unique_ptr<State>(
        new PentagoState(shared_from_this(), ansi_color_output_));
  }
  int NumPlayers() const override { return kNumPlayers; }
  double MinUtility() const override { return -1; }
  double UtilitySum() const override { return 0; }
  double MaxUtility() const override { return 1; }
  std::vector<int> ObservationTensorShape() const override {
    return {kCellStates, kBoardSize, kBoardSize};
  }
  int MaxGameLength() const override {
    // No stones are removed, and it is possible to draw by filling the board.
    // Increase this by one if swap is ever implemented.
    return kBoardPositions;
  }

 private:
  const bool ansi_color_output_ = false;
};

}  // namespace pentago
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_PENTAGO_H_
