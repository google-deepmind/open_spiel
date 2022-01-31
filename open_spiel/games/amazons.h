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

#ifndef OPEN_SPIEL_GAMES_AMAZONS_H_
#define OPEN_SPIEL_GAMES_AMAZONS_H_

#include <array>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "open_spiel/spiel.h"

// The "Game of Amazons":
// https://en.wikipedia.org/wiki/Game_of_the_Amazons
//
// Parameters: TODO: let the user choose the dimension

namespace open_spiel {
namespace amazons {

// Constants.

inline constexpr int kNumPlayers = 2;
inline constexpr int kNumRows = 6;
inline constexpr int kNumCols = 6;
inline constexpr int kNumCells = kNumRows * kNumCols;
inline constexpr int kCellStates = 4;  // empty, 'X', 'O', '@'.

// Hensgens et al = 10e40 for 10x10
inline constexpr int kNumberStates = 1000000000;

// State of a cell.
enum class CellState { kEmpty, kNought, kCross, kBlock };

class AmazonsGame;

// State of an in-play game.
class AmazonsState : public State {
 public:
  enum MoveState { amazon_select, destination_select, shot_select };

  AmazonsState(std::shared_ptr<const Game> game);

  AmazonsState(const AmazonsState&) = default;

  AmazonsState& operator=(const AmazonsState&) = default;

  Player CurrentPlayer() const override {
    return IsTerminal() ? kTerminalPlayerId : current_player_;
  }
  std::string ActionToString(Player player, Action action_id) const override;

  std::string ToString() const override;

  bool IsTerminal() const override;

  void SetState(int cur_player, MoveState move_state,
                const std::array<CellState, kNumCells>& board);

  std::vector<double> Returns() const override;

  std::string InformationStateString(Player player) const override;

  std::string ObservationString(Player player) const override;

  void ObservationTensor(Player player,
                         absl::Span<float> values) const override;

  std::unique_ptr<State> Clone() const override;

  void UndoAction(Player player, Action move) override;

  std::vector<Action> LegalActions() const override;

  CellState BoardAt(int cell) const { return board_[cell]; }
  CellState BoardAt(int row, int column) const {
    return board_[row * kNumCols + column];
  }

 protected:
  std::array<CellState, kNumCells> board_;

  void DoApplyAction(Action action) override;

 private:
  MoveState state_ = amazon_select;
  int from_ = 0;
  int to_ = 0;
  int shoot_ = 0;

  std::vector<Action> GetAllMoves(Action) const;
  std::vector<Action> GetDiagonalMoves(Action) const;
  std::vector<Action> GetVerticalMoves(Action) const;
  std::vector<Action> GetHorizontalMoves(Action) const;

  bool IsGameOver() const;

  Player current_player_ = 0;        // Player zero goes first
  Player outcome_ = kInvalidPlayer;  // Outcome unclear at init
  int num_moves_ = 0;
};

// Game object.
class AmazonsGame : public Game {
 public:
  explicit AmazonsGame(const GameParameters& params);

  int NumDistinctActions() const override { return kNumCells; }

  std::unique_ptr<State> NewInitialState() const override {
    return std::unique_ptr<State>(new AmazonsState(shared_from_this()));
  }

  int NumPlayers() const override { return kNumPlayers; }

  double MinUtility() const override { return -1; }
  double UtilitySum() const override { return 0; }
  double MaxUtility() const override { return 1; }

  std::vector<int> ObservationTensorShape() const override {
    return {kCellStates, kNumRows, kNumCols};
  }

  int MaxGameLength() const override { return 3 * kNumCells; }
};

CellState PlayerToState(Player player);
std::string StateToString(CellState state);

inline std::ostream& operator<<(std::ostream& stream, const CellState& state) {
  return stream << StateToString(state);
}

}  // namespace amazons
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_AMAZONS_H_
