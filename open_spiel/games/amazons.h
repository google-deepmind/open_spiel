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
// Parameters: none

namespace open_spiel {
namespace amazons {

// Constants.
inline constexpr int kNumPlayers = 2;
inline constexpr int kNumRows = 8;
inline constexpr int kNumCols = 8;
inline constexpr int kNumCells = kNumRows * kNumCols;
inline constexpr int kCellStates = 4;  // empty, 'X', 'O', '@'.

//Hensgens et al = 10e40 for 10x10
inline constexpr int kNumberStates = 1000000000;

// State of a cell.
enum class CellState {
  kEmpty,
  kNought,
  kCross,
  kBlock
};

// State of an in-play game.
class AmazonsState : public State {
 public:
  AmazonsState(std::shared_ptr<const Game> game);
  AmazonsState(const AmazonsState&) = default;

  //??
  AmazonsState& operator=(const AmazonsState&) = default;

  // Is this okay?
  Player CurrentPlayer() const override {
    return IsTerminal() ? kTerminalPlayerId : current_player_;
  }
  std::string ActionToString(Player player, Action action_id) const override;

  std::string ToString() const override;

  bool IsTerminal() const override;

  std::vector<double> Returns() const override;
  
  //?
  std::string InformationStateString(Player player) const override;

  //?
  std::string ObservationString(Player player) const override;

  //?
  void ObservationTensor(Player player, absl::Span<float> values) const override;

  std::unique_ptr<State> Clone() const override;
  
  void UndoAction(Player player, Action move) override;

  std::vector<Action> LegalActions() const override;

  CellState BoardAt(int cell) const { return board_[cell]; }
  CellState BoardAt(int row, int column) const {
    return board_[row * kNumCols + column];
  }

 protected:

  std::array<CellState, kNumCells> board_;

  void DoApplyAction(Action move) override;

 private:
  std::vector<unsigned char> GetAllMoves(unsigned char, std::array<CellState, kNumCells>) const;
  std::vector<unsigned char> GetDiagonalMoves(unsigned char, std::array<CellState, kNumCells>) const;
  std::vector<unsigned char> GetVerticalMoves(unsigned char, std::array<CellState, kNumCells>) const;
  std::vector<unsigned char> GetHorizontalMoves(unsigned char, std::array<CellState, kNumCells>) const;

  Action EncodeAction(std::vector<unsigned char>) const;
  std::vector<unsigned char> DecodeAction(Action) const;

  Player current_player_ = 0;         // Player zero goes first
  Player outcome_ = kInvalidPlayer;   // Outcome unclear at init
  int num_moves_ = 0;
};

// Game object.
class AmazonsGame : public Game {
 public:
  explicit AmazonsGame(const GameParameters& params);

  // 4 8 9 8 9
  int NumDistinctActions() const override { return 20736; }
  
  std::unique_ptr<State> NewInitialState() const override {
    return std::unique_ptr<State>(new AmazonsState(shared_from_this()));
  }

  int NumPlayers() const override { return kNumPlayers; }

  double MinUtility() const override { return -1; }
  double UtilitySum() const override { return 0; }
  double MaxUtility() const override { return 1; }

  // ??
  std::vector<int> ObservationTensorShape() const override {
    return {kCellStates, kNumRows, kNumCols};
  }

  // ???
  int MaxGameLength() const override { return kNumCells * 2; }
};

CellState PlayerToState(Player player);
std::string StateToString(CellState state);

inline std::ostream& operator<<(std::ostream& stream, const CellState& state) {
  return stream << StateToString(state);
}

}  // namespace amazons
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_AMAZONS_H_
