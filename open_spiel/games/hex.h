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

#ifndef OPEN_SPIEL_GAMES_HEX_H_
#define OPEN_SPIEL_GAMES_HEX_H_

#include <array>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "open_spiel/spiel.h"

// The classic game of Hex: https://en.wikipedia.org/wiki/Hex_(board_game)
// Does not implement pie rule to balance the game
//
// Parameters:
//       "board_size"    int     size of the board   (default = 11)
//       "num_cols"      int     number of columns (optional)
//       "num_rows"      int     number of rows (optional)

namespace open_spiel {
namespace hex {

// Constants.
inline constexpr int kNumPlayers = 2;
inline constexpr int kDefaultBoardSize = 11;
inline constexpr int kMaxNeighbours =
    6;  // Maximum number of neighbours for a cell
inline constexpr int kCellStates = 1 + 4 * kNumPlayers;
inline constexpr int kMinValueCellState = -4;
// State of a cell.
// Describes if a cell is
//   - empty, black or white
//   - connected to N/S edges if black, or was a winning move
//   - connected to E/W edges if white, or was a winning move
// These are used in calculation of winning connections, and may be useful
// features for learning agents
//
// Convention is that black plays first (and is player 0)
enum class CellState {
  kEmpty = 0,
  kWhiteWest = -3,
  kWhiteEast = -2,
  kWhiteWin = -4,
  kWhite = -1,  // White and not edge connected
  kBlackNorth = 3,
  kBlackSouth = 2,
  kBlackWin = 4,
  kBlack = 1,  // Black and not edge connected
};

// State of an in-play game.
class HexState : public State {
 public:
  HexState(std::shared_ptr<const Game> game, int num_cols, int num_rows);

  HexState(const HexState&) = default;

  Player CurrentPlayer() const override {
    return IsTerminal() ? kTerminalPlayerId : current_player_;
  }
  std::string ActionToString(Player player, Action action_id) const override;
  std::string ToString() const override;
  bool IsTerminal() const override;
  std::vector<double> Returns() const override;
  std::string InformationStateString(Player player) const override;
  std::string ObservationString(Player player) const override;
  void ObservationTensor(Player player,
                         absl::Span<float> values) const override;
  std::unique_ptr<State> Clone() const override;
  std::vector<Action> LegalActions() const override;

  CellState BoardAt(int cell) const { return board_[cell]; }
  void ChangePlayer() { current_player_ = current_player_ == 0 ? 1 : 0; }

 protected:
  std::vector<CellState> board_;
  void DoApplyAction(Action move) override;

 private:
  CellState PlayerAndActionToState(Player player, Action move) const;
  Player current_player_ = 0;                      // Player zero goes first
  double result_black_perspective_ = 0;            // 1 if Black (player 0) wins
  std::vector<int> AdjacentCells(int cell) const;  // Cells adjacent to cell

  const int num_cols_;  // x
  const int num_rows_;  // y
};

// Game object.
class HexGame : public Game {
 public:
  explicit HexGame(const GameParameters& params);
  int NumDistinctActions() const override { return num_cols_ * num_rows_; }
  std::unique_ptr<State> NewInitialState() const override {
    return std::unique_ptr<State>(
        new HexState(shared_from_this(), num_cols_, num_rows_));
  }
  int NumPlayers() const override { return kNumPlayers; }
  double MinUtility() const override { return -1; }
  double UtilitySum() const override { return 0; }
  double MaxUtility() const override { return 1; }
  std::vector<int> ObservationTensorShape() const override {
    return {kCellStates, num_cols_, num_rows_};
  }
  int MaxGameLength() const override { return num_cols_ * num_rows_; }

 private:
  const int num_cols_;
  const int num_rows_;
};

CellState PlayerToState(Player player);
std::string StateToString(CellState state);

inline std::ostream& operator<<(std::ostream& stream, const CellState& state) {
  return stream << StateToString(state);
}

}  // namespace hex
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_HEX_H_
