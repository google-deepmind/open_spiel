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

#ifndef OPEN_SPIEL_GAMES_LINES_OF_ACTION_H_
#define OPEN_SPIEL_GAMES_LINES_OF_ACTION_H_

#include <array>
#include <memory>
#include <string>
#include <vector>

#include "open_spiel/abseil-cpp/absl/container/flat_hash_set.h"
#include "open_spiel/abseil-cpp/absl/types/optional.h"
#include "open_spiel/abseil-cpp/absl/types/span.h"
#include "open_spiel/game_parameters.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_globals.h"
#include "open_spiel/spiel_utils.h"

// Lines of Action: https://en.wikipedia.org/wiki/Lines_of_Action
//
// Parameters: none
//
// Move notation and example games taken from Mark Winands' master thesis:
// (Winands, 2000, "Analysis and Implementation of Lines of Action")
// https://project.dke.maastrichtuniversity.nl/games/files/msc/Winands_thesis.pdf
//
// Uses the MSO rules on Page 10 of this thesis, including the two extra rules:
//
// 9. If a player cannot move, this player loses.2
// 10. If a position with the same player to move occurs for the second time,
//     the game is drawn

namespace open_spiel {
namespace lines_of_action {

inline constexpr int kNumRows = 8;
inline constexpr int kNumCols = 8;
inline constexpr int kNumCells = kNumRows * kNumCols;
inline constexpr int kNumLines = 4;
inline constexpr int kNumDirections = 8;
inline constexpr int kCellStates = 3;
inline constexpr int kNumPlayers = 2;
inline constexpr int kBlackPlayer = 0;
inline constexpr int kWhitePlayer = 1;
inline constexpr int kMaxGameLength = 1000;

// State of a cell.
enum class CellState {
  kBlack,  // x
  kWhite,  // o
  kEmpty,  // .
};

// Order of lines chosen so that direction mod 4 results in the correct line.
enum Line {
  kVertical = 0,
  kDiagonalSlash = 1,
  kHorizontal = 2,
  kDiagonalBackslash = 3
};

enum Direction {
  kUp = 0,
  kUpRight = 1,
  kRight = 2,
  kDownRight = 3,
  kDown = 4,
  kDownLeft = 5,
  kLeft = 6,
  kUpLeft = 7
};

constexpr std::array<int, 8> kRowOffsets = {1, 1, 0, -1, -1, -1, 0, 1};
constexpr std::array<int, 8> kColOffsets = {0, 1, 1, 1, 0, -1, -1, -1};

// State of an in-play game.
class LinesOfActionState : public State {
 public:
  LinesOfActionState(std::shared_ptr<const Game> game);

  LinesOfActionState(const LinesOfActionState&) = default;
  LinesOfActionState& operator=(const LinesOfActionState&) = default;

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
  std::vector<CellState> Board() const;
  CellState BoardAt(int cell) const { return board_[cell]; }
  CellState BoardAt(int row, int column) const {
    return board_[row * kNumCols + column];
  }
  void ChangePlayer() { current_player_ = current_player_ == 0 ? 1 : 0; }

 protected:
  std::array<CellState, kNumCells> board_;
  void DoApplyAction(Action move) override;

 private:
  CellState& board(int row, int column);
  std::vector<int> CountPiecesPerLine(int row, int col) const;
  void CheckTerminalState();
  int CountPiecesFloodFill(std::array<bool, kNumCells>* marked_board,
                           CellState cell_state, int row, int col) const;
  std::vector<Action> InternalLegalActions() const;
  std::string BoardToString() const;

  Player current_player_ = kBlackPlayer;  // Player zero (black) goes first
  Player winner_ = kInvalidPlayer;
  std::vector<Action> cached_legal_actions_;
  absl::flat_hash_set<std::string> visited_boards_;
};

class LinesOfActionGame : public Game {
 public:
  explicit LinesOfActionGame(const GameParameters& params);
  int NumDistinctActions() const override {
    // Move encodes the from square, to square, and boolean capture vs. no
    // capture.

    // For the from square, there is 8*8 = 64 possibilities.
    // For the to square, there is 8*8 = 64 possibilities.
    // For capture, there are 2 possibilities.
    // Total = (8*8) * (8*8) * 2.
    return kNumRows * kNumCols * kNumRows * kNumCols * 2;
  }
  std::unique_ptr<State> NewInitialState() const override {
    return std::unique_ptr<State>(new LinesOfActionState(shared_from_this()));
  }
  int NumPlayers() const override { return kNumPlayers; }
  double MinUtility() const override { return -1; }
  absl::optional<double> UtilitySum() const override { return 0; }
  double MaxUtility() const override { return 1; }
  std::vector<int> ObservationTensorShape() const override {
    return {kCellStates, kNumRows, kNumCols};
  }
  int MaxGameLength() const override { return kMaxGameLength; }
  std::string ActionToString(Player player, Action action_id) const override;

  const std::vector<int>& ActionBases() const { return kActionBases; }

 private:
  // Used for mapping actions to integers and back.
  // From square, to square, and boolean capture vs. no capture.
  const std::vector<int> kActionBases = {8, 8, 8, 8, 2};
};

}  // namespace lines_of_action
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_LINES_OF_ACTION_H_
