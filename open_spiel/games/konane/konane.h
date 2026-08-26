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

#ifndef OPEN_SPIEL_GAMES_KONANE_H_
#define OPEN_SPIEL_GAMES_KONANE_H_

#include <memory>
#include <string>
#include <vector>

#include "open_spiel/spiel.h"

// Konane (Hawaiian checkers).
// https://en.wikipedia.org/wiki/Konane
//
// A two-player combinatorial game played on a rectangular board that starts
// completely filled with alternating black and white stones. The game has two
// opening moves followed by ordinary play:
//
//   1. Black removes one black stone from a corner or from the centre.
//   2. White removes one white stone orthogonally adjacent to that hole.
//   3. Thereafter players alternate, moving one of their own stones by jumping
//      orthogonally over an adjacent enemy stone into the empty square directly
//      beyond it, capturing the jumped stone. A turn may chain several jumps,
//      but every jump in a turn must be in the same direction.
//
// The first player unable to move loses.
//
// Parameters:
//   "rows"     int  number of rows on the board    (default = 8)
//   "columns"  int  number of columns on the board (default = 8)

namespace open_spiel {
namespace konane {

inline constexpr int kNumPlayers = 2;

enum class CellState {
  kEmpty,  // Represented by '.'.
  kBlack,  // Represented by 'x'; player 0, moves first.
  kWhite,  // Represented by 'o'; player 1.
};

class KonaneState : public State {
 public:
  KonaneState(std::shared_ptr<const Game> game, int rows, int columns);
  KonaneState(const KonaneState&) = default;

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
  std::unique_ptr<State> Clone() const override {
    return std::unique_ptr<State>(new KonaneState(*this));
  }
  std::vector<Action> LegalActions() const override;

  bool InBounds(int row, int column) const {
    return row >= 0 && row < rows_ && column >= 0 && column < columns_;
  }
  void SetBoard(int row, int column, CellState state) {
    board_[row * columns_ + column] = state;
  }
  CellState BoardAt(int row, int column) const {
    return board_[row * columns_ + column];
  }

 protected:
  void DoApplyAction(Action action) override;

 private:
  // The two opening moves remove a stone rather than jumping with one.
  bool IsRemovalPhase() const { return num_moves_ < 2; }
  std::vector<Action> FirstRemovalActions() const;
  std::vector<Action> SecondRemovalActions() const;
  std::vector<Action> JumpActions() const;

  Player current_player_ = 0;  // Player zero (Black, 'x') goes first.
  Player outcome_ = kInvalidPlayer;
  int num_moves_ = 0;
  int rows_;
  int columns_;
  int max_jumps_;
  std::vector<CellState> board_;
};

class KonaneGame : public Game {
 public:
  explicit KonaneGame(const GameParameters& params);
  int NumDistinctActions() const override;
  std::unique_ptr<State> NewInitialState() const override {
    return std::make_unique<KonaneState>(shared_from_this(), rows_, columns_);
  }
  int NumPlayers() const override { return kNumPlayers; }
  double MinUtility() const override { return -1; }
  absl::optional<double> UtilitySum() const override { return 0; }
  double MaxUtility() const override { return 1; }
  std::vector<int> ObservationTensorShape() const override {
    return {kNumPlayers + 1, rows_, columns_};
  }
  // Two removals, then every turn captures at least one stone.
  int MaxGameLength() const override { return rows_ * columns_; }

 private:
  int rows_;
  int columns_;
  int max_jumps_;
};

std::ostream& operator<<(std::ostream& stream, const CellState& state);

// Longest chain of jumps possible in a single direction on this board.
int MaxJumps(int rows, int columns);

}  // namespace konane
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_KONANE_H_
