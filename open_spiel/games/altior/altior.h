// Copyright 2026 DeepMind Technologies Limited
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

#ifndef OPEN_SPIEL_GAMES_ALTIOR_H_
#define OPEN_SPIEL_GAMES_ALTIOR_H_

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/types/optional.h"
#include "absl/types/span.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

// Altior: an abstract strategy game by Edgar San Martin, Jr..
//   Playable: https://esanmartin.com/fun/altior/
//   Rules:    https://esanmartin.com/fun/altior-rules.pdf
//
// Rules implemented:
//   * Two players, Ivory (player 0, moves first) and Onyx (player 1), on a
//     7x7 board with the center cell removed. The 48 playable cells start with
//     one disc each, in a checkerboard: Ivory on every cell whose (row + col)
//     is even, counting rows down from the top, and Onyx on the others. So
//     Ivory holds a7 and Onyx holds b7.
//   * A stack may be moved only if its top disc belongs to the mover. It moves
//     orthogonally, in a straight line, exactly as many cells as it is tall.
//     It may jump over any intervening cells, including the removed center.
//   * The destination must be a playable cell. If it is occupied the moving
//     stack lands on top of it, keeping the bottom-to-top order of all discs.
//     The top disc decides who controls the merged stack.
//   * No stack may exceed five discs; such a move is illegal. A stack of five
//     discs can neither move nor be landed on.
//   * A player wins immediately by moving a stack of exactly four discs four
//     cells onto an EMPTY playable cell.
//   * If the player to move has no legal move they lose, unless the opponent
//     could not move either, in which case the game is a draw.
//   * A position (every stack, bottom to top, plus the player to move) that
//     occurs for the third time is a draw.
//
// Parameters:
//   "max_game_length" (int, default 1000): number of plies after which the
//       game is declared a draw. This is NOT part of the Altior rules. It is
//       an implementation safeguard because OpenSpiel requires a finite
//       MaxGameLength(). It is reported separately from the repetition draw.
//
// Actions are encoded as `source_cell * 4 + direction` where cells are
// numbered row-major from the top-left (a7 = 0, g1 = 48) and the directions
// are kUp, kDown, kLeft, kRight. The distance is the height of the source
// stack, so 196 action ids exist and most are illegal in any given state.

namespace open_spiel {
namespace altior {

inline constexpr int kNumPlayers = 2;
inline constexpr int kIvory = 0;
inline constexpr int kOnyx = 1;
inline constexpr int kNumRows = 7;
inline constexpr int kNumCols = 7;
inline constexpr int kNumCells = kNumRows * kNumCols;
inline constexpr int kCenterCell = (kNumRows / 2) * kNumCols + kNumCols / 2;
inline constexpr int kNumDirections = 4;
inline constexpr int kNumDistinctActions = kNumCells * kNumDirections;
inline constexpr int kMaxHeight = 5;
inline constexpr int kWinHeight = 4;
inline constexpr int kRepetitionsForDraw = 3;
inline constexpr int kDefaultMaxGameLength = 1000;

// Observation planes, each of size kNumRows x kNumCols:
//   [0, 5)   Ivory disc at stack depth d (0 = bottom of the stack).
//   [5, 10)  Onyx disc at stack depth d.
//   10       empty playable cell.
//   11       unplayable (removed) cell.
//   12       Ivory to move (all zero if terminal).
//   13       Onyx to move (all zero if terminal).
inline constexpr int kOnyxPlaneOffset = kMaxHeight;
inline constexpr int kEmptyPlane = 2 * kMaxHeight;
inline constexpr int kUnplayablePlane = kEmptyPlane + 1;
inline constexpr int kToMovePlaneOffset = kUnplayablePlane + 1;
inline constexpr int kNumObservationPlanes = kToMovePlaneOffset + kNumPlayers;

// Action directions. "Up" decreases the row, i.e. moves toward rank 7.
enum Direction { kUp = 0, kDown = 1, kLeft = 2, kRight = 3 };

// A stack of discs. Only the first `height` entries of `discs` are meaningful;
// each holds the owning player (kIvory or kOnyx), bottom disc first.
struct Stack {
  int8_t height = 0;
  std::array<int8_t, kMaxHeight> discs = {};

  bool empty() const { return height == 0; }
  // Owner of the top disc, or kInvalidPlayer if the stack is empty.
  Player Top() const {
    return height == 0 ? kInvalidPlayer : discs[height - 1];
  }
  // Builds a stack from a string such as "ioi" ('i' = Ivory, 'o' = Onyx,
  // bottom disc first). Fails on any other character or on > kMaxHeight discs.
  static Stack FromString(const std::string& discs);
  // Inverse of FromString; the empty stack yields an empty string.
  std::string ToString() const;
};

using Board = std::array<Stack, kNumCells>;

// Why the game ended. kNone means it is still in progress.
enum class EndReason {
  kNone,
  kFourStackWin,      // The mover landed a four-stack on an empty cell.
  kOpponentImmobile,  // The player to move had no move and the mover did.
  kBothImmobile,      // Neither player had a move: draw.
  kRepetition,        // Third occurrence of a position: draw.
  kMoveLimit,         // max_game_length reached: draw (safeguard, not a rule).
};

// Returns the starting arrangement: a checkerboard of single discs.
Board InitialBoard();

class AltiorState : public State {
 public:
  explicit AltiorState(std::shared_ptr<const Game> game);
  // Starts from an arbitrary position. Intended for tests and analysis. The
  // resulting state has an empty history, so Serialize() (which is based on
  // the action history) does not capture it. The initial position counts as
  // the first occurrence for repetition purposes. If `to_move` cannot move,
  // the game is immediately terminal under the usual immobility rule.
  AltiorState(std::shared_ptr<const Game> game, const Board& board,
              Player to_move);
  AltiorState(const AltiorState&) = default;
  // State holds a const Game pointer, so it cannot be assigned. Clone() uses
  // the copy constructor.
  AltiorState& operator=(const AltiorState&) = delete;

  Player CurrentPlayer() const override {
    return IsTerminal() ? kTerminalPlayerId : current_player_;
  }
  std::string ActionToString(Player player, Action action_id) const override;
  std::string ToString() const override;
  bool IsTerminal() const override { return end_reason_ != EndReason::kNone; }
  std::vector<double> Returns() const override;
  std::string ObservationString(Player player) const override;
  void ObservationTensor(Player player,
                         absl::Span<float> values) const override;
  std::unique_ptr<State> Clone() const override;
  void UndoAction(Player player, Action move) override;
  std::vector<Action> LegalActions() const override { return legal_actions_; }

  const Board& board() const { return board_; }
  const Stack& StackAt(int cell) const { return board_[cell]; }
  EndReason end_reason() const { return end_reason_; }
  // The winning player, or kInvalidPlayer if in progress or drawn.
  Player winner() const { return winner_; }
  // How many times the current position has occurred in this game.
  int RepetitionCount() const;

 protected:
  void DoApplyAction(Action action) override;

 private:
  // Legal moves for `player` (sorted ascending), ignoring the game result.
  std::vector<Action> MovesFor(Player player) const;
  bool HasMoveFor(Player player) const;
  // Decides the result after `mover` has just moved (or at construction).
  void Settle(Player mover);
  // Compact unique encoding of the board and the player to move.
  std::string PositionKey() const;

  Board board_;
  Player current_player_ = kIvory;
  Player winner_ = kInvalidPlayer;
  EndReason end_reason_ = EndReason::kNone;
  std::vector<Action> legal_actions_;
  // Height of the stack moved at each ply, needed to undo a move.
  std::vector<int8_t> moved_heights_;
  // Occurrences of each position. The terminal position of a win or an
  // immobility result is never recorded.
  absl::flat_hash_map<std::string, int> position_counts_;
};

class AltiorGame : public Game {
 public:
  explicit AltiorGame(const GameParameters& params);
  int NumDistinctActions() const override { return kNumDistinctActions; }
  std::unique_ptr<State> NewInitialState() const override {
    return std::make_unique<AltiorState>(shared_from_this());
  }
  int NumPlayers() const override { return kNumPlayers; }
  double MinUtility() const override { return -1; }
  double MaxUtility() const override { return 1; }
  absl::optional<double> UtilitySum() const override { return 0; }
  std::vector<int> ObservationTensorShape() const override {
    return {kNumObservationPlanes, kNumRows, kNumCols};
  }
  int MaxGameLength() const override { return max_game_length_; }

 private:
  const int max_game_length_;
};

}  // namespace altior
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_ALTIOR_H_
