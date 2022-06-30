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

#ifndef OPEN_SPIEL_GAMES_CURSOR_GO_H_
#define OPEN_SPIEL_GAMES_CURSOR_GO_H_

#include <array>
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include "open_spiel/games/go/go_board.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

// Game of Go, with a cursor interface. Actions are to move the cursor up, down,
// left, or right. Or to pass or place a stone in the current cursor position.
// https://en.wikipedia.org/wiki/Go_(game)
//
// Parameters:
//  "komi"              float  (default 7.5) compensation for white
//  "board_size"        int    (default 19)  rows of the board
//  "handicap"          int    (default 0)   number of handicap stones for black
//  "max_cursor_moves"  int    (default 100) maximum number of cursor moves
//                             before a player must pass or play.
//
// Handicap stones assume a 19x19 board.

namespace open_spiel {
namespace cursor_go {

using go::GoBoard;
using go::GoColor;

// Actions
enum CursorGoAction : Action {
  kActionUp,
  kActionDown,
  kActionLeft,
  kActionRight,
  kActionPlaceStone,
  kActionPass
};

// Constants.
inline constexpr int kNumPlayers = 2;
inline constexpr double kLossUtility = -1;
inline constexpr double kWinUtility = 1;
inline constexpr int kCellStates = 3;

// Go can only end in a draw when using a round komi.
// We also treat superko as a draw.
inline constexpr double kDrawUtility = 0;

// All actions must be in [0; NumDistinctActions).
inline constexpr int kNumDistinctActions = kActionPass + 1;

// In theory Go games have no length limit, but we limit them to twice the
// number of points on the board for practicality - only random games last
// this long.
// The maximum number of cursor go moves is greater by a factor of
// (1+max_cursor_moves).
inline int MaxGameLength(int board_size) { return board_size * board_size * 2; }

inline int ColorToPlayer(GoColor c) { return static_cast<int>(c); }

// State of an in-play game.
class CursorGoState : public State {
 public:
  // Constructs a Go state for the empty board.
  CursorGoState(std::shared_ptr<const Game> game, int board_size, float komi,
                int handicap, int max_cursor_moves);

  Player CurrentPlayer() const override {
    return is_terminal_ ? kTerminalPlayerId : ColorToPlayer(to_play_);
  }
  std::vector<Action> LegalActions() const override;
  std::string ActionToString(Player player, Action action) const override;
  std::string ToString() const override;

  bool IsTerminal() const override { return is_terminal_; }

  std::string InformationStateString(Player player) const override;
  std::string ObservationString(Player player) const override;

  // Five planes: black, white, empty, cursor position, and a bias plane of bits
  // indicating komi (whether white is to play).
  void ObservationTensor(Player player,
                         absl::Span<float> values) const override;

  std::vector<double> Returns() const override;

  std::unique_ptr<State> Clone() const override;

  const GoBoard& board() const { return board_; }

 protected:
  void DoApplyAction(Action action) override;

 private:
  void ResetBoard();

  GoBoard board_;

  // RepetitionTable records which positions we have already encountered.
  // We are already indexing by board hash, so there is no need to hash that
  // hash again, so we use a custom passthrough hasher.
  class PassthroughHash {
   public:
    std::size_t operator()(uint64_t x) const {
      return static_cast<std::size_t>(x);
    }
  };
  using RepetitionTable = std::unordered_set<uint64_t, PassthroughHash>;
  RepetitionTable repetitions_;

  const float komi_;
  const int handicap_;
  const int max_cursor_moves_;
  GoColor to_play_;
  int cursor_moves_count_;
  bool superko_;
  bool last_move_was_pass_;
  bool is_terminal_;
  std::array<std::pair<int, int>, kNumPlayers> cursor_;
};

// Game object.
class CursorGoGame : public Game {
 public:
  explicit CursorGoGame(const GameParameters& params);

  int NumDistinctActions() const override { return kNumDistinctActions; }

  std::unique_ptr<State> NewInitialState() const override {
    return std::unique_ptr<State>(new CursorGoState(
        shared_from_this(), board_size_, komi_, handicap_, max_cursor_moves_));
  }

  std::vector<int> ObservationTensorShape() const override {
    // Planes: black, white, empty, cursor position, and bias planes indicating
    // komi (whether white is to play) and the number of cursor moves made.
    return {kCellStates + 3, board_size_, board_size_};
  }

  int NumPlayers() const override { return kNumPlayers; }

  double MinUtility() const override { return kLossUtility; }
  double UtilitySum() const override { return kLossUtility + kWinUtility; }
  double MaxUtility() const override { return kWinUtility; }

  int MaxGameLength() const override {
    return cursor_go::MaxGameLength(board_size_) * (1 + max_cursor_moves_);
  }

 private:
  const float komi_;
  const int board_size_;
  const int handicap_;
  const int max_cursor_moves_;
};

}  // namespace cursor_go
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_GO_H_
