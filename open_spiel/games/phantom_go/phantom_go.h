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

#ifndef OPEN_SPIEL_GAMES_PHANTOM_GO_H_
#define OPEN_SPIEL_GAMES_PHANTOM_GO_H_

#include <array>
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include "open_spiel/games/phantom_go/phantom_go_board.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

// Game of Phantom Go:
// https://www.chessprogramming.org/Phantom_Go
//
// Parameters:
//  "komi"              float  compensation for white                  (default
//  = 7.5) "board_size"        int    rows of the board, usually 9, 13 or 19
//  (default = 9) "handicap"          int    number of handicap stones for black
//  (default = 0) "max_game_length"   int    maximal lenght of a game (default =
//  board_size * board_size * 4)

namespace open_spiel {
namespace phantom_go {

class PhantomGoObserver;

// Constants.
inline constexpr int NumPlayers() { return 2; }
inline constexpr double LossUtility() { return -1; }
inline constexpr double WinUtility() { return 1; }
inline constexpr int CellStates() { return 3; }  // Black, white, empty.

// Go can only end in a draw when using a round komi.
// We also treat superko as a draw.
inline constexpr double DrawUtility() { return 0; }

// All actions must be in [0; NumDistinctActions).
inline int NumDistinctActions(int board_size) {
  return board_size * board_size + 1;
}

// Such high number has been set, mainly because moves on enemy stones are also
// counted into length And for "clear" resampling, lot of passes and
// "observation moves" are needed
inline int DefaultMaxGameLength(int board_size) {
  return board_size * board_size * 4;
}

inline int MaxGameLength(int board_size) { return board_size * board_size * 4; }

inline int ColorToPlayer(GoColor c) { return static_cast<int>(c); }
inline GoColor PlayerToColor(Player p) { return static_cast<GoColor>(p); }

// State of an in-play game.
// Actions are contiguous from 0 to board_size * board_size - 1, row-major, i.e.
// the (row, col) action is encoded as row * board_size + col.
// The pass action is board_size * board_size.
class PhantomGoState : public State {
 public:
  // Constructs a Go state for the empty board.
  PhantomGoState(std::shared_ptr<const Game> game, int board_size, float komi,
                 int handicap);

  Player CurrentPlayer() const override {
    return IsTerminal() ? kTerminalPlayerId : ColorToPlayer(to_play_);
  }
  std::vector<Action> LegalActions() const override;

  std::array<int, 2> GetStoneCount() const;

  int GetMaxGameLenght() const;

  static bool equalMetaposition(const PhantomGoState& state1,
                                const PhantomGoState& state2, int playerID);

  std::string ActionToString(Player player, Action action) const override;
  std::string ToString() const override;

  bool IsTerminal() const override;

  std::string ObservationString(int player) const override;

  // Four planes: black, white, empty, and a bias plane of bits indicating komi
  // (whether white is to play).
  void ObservationTensor(int player, absl::Span<float> values) const override;

  std::vector<double> Returns() const override;

  std::unique_ptr<State> Clone() const override;
  void UndoAction(Player player, Action action) override;

  const PhantomGoBoard& board() const { return board_; }

 protected:
  void DoApplyAction(Action action) override;

 private:
  void ResetBoard();

  PhantomGoBoard board_;

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
  const int max_game_length_;
  GoColor to_play_;
  bool superko_;
};

class PhantomGoGame : public Game {
 public:
  explicit PhantomGoGame(const GameParameters& params);

  std::shared_ptr<PhantomGoObserver> default_observer_;

  int NumDistinctActions() const override {
    return phantom_go::NumDistinctActions(board_size_);
  }

  std::unique_ptr<State> NewInitialState() const override {
    return std::unique_ptr<State>(
        new PhantomGoState(shared_from_this(), board_size_, komi_, handicap_));
  }

  std::vector<int> ObservationTensorShape() const override {
    // Planes: black, white, empty, and a bias plane indicating komi (whether
    // white is to play)
    // and 2 for stone count of white and black
    return {2 + board_size_ * board_size_ * (CellStates() + 1)};
  }

  TensorLayout ObservationTensorLayout() const override {
    return TensorLayout::kCHW;
  }

  int NumPlayers() const override { return phantom_go::NumPlayers(); }

  double MinUtility() const override { return LossUtility(); }
  absl::optional<double> UtilitySum() const override {
    return LossUtility() + WinUtility();
  }
  double MaxUtility() const override { return WinUtility(); }

  int MaxGameLength() const override { return max_game_length_; }

 private:
  const float komi_;
  const int board_size_;
  const int handicap_;
  const int max_game_length_;
};

}  // namespace phantom_go
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_PHANTOM_GO_H_
