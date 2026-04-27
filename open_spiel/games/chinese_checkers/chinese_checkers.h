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

#ifndef OPEN_SPIEL_GAMES_CHINESE_CHECKERS_H_
#define OPEN_SPIEL_GAMES_CHINESE_CHECKERS_H_

#include <array>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "open_spiel/abseil-cpp/absl/types/optional.h"
#include "open_spiel/abseil-cpp/absl/types/span.h"
#include "open_spiel/game_parameters.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_globals.h"
#include "open_spiel/spiel_utils.h"

// Chinese Checkers:
// A multi-player perfect-information board game played on a 6-pointed star
// (121 positions). Players race to move all 10 pieces from their home triangle
// to the opposite triangle. Pieces move one step to an adjacent cell or hop
// over adjacent pieces (chain hops allowed).
// See https://en.wikipedia.org/wiki/Chinese_checkers

namespace open_spiel {
namespace chinese_checkers {

inline constexpr int kNumPositions = 121;
inline constexpr int kNumDirections = 6;
inline constexpr int kNumTriangles = 6;
inline constexpr int kTriangleSize = 10;
inline constexpr int kMaxNumPlayers = 6;
inline constexpr int kDefaultNumPlayers = 2;
inline constexpr int kDefaultMaxMoves = 1000;
inline constexpr int kNumRows = 17;

// Action encoding:
//   Move/Hop: position * 6 + direction  (0..725)
//   Pass:     726 (end hop chain)
inline constexpr int kPassAction = kNumPositions * kNumDirections;  // 726
inline constexpr int kNumDistinctActions = kPassAction + 1;        // 727

inline constexpr int kEmpty = -1;

// Direction offsets in doubled coordinates: {row_delta, col_delta}.
inline constexpr int kDirRowOffset[kNumDirections] = {-1, -1, 0, 0, 1, 1};
inline constexpr int kDirColOffset[kNumDirections] = {-1, 1, -2, 2, -1, 1};

// Precomputed board topology (defined in .cc).
extern const int kCellRow[kNumPositions];
extern const int kCellCol[kNumPositions];
extern const int kNeighbor[kNumPositions][kNumDirections];
extern const int kHopDest[kNumPositions][kNumDirections];
extern const int kTriangleCells[kNumTriangles][kTriangleSize];

// Target triangle for each starting triangle: (s + 3) % 6.
inline int TargetTriangle(int home) { return (home + 3) % kNumTriangles; }

// Player slot assignments per player count.
// Maps (num_players, player_index) -> triangle index.
const std::vector<int>& PlayerSlots(int num_players);

struct UndoInfo {
  int piece_origin;
  int piece_dest;
  int prev_hop_from;
  Player prev_current_player;
  int prev_total_moves;
  Player prev_outcome;
  std::set<int> prev_visited;
};

class ChineseCheckersState : public State {
 public:
  ChineseCheckersState(std::shared_ptr<const Game> game, int num_players,
                       int max_moves);
  ChineseCheckersState(const ChineseCheckersState&) = default;
  ChineseCheckersState& operator=(const ChineseCheckersState&) = default;

  Player CurrentPlayer() const override;
  std::vector<Action> LegalActions() const override;
  std::string ActionToString(Player player, Action action_id) const override;
  std::string ToString() const override;
  bool IsTerminal() const override;
  std::vector<double> Returns() const override;
  std::string ObservationString(Player player) const override;
  void ObservationTensor(Player player,
                         absl::Span<float> values) const override;
  std::unique_ptr<State> Clone() const override;
  void UndoAction(Player player, Action action) override;

  // For testing.
  int BoardAt(int pos) const { return board_[pos]; }
  void SetBoard(int pos, int owner) { board_[pos] = owner; }

 protected:
  void DoApplyAction(Action action) override;

 private:
  bool CheckWinner(Player player) const;
  void AdvanceTurn();
  bool HasContinuationHops(int pos) const;

  int num_players_;
  int max_moves_;
  Player current_player_ = 0;
  Player outcome_ = kInvalidPlayer;
  int total_moves_ = 0;

  // Hop chain state.
  int hop_from_ = -1;
  std::set<int> visited_;

  std::array<int, kNumPositions> board_;
  std::vector<int> player_slots_;
  std::vector<UndoInfo> undo_stack_;
};

class ChineseCheckersGame : public Game {
 public:
  explicit ChineseCheckersGame(const GameParameters& params);

  int NumDistinctActions() const override { return kNumDistinctActions; }
  std::unique_ptr<State> NewInitialState() const override {
    return std::unique_ptr<State>(
        new ChineseCheckersState(shared_from_this(), num_players_, max_moves_));
  }
  int NumPlayers() const override { return num_players_; }
  double MinUtility() const override { return -1; }
  double MaxUtility() const override { return num_players_ - 1; }
  absl::optional<double> UtilitySum() const override { return 0; }
  std::vector<int> ObservationTensorShape() const override {
    return {(num_players_ + 1) * kNumPositions + num_players_};
  }
  int MaxGameLength() const override {
    // Each completed turn is 1+ actions: step/pass is 1, hop chains add more.
    // Max hop chain length bounded by number of pieces on the board.
    return max_moves_ * (num_players_ * kTriangleSize + 2);
  }

 private:
  int num_players_;
  int max_moves_;
};

}  // namespace chinese_checkers
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_CHINESE_CHECKERS_H_
