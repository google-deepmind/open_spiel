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

#ifndef OPEN_SPIEL_GAMES_BANQI_H_
#define OPEN_SPIEL_GAMES_BANQI_H_

#include <array>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/types/optional.h"
#include "absl/types/span.h"
#include "open_spiel/game_parameters.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_globals.h"
#include "open_spiel/spiel_utils.h"

// Banqi (Chinese Dark Chess / Half Chess):
// A two-player imperfect-information board game played on a 4x8 grid.
// All 32 pieces start face-down; players flip, move, and capture pieces.
// Uses the Taiwanese ruleset: piece hierarchy with General-Soldier circular
// capture and cannon jump captures. See https://en.wikipedia.org/wiki/Banqi

namespace open_spiel {
namespace banqi {

inline constexpr int kNumPlayers = 2;
inline constexpr int kNumRows = 4;
inline constexpr int kNumCols = 8;
inline constexpr int kNumSquares = kNumRows * kNumCols;  // 32
inline constexpr int kNumPieceTypes = 7;
inline constexpr int kNumColors = 2;
inline constexpr int kNumDistinctPieces = kNumPieceTypes * kNumColors;  // 14

// Piece types (rank order for capture: lower index = higher rank).
// Red pieces: 0-6, Black pieces: 7-13.
inline constexpr int kGeneral = 0;
inline constexpr int kAdvisor = 1;
inline constexpr int kElephant = 2;
inline constexpr int kChariot = 3;
inline constexpr int kHorse = 4;
inline constexpr int kCannon = 5;
inline constexpr int kSoldier = 6;

inline constexpr int kRed = 0;
inline constexpr int kBlack = 1;

inline constexpr int kEmpty = -1;
inline constexpr int kFaceDown = -2;
inline constexpr int kNoColor = -1;

// Action encoding:
//   Flip:         action = square (0..31)
//   Move/Capture: action = 32 + source*32 + dest
// Total distinct actions = 32 + 32*32 = 1056
inline constexpr int kNumFlipActions = kNumSquares;
inline constexpr int kNumMoveActions = kNumSquares * kNumSquares;
inline constexpr int kNumDistinctActions = kNumFlipActions + kNumMoveActions;

// Observation: 16 channels x 32 squares + 2 (current player one-hot)
// Channels: 14 piece types + empty + face-down
inline constexpr int kNumObsChannels = kNumDistinctPieces + 2;  // 16
inline constexpr int kObsTensorSize =
    kNumObsChannels * kNumSquares + kNumPlayers;  // 514

// Default draw rule: 50 moves without flip or capture.
inline constexpr int kDefaultMaxNoProgress = 50;

// Initial piece counts per type.
inline const std::array<int, kNumDistinctPieces> kInitialPieceCounts = {
    1, 2, 2, 2, 2, 2, 5,  // Red
    1, 2, 2, 2, 2, 2, 5   // Black
};

inline int PieceColor(int piece) { return piece / kNumPieceTypes; }
inline int PieceType(int piece) { return piece % kNumPieceTypes; }
inline int MakePiece(int color, int type) {
  return color * kNumPieceTypes + type;
}

class BanqiState : public State {
 public:
  explicit BanqiState(std::shared_ptr<const Game> game, int max_no_progress);

  BanqiState(const BanqiState&) = default;
  BanqiState& operator=(const BanqiState&) = default;

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
  std::vector<std::pair<Action, double>> ChanceOutcomes() const override;

  // Adjacency-based capture check (returns false for cannon attackers, since
  // cannons capture by jumping). Public for testing.
  bool CanCapture(int attacker, int defender) const;

 protected:
  void DoApplyAction(Action action) override;

 private:
  void AddCannonCaptures(int source, std::vector<Action>* moves) const;
  bool HasLegalMovesForPlayer(Player player) const;
  void CheckTerminal();

  std::array<int, kNumSquares> board_;
  std::array<int, kNumDistinctPieces> remaining_pieces_;
  int num_face_down_ = kNumSquares;
  std::array<int, kNumPlayers> player_color_;
  Player current_player_ = 0;
  Player outcome_ = kInvalidPlayer;
  bool is_draw_ = false;
  int pending_flip_square_ = -1;
  Player player_before_chance_ = kInvalidPlayer;
  int no_progress_count_ = 0;
  int total_moves_ = 0;
  int max_no_progress_;

  struct UndoInfo {
    int captured_piece;
    int no_progress_before;
    std::array<int, kNumPlayers> player_color_before;
    bool was_draw;
    Player outcome_before;
  };
  std::vector<UndoInfo> undo_stack_;
};

class BanqiGame : public Game {
 public:
  explicit BanqiGame(const GameParameters& params);

  int NumDistinctActions() const override { return kNumDistinctActions; }
  std::unique_ptr<State> NewInitialState() const override {
    return std::unique_ptr<State>(
        new BanqiState(shared_from_this(), max_no_progress_));
  }
  int NumPlayers() const override { return kNumPlayers; }
  double MinUtility() const override { return -1; }
  double MaxUtility() const override { return 1; }
  absl::optional<double> UtilitySum() const override { return 0; }
  std::vector<int> ObservationTensorShape() const override {
    return {kObsTensorSize};
  }
  int MaxGameLength() const override;
  int MaxChanceOutcomes() const override { return kNumDistinctPieces; }
  int MaxChanceNodesInHistory() const override { return kNumSquares; }

 private:
  int max_no_progress_;
};

}  // namespace banqi
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_BANQI_H_
