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

#ifndef OPEN_SPIEL_GAMES_SHOGI_H_
#define OPEN_SPIEL_GAMES_SHOGI_H_

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "open_spiel/abseil-cpp/absl/types/optional.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

// Game of Shogi (Japanese Chess):
// https://en.wikipedia.org/wiki/Shogi
//
// Parameters: none

namespace open_spiel {
namespace shogi {

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

inline constexpr int kNumPlayers = 2;
inline constexpr int kBoardSize = 9;
inline constexpr int kNumSquares = kBoardSize * kBoardSize;  // 81

enum PieceType : int8_t {
  kEmpty = 0,
  kPawn = 1,
  kLance = 2,
  kKnight = 3,
  kSilver = 4,
  kGold = 5,
  kBishop = 6,
  kRook = 7,
  kKing = 8,
  kNumPieceTypes = 9,
};

// Piece types that can be held in hand and dropped (King excluded).
inline constexpr int kNumDropPieceTypes = 7;
inline constexpr PieceType kDropIndexToType[kNumDropPieceTypes] = {
    kPawn, kLance, kKnight, kSilver, kGold, kBishop, kRook};

struct Piece {
  PieceType type = kEmpty;
  int player = -1;       // 0 = Black (sente), 1 = White (gote), -1 = empty
  bool promoted = false;
  bool IsEmpty() const { return type == kEmpty; }
};

// ---------------------------------------------------------------------------
// Action encoding
// ---------------------------------------------------------------------------
//
// Normal move:     from_sq * 81 + to_sq                     [0,      6560]
// Promotion move:  from_sq * 81 + to_sq + kPromotionOffset  [6561,  13121]
// Drop move:       kDropOffset + piece_index * 81 + to_sq   [13122, 13688]
//
// piece_index for drops:
//   Pawn=0, Lance=1, Knight=2, Silver=3, Gold=4, Bishop=5, Rook=6

inline constexpr int kPromotionOffset = kNumSquares * kNumSquares;  // 6561
inline constexpr int kDropOffset = 2 * kNumSquares * kNumSquares;   // 13122
inline constexpr int kNumDistinctActions =
    kDropOffset + kNumDropPieceTypes * kNumSquares;                  // 13689

inline constexpr int kMaxGameLength = 512;

inline constexpr double kLossUtility = -1.0;
inline constexpr double kDrawUtility = 0.0;
inline constexpr double kWinUtility = 1.0;

// ---------------------------------------------------------------------------
// Observation tensor
// ---------------------------------------------------------------------------
//
// 43 planes of 9x9:
//   28 piece planes (2 players x 14 piece states):
//       Per player: 8 unpromoted (P L N S G B R K) + 6 promoted (+P +L +N +S +B +R)
//   1 current-player plane
//   14 hand planes (2 players x 7 droppable types, count / 18.0)

inline constexpr int kNumPiecePlanes = 28;
inline constexpr int kNumObservationPlanes =
    kNumPiecePlanes + 1 + 2 * kNumDropPieceTypes;  // 43

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

int SquareIndex(int row, int col);
std::pair<int, int> IndexToRowCol(int index);

Action EncodeMove(int from_sq, int to_sq);
Action EncodeMoveWithPromotion(int from_sq, int to_sq);
Action EncodeDrop(int piece_index, int to_sq);

// ---------------------------------------------------------------------------
// ShogiState
// ---------------------------------------------------------------------------

class ShogiState : public State {
 public:
  explicit ShogiState(std::shared_ptr<const Game> game);
  ShogiState(const ShogiState&) = default;
  ShogiState& operator=(const ShogiState&) = default;

  Player CurrentPlayer() const override;
  std::vector<Action> LegalActions() const override;
  std::string ActionToString(Player player, Action action_id) const override;
  std::string ToString() const override;
  bool IsTerminal() const override;
  std::vector<double> Returns() const override;
  std::string InformationStateString(Player player) const override;
  std::string ObservationString(Player player) const override;
  void ObservationTensor(Player player,
                         absl::Span<float> values) const override;
  std::unique_ptr<State> Clone() const override;
  void UndoAction(Player player, Action action) override;

 protected:
  void DoApplyAction(Action action) override;

 private:
  struct UndoInfo {
    Action action;
    Piece captured;
    bool was_promoted;
    int from_sq;
    int to_sq;
  };

  void SetupInitialPosition();
  void MaybeGenerateLegalActions() const;
  void GenerateMoveMoves(std::vector<Action>* actions) const;
  void GenerateDropMoves(std::vector<Action>* actions,
                         bool check_pawn_drop_mate = true) const;
  void GeneratePieceMoves(int from_sq, std::vector<Action>* actions) const;

  bool InBounds(int row, int col) const;
  bool IsInPromotionZone(int sq, int player) const;
  bool MustPromote(int sq, PieceType type, int player) const;
  bool CanPromote(int from_sq, int to_sq, int player) const;

  bool IsInCheck(int player) const;
  bool SquareAttackedBy(int sq, int attacker) const;
  int FindKing(int player) const;

  bool WouldLeaveInCheck(int from_sq, int to_sq, bool promote) const;
  bool DropWouldLeaveInCheck(PieceType type, int to_sq) const;
  bool IsPawnDropCheckmate(int to_sq) const;
  bool HasPawnOnFile(int player, int col) const;

  int PieceTypeToDropIndex(PieceType type) const;

  std::array<Piece, kNumSquares> board_;
  std::array<std::array<int, kNumDropPieceTypes>, kNumPlayers> pieces_in_hand_;
  int current_player_ = 0;
  int outcome_ = -1;  // -1 = ongoing, 0 = black wins, 1 = white wins, 2 = draw
  int num_moves_ = 0;
  std::vector<UndoInfo> undo_stack_;
  mutable absl::optional<std::vector<Action>> cached_legal_actions_;
};

// ---------------------------------------------------------------------------
// ShogiGame
// ---------------------------------------------------------------------------

class ShogiGame : public Game {
 public:
  explicit ShogiGame(const GameParameters& params);
  int NumDistinctActions() const override { return kNumDistinctActions; }
  std::unique_ptr<State> NewInitialState() const override {
    return std::unique_ptr<State>(new ShogiState(shared_from_this()));
  }
  int NumPlayers() const override { return kNumPlayers; }
  double MinUtility() const override { return kLossUtility; }
  absl::optional<double> UtilitySum() const override { return kDrawUtility; }
  double MaxUtility() const override { return kWinUtility; }
  std::vector<int> ObservationTensorShape() const override {
    return {kNumObservationPlanes, kBoardSize, kBoardSize};
  }
  int MaxGameLength() const override { return kMaxGameLength; }
};

}  // namespace shogi
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_SHOGI_H_
