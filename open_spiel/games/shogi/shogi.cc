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

#include "open_spiel/games/shogi/shogi.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/abseil-cpp/absl/types/optional.h"
#include "open_spiel/abseil-cpp/absl/types/span.h"
#include "open_spiel/observer.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_globals.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/utils/tensor_view.h"

namespace open_spiel {
namespace shogi {
namespace {

// ---------------------------------------------------------------------------
// Game registration
// ---------------------------------------------------------------------------

const GameType kGameType{
    /*short_name=*/"shogi",
    /*long_name=*/"Shogi",
    GameType::Dynamics::kSequential,
    GameType::ChanceMode::kDeterministic,
    GameType::Information::kPerfectInformation,
    GameType::Utility::kZeroSum,
    GameType::RewardModel::kTerminal,
    /*max_num_players=*/2,
    /*min_num_players=*/2,
    /*provides_information_state_string=*/true,
    /*provides_information_state_tensor=*/false,
    /*provides_observation_string=*/true,
    /*provides_observation_tensor=*/true,
    /*parameter_specification=*/{}};

std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::shared_ptr<const Game>(new ShogiGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);

RegisterSingleTensorObserver single_tensor(kGameType.short_name);

// ---------------------------------------------------------------------------
// String helpers
// ---------------------------------------------------------------------------

// Uppercase = Black (player 0), lowercase = White (player 1).
const char* kPieceChars = ".PLNSGBRKplnsgbrk";

char PieceToChar(const Piece& p) {
  if (p.IsEmpty()) return '.';
  return kPieceChars[(p.player == 0 ? 0 : 8) + static_cast<int>(p.type)];
}

std::string PieceToString(const Piece& p) {
  if (p.IsEmpty()) return " . ";
  std::string s;
  if (p.promoted) {
    s += "+";
  } else {
    s += " ";
  }
  s += PieceToChar(p);
  s += " ";
  return s;
}

std::string PieceTypeName(PieceType type) {
  switch (type) {
    case kPawn:   return "P";
    case kLance:  return "L";
    case kKnight: return "N";
    case kSilver: return "S";
    case kGold:   return "G";
    case kBishop: return "B";
    case kRook:   return "R";
    case kKing:   return "K";
    default:      return "?";
  }
}

std::string SquareToString(int sq) {
  auto [row, col] = IndexToRowCol(sq);
  return absl::StrCat(std::string(1, 'a' + col), row + 1);
}

// ---------------------------------------------------------------------------
// Direction tables
// ---------------------------------------------------------------------------

struct Direction { int dr; int dc; };

constexpr Direction kAllDirs[] = {
    {-1, -1}, {-1, 0}, {-1, 1}, {0, -1}, {0, 1}, {1, -1}, {1, 0}, {1, 1}};
constexpr Direction kOrthogonal[] = {{-1, 0}, {0, -1}, {0, 1}, {1, 0}};
constexpr Direction kDiagonal[] = {{-1, -1}, {-1, 1}, {1, -1}, {1, 1}};

// Black (player 0) moves toward lower rows; White toward higher rows.
int ForwardDir(int player) { return (player == 0) ? -1 : 1; }

// Returns the plane index (0..5) for a promoted piece in the observation
// tensor. Only the six promotable types are valid inputs.
int PromotedPlaneIndex(PieceType t) {
  switch (t) {
    case kPawn:   return 0;
    case kLance:  return 1;
    case kKnight: return 2;
    case kSilver: return 3;
    case kBishop: return 4;
    case kRook:   return 5;
    default:      return -1;
  }
}

}  // namespace

// ---------------------------------------------------------------------------
// Public utility functions
// ---------------------------------------------------------------------------

int SquareIndex(int row, int col) { return row * kBoardSize + col; }

std::pair<int, int> IndexToRowCol(int index) {
  return {index / kBoardSize, index % kBoardSize};
}

Action EncodeMove(int from_sq, int to_sq) {
  return from_sq * kNumSquares + to_sq;
}

Action EncodeMoveWithPromotion(int from_sq, int to_sq) {
  return from_sq * kNumSquares + to_sq + kPromotionOffset;
}

Action EncodeDrop(int piece_index, int to_sq) {
  return kDropOffset + piece_index * kNumSquares + to_sq;
}

// ===========================================================================
// ShogiState implementation
// ===========================================================================

ShogiState::ShogiState(std::shared_ptr<const Game> game) : State(game) {
  SetupInitialPosition();
}

// ---------------------------------------------------------------------------
// Initial position
// ---------------------------------------------------------------------------

void ShogiState::SetupInitialPosition() {
  for (auto& p : board_) p = Piece{kEmpty, -1, false};
  for (auto& hand : pieces_in_hand_) hand.fill(0);

  // Standard Shogi starting position.
  // Row 0 is White's back rank (top); row 8 is Black's back rank (bottom).
  const PieceType back_rank[] = {kLance, kKnight, kSilver, kGold, kKing,
                                 kGold, kSilver, kKnight, kLance};
  for (int c = 0; c < kBoardSize; ++c) {
    board_[SquareIndex(0, c)] = Piece{back_rank[c], 1, false};  // White
    board_[SquareIndex(8, c)] = Piece{back_rank[c], 0, false};  // Black
    board_[SquareIndex(2, c)] = Piece{kPawn, 1, false};          // White pawns
    board_[SquareIndex(6, c)] = Piece{kPawn, 0, false};          // Black pawns
  }
  board_[SquareIndex(1, 1)] = Piece{kRook, 1, false};
  board_[SquareIndex(1, 7)] = Piece{kBishop, 1, false};
  board_[SquareIndex(7, 1)] = Piece{kBishop, 0, false};
  board_[SquareIndex(7, 7)] = Piece{kRook, 0, false};

  current_player_ = 0;
  outcome_ = -1;
  num_moves_ = 0;
}

// ---------------------------------------------------------------------------
// Core State interface
// ---------------------------------------------------------------------------

Player ShogiState::CurrentPlayer() const {
  return IsTerminal() ? kTerminalPlayerId : current_player_;
}

bool ShogiState::InBounds(int row, int col) const {
  return row >= 0 && row < kBoardSize && col >= 0 && col < kBoardSize;
}

// ---------------------------------------------------------------------------
// Promotion logic
// ---------------------------------------------------------------------------
// Promotion zone = opponent's last 3 ranks.
// Pawn/Lance must promote on the last rank; Knight must promote on the last
// two ranks. All other promotable pieces may optionally promote when moving
// into, out of, or within the promotion zone.
// Gold and King never promote.

bool ShogiState::IsInPromotionZone(int sq, int player) const {
  int row = sq / kBoardSize;
  return (player == 0) ? (row <= 2) : (row >= 6);
}

bool ShogiState::MustPromote(int sq, PieceType type, int player) const {
  int row = sq / kBoardSize;
  if (player == 0) {
    if (type == kPawn || type == kLance) return row == 0;
    if (type == kKnight) return row <= 1;
  } else {
    if (type == kPawn || type == kLance) return row == 8;
    if (type == kKnight) return row >= 7;
  }
  return false;
}

bool ShogiState::CanPromote(int from_sq, int to_sq, int player) const {
  return IsInPromotionZone(from_sq, player) ||
         IsInPromotionZone(to_sq, player);
}

// ---------------------------------------------------------------------------
// King / check helpers
// ---------------------------------------------------------------------------

int ShogiState::FindKing(int player) const {
  for (int i = 0; i < kNumSquares; ++i) {
    if (board_[i].type == kKing && board_[i].player == player) return i;
  }
  return -1;
}

// Returns true if `sq` is attacked by any piece belonging to `attacker`.
// Tests every piece of the attacker for reachability to `sq`, including
// sliding-piece obstruction checks.
bool ShogiState::SquareAttackedBy(int sq, int attacker) const {
  auto [tr, tc] = IndexToRowCol(sq);
  int fwd = ForwardDir(attacker);

  for (int i = 0; i < kNumSquares; ++i) {
    const Piece& p = board_[i];
    if (p.IsEmpty() || p.player != attacker) continue;
    auto [fr, fc] = IndexToRowCol(i);
    int dr = tr - fr;
    int dc = tc - fc;

    if (p.type == kKing) {
      if (std::abs(dr) <= 1 && std::abs(dc) <= 1 && (dr != 0 || dc != 0))
        return true;
      continue;
    }

    // Helper: check that a sliding path from (fr,fc) to (tr,tc) is clear.
    auto PathClear = [&](int step_r, int step_c) -> bool {
      int r = fr + step_r, c = fc + step_c;
      while (r != tr || c != tc) {
        if (!board_[SquareIndex(r, c)].IsEmpty()) return false;
        r += step_r;
        c += step_c;
      }
      return true;
    };

    // Gold-movement predicate (used by Gold and all promoted minor pieces).
    auto GoldReaches = [&]() -> bool {
      if (std::abs(dr) > 1 || std::abs(dc) > 1 || (dr == 0 && dc == 0))
        return false;
      return dr == fwd || dr == 0 || (dr == -fwd && dc == 0);
    };

    if (!p.promoted) {
      switch (p.type) {
        case kPawn:
          if (dr == fwd && dc == 0) return true;
          break;
        case kLance:
          if (dc == 0 && dr * fwd > 0 && PathClear(fwd, 0)) return true;
          break;
        case kKnight:
          if (dr == 2 * fwd && std::abs(dc) == 1) return true;
          break;
        case kSilver:
          if (std::abs(dr) <= 1 && std::abs(dc) <= 1 &&
              (dr != 0 || dc != 0)) {
            if (dr == fwd || (dr == -fwd && std::abs(dc) == 1)) return true;
          }
          break;
        case kGold:
          if (GoldReaches()) return true;
          break;
        case kBishop:
          if (std::abs(dr) == std::abs(dc) && dr != 0 &&
              PathClear(dr > 0 ? 1 : -1, dc > 0 ? 1 : -1))
            return true;
          break;
        case kRook:
          if ((dr == 0) != (dc == 0) &&
              PathClear(dr == 0 ? 0 : (dr > 0 ? 1 : -1),
                        dc == 0 ? 0 : (dc > 0 ? 1 : -1)))
            return true;
          break;
        default: break;
      }
    } else {
      switch (p.type) {
        case kPawn: case kLance: case kKnight: case kSilver:
          if (GoldReaches()) return true;
          break;
        case kBishop:  // Dragon Horse: Bishop + one-step orthogonal.
          if (std::abs(dr) <= 1 && std::abs(dc) <= 1 && (dr || dc))
            return true;
          if (std::abs(dr) == std::abs(dc) && dr != 0 &&
              PathClear(dr > 0 ? 1 : -1, dc > 0 ? 1 : -1))
            return true;
          break;
        case kRook:  // Dragon King: Rook + one-step diagonal.
          if (std::abs(dr) <= 1 && std::abs(dc) <= 1 && (dr || dc))
            return true;
          if ((dr == 0) != (dc == 0) &&
              PathClear(dr == 0 ? 0 : (dr > 0 ? 1 : -1),
                        dc == 0 ? 0 : (dc > 0 ? 1 : -1)))
            return true;
          break;
        default: break;
      }
    }
  }
  return false;
}

bool ShogiState::IsInCheck(int player) const {
  int king_sq = FindKing(player);
  if (king_sq < 0) return true;
  return SquareAttackedBy(king_sq, 1 - player);
}

bool ShogiState::WouldLeaveInCheck(int from_sq, int to_sq,
                                    bool promote) const {
  ShogiState copy(*this);
  copy.board_[to_sq] = copy.board_[from_sq];
  copy.board_[from_sq] = Piece{kEmpty, -1, false};
  if (promote) copy.board_[to_sq].promoted = true;
  return copy.IsInCheck(current_player_);
}

bool ShogiState::DropWouldLeaveInCheck(PieceType type, int to_sq) const {
  ShogiState copy(*this);
  copy.board_[to_sq] = Piece{type, current_player_, false};
  return copy.IsInCheck(current_player_);
}

// ---------------------------------------------------------------------------
// Pawn-drop helpers
// ---------------------------------------------------------------------------
// Shogi forbids dropping a pawn on a file that already contains an unpromoted
// pawn of the same player. It also forbids dropping a pawn to give immediate
// checkmate (uchifuzume).

bool ShogiState::HasPawnOnFile(int player, int col) const {
  for (int row = 0; row < kBoardSize; ++row) {
    const Piece& p = board_[SquareIndex(row, col)];
    if (p.type == kPawn && p.player == player && !p.promoted) return true;
  }
  return false;
}

bool ShogiState::IsPawnDropCheckmate(int to_sq) const {
  // Place the pawn on a copy, switch sides, then check whether the opponent
  // has any legal response. Uses check_pawn_drop_mate=false when generating
  // opponent drop moves to avoid unbounded recursion.
  ShogiState copy(*this);
  copy.board_[to_sq] = Piece{kPawn, current_player_, false};
  copy.current_player_ = 1 - current_player_;
  copy.cached_legal_actions_.reset();

  if (!copy.IsInCheck(copy.current_player_)) return false;

  std::vector<Action> opp_moves;
  copy.GenerateMoveMoves(&opp_moves);
  copy.GenerateDropMoves(&opp_moves, /*check_pawn_drop_mate=*/false);
  return opp_moves.empty();
}

int ShogiState::PieceTypeToDropIndex(PieceType type) const {
  for (int i = 0; i < kNumDropPieceTypes; ++i) {
    if (kDropIndexToType[i] == type) return i;
  }
  return -1;
}

// ---------------------------------------------------------------------------
// Move generation — board moves
// ---------------------------------------------------------------------------

void ShogiState::GeneratePieceMoves(int from_sq,
                                     std::vector<Action>* actions) const {
  const Piece& p = board_[from_sq];
  if (p.IsEmpty() || p.player != current_player_) return;

  auto [fr, fc] = IndexToRowCol(from_sq);
  int fwd = ForwardDir(current_player_);

  // Try to add a move (with optional / forced promotion) after validating
  // bounds, friendly-fire, and king safety.
  auto TryAddMove = [&](int tr, int tc) {
    if (!InBounds(tr, tc)) return;
    int to_sq = SquareIndex(tr, tc);
    if (!board_[to_sq].IsEmpty() && board_[to_sq].player == current_player_)
      return;

    bool promotable = !p.promoted && p.type != kGold && p.type != kKing &&
                      CanPromote(from_sq, to_sq, current_player_);
    bool must = !p.promoted && MustPromote(to_sq, p.type, current_player_);

    if (must) {
      if (!WouldLeaveInCheck(from_sq, to_sq, true))
        actions->push_back(EncodeMoveWithPromotion(from_sq, to_sq));
    } else if (promotable) {
      if (!WouldLeaveInCheck(from_sq, to_sq, false))
        actions->push_back(EncodeMove(from_sq, to_sq));
      if (!WouldLeaveInCheck(from_sq, to_sq, true))
        actions->push_back(EncodeMoveWithPromotion(from_sq, to_sq));
    } else {
      if (!WouldLeaveInCheck(from_sq, to_sq, false))
        actions->push_back(EncodeMove(from_sq, to_sq));
    }
  };

  auto SlideMoves = [&](const Direction* dirs, int num_dirs) {
    for (int d = 0; d < num_dirs; ++d) {
      int r = fr + dirs[d].dr, c = fc + dirs[d].dc;
      while (InBounds(r, c)) {
        const Piece& target = board_[SquareIndex(r, c)];
        if (!target.IsEmpty() && target.player == current_player_) break;
        TryAddMove(r, c);
        if (!target.IsEmpty()) break;
        r += dirs[d].dr;
        c += dirs[d].dc;
      }
    }
  };

  // Gold-style step moves (also used by all promoted minor pieces).
  auto GoldMoves = [&]() {
    TryAddMove(fr + fwd, fc - 1);
    TryAddMove(fr + fwd, fc);
    TryAddMove(fr + fwd, fc + 1);
    TryAddMove(fr, fc - 1);
    TryAddMove(fr, fc + 1);
    TryAddMove(fr - fwd, fc);
  };

  if (!p.promoted) {
    switch (p.type) {
      case kPawn:
        TryAddMove(fr + fwd, fc);
        break;
      case kLance:
        for (int r = fr + fwd; InBounds(r, fc); r += fwd) {
          const Piece& t = board_[SquareIndex(r, fc)];
          if (!t.IsEmpty() && t.player == current_player_) break;
          TryAddMove(r, fc);
          if (!t.IsEmpty()) break;
        }
        break;
      case kKnight:
        TryAddMove(fr + 2 * fwd, fc - 1);
        TryAddMove(fr + 2 * fwd, fc + 1);
        break;
      case kSilver:
        TryAddMove(fr + fwd, fc - 1);
        TryAddMove(fr + fwd, fc);
        TryAddMove(fr + fwd, fc + 1);
        TryAddMove(fr - fwd, fc - 1);
        TryAddMove(fr - fwd, fc + 1);
        break;
      case kGold:   GoldMoves(); break;
      case kBishop: SlideMoves(kDiagonal, 4); break;
      case kRook:   SlideMoves(kOrthogonal, 4); break;
      case kKing:
        for (const auto& d : kAllDirs) TryAddMove(fr + d.dr, fc + d.dc);
        break;
      default: break;
    }
  } else {
    switch (p.type) {
      case kPawn: case kLance: case kKnight: case kSilver:
        GoldMoves();
        break;
      case kBishop:  // Dragon Horse
        SlideMoves(kDiagonal, 4);
        for (const auto& d : kOrthogonal) TryAddMove(fr + d.dr, fc + d.dc);
        break;
      case kRook:  // Dragon King
        SlideMoves(kOrthogonal, 4);
        for (const auto& d : kDiagonal) TryAddMove(fr + d.dr, fc + d.dc);
        break;
      default: break;
    }
  }
}

void ShogiState::GenerateMoveMoves(std::vector<Action>* actions) const {
  for (int sq = 0; sq < kNumSquares; ++sq) {
    if (!board_[sq].IsEmpty() && board_[sq].player == current_player_) {
      GeneratePieceMoves(sq, actions);
    }
  }
}

// ---------------------------------------------------------------------------
// Move generation — drops
// ---------------------------------------------------------------------------
// Captured pieces (minus promotion) go into the capturing player's hand and
// may be dropped onto any empty square, subject to:
//   Pawn:   cannot drop on last rank; cannot drop on a file that already
//           has an unpromoted friendly pawn; cannot give immediate checkmate.
//   Lance:  cannot drop on last rank.
//   Knight: cannot drop on last two ranks.
//   All:    cannot drop if it would leave own king in check.

void ShogiState::GenerateDropMoves(std::vector<Action>* actions,
                                    bool check_pawn_drop_mate) const {
  for (int di = 0; di < kNumDropPieceTypes; ++di) {
    if (pieces_in_hand_[current_player_][di] <= 0) continue;
    PieceType type = kDropIndexToType[di];

    for (int sq = 0; sq < kNumSquares; ++sq) {
      if (!board_[sq].IsEmpty()) continue;
      int row = sq / kBoardSize;
      int col = sq % kBoardSize;

      if (type == kPawn) {
        if (current_player_ == 0 && row == 0) continue;
        if (current_player_ == 1 && row == 8) continue;
        if (HasPawnOnFile(current_player_, col)) continue;
        if (DropWouldLeaveInCheck(type, sq)) continue;
        if (check_pawn_drop_mate) {
          ShogiState tmp(*this);
          tmp.board_[sq] = Piece{kPawn, current_player_, false};
          if (tmp.IsInCheck(1 - current_player_)) {
            if (IsPawnDropCheckmate(sq)) continue;
          }
        }
        actions->push_back(EncodeDrop(di, sq));
        continue;
      }

      if (type == kLance) {
        if (current_player_ == 0 && row == 0) continue;
        if (current_player_ == 1 && row == 8) continue;
      }
      if (type == kKnight) {
        if (current_player_ == 0 && row <= 1) continue;
        if (current_player_ == 1 && row >= 7) continue;
      }

      if (!DropWouldLeaveInCheck(type, sq)) {
        actions->push_back(EncodeDrop(di, sq));
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Legal actions (cached)
// ---------------------------------------------------------------------------

void ShogiState::MaybeGenerateLegalActions() const {
  if (cached_legal_actions_.has_value()) return;
  cached_legal_actions_ = std::vector<Action>();
  GenerateMoveMoves(&*cached_legal_actions_);
  GenerateDropMoves(&*cached_legal_actions_);
  std::sort(cached_legal_actions_->begin(), cached_legal_actions_->end());
}

std::vector<Action> ShogiState::LegalActions() const {
  if (IsTerminal()) return {};
  return *cached_legal_actions_;
}

// ---------------------------------------------------------------------------
// String representations
// ---------------------------------------------------------------------------

std::string ShogiState::ActionToString(Player player,
                                        Action action_id) const {
  if (action_id >= kDropOffset) {
    int remainder = action_id - kDropOffset;
    int piece_index = remainder / kNumSquares;
    int to_sq = remainder % kNumSquares;
    return absl::StrCat(PieceTypeName(kDropIndexToType[piece_index]), "*",
                        SquareToString(to_sq));
  }
  bool promote = (action_id >= kPromotionOffset);
  int raw = promote ? action_id - kPromotionOffset : action_id;
  int from_sq = raw / kNumSquares;
  int to_sq = raw % kNumSquares;
  std::string result =
      absl::StrCat(SquareToString(from_sq), SquareToString(to_sq));
  if (promote) result += "+";
  return result;
}

std::string ShogiState::ToString() const {
  std::string result;
  for (int row = 0; row < kBoardSize; ++row) {
    for (int col = 0; col < kBoardSize; ++col) {
      absl::StrAppend(&result, PieceToString(board_[SquareIndex(row, col)]));
    }
    absl::StrAppend(&result, "\n");
  }
  for (int pl = 0; pl < kNumPlayers; ++pl) {
    absl::StrAppend(&result, (pl == 0) ? "Black hand: " : "White hand: ");
    bool any = false;
    for (int di = 0; di < kNumDropPieceTypes; ++di) {
      if (pieces_in_hand_[pl][di] > 0) {
        absl::StrAppend(&result, PieceTypeName(kDropIndexToType[di]),
                        "x", pieces_in_hand_[pl][di], " ");
        any = true;
      }
    }
    if (!any) absl::StrAppend(&result, "empty");
    absl::StrAppend(&result, "\n");
  }
  absl::StrAppend(&result, "Current player: ",
                   (current_player_ == 0) ? "Black" : "White", "\n");
  return result;
}

// ---------------------------------------------------------------------------
// Terminal / returns
// ---------------------------------------------------------------------------

bool ShogiState::IsTerminal() const {
  if (outcome_ != -1) return true;
  MaybeGenerateLegalActions();
  if (cached_legal_actions_->empty()) {
    // No legal moves = checkmate (or stalemate, which counts as a loss in
    // Shogi).
    const_cast<ShogiState*>(this)->outcome_ = 1 - current_player_;
    return true;
  }
  if (num_moves_ >= kMaxGameLength) {
    const_cast<ShogiState*>(this)->outcome_ = 2;
    return true;
  }
  return false;
}

std::vector<double> ShogiState::Returns() const {
  if (outcome_ == 0) return {kWinUtility, kLossUtility};
  if (outcome_ == 1) return {kLossUtility, kWinUtility};
  if (outcome_ == 2) return {kDrawUtility, kDrawUtility};
  return {0.0, 0.0};
}

// ---------------------------------------------------------------------------
// Observation
// ---------------------------------------------------------------------------

std::string ShogiState::InformationStateString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, kNumPlayers);
  return HistoryString();
}

std::string ShogiState::ObservationString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, kNumPlayers);
  return ToString();
}

void ShogiState::ObservationTensor(Player player,
                                    absl::Span<float> values) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, kNumPlayers);
  SPIEL_CHECK_EQ(static_cast<int>(values.size()),
                 kNumObservationPlanes * kBoardSize * kBoardSize);

  std::fill(values.begin(), values.end(), 0.0f);
  TensorView<3> view(values, {kNumObservationPlanes, kBoardSize, kBoardSize},
                     false);

  // Piece planes (14 per player):
  //   Unpromoted Pawn..King -> planes 0..7
  //   Promoted +P..+R      -> planes 8..13
  for (int sq = 0; sq < kNumSquares; ++sq) {
    const Piece& p = board_[sq];
    if (p.IsEmpty()) continue;
    auto [row, col] = IndexToRowCol(sq);
    int plane = p.player * 14;
    if (!p.promoted) {
      plane += static_cast<int>(p.type) - 1;
    } else {
      plane += 8 + PromotedPlaneIndex(p.type);
    }
    view[{plane, row, col}] = 1.0;
  }

  // Current player plane.
  float cp_val = (current_player_ == player) ? 1.0f : 0.0f;
  for (int r = 0; r < kBoardSize; ++r)
    for (int c = 0; c < kBoardSize; ++c)
      view[{kNumPiecePlanes, r, c}] = cp_val;

  // Hand count planes (normalized by 18).
  int hand_start = kNumPiecePlanes + 1;
  for (int pl = 0; pl < kNumPlayers; ++pl) {
    for (int di = 0; di < kNumDropPieceTypes; ++di) {
      float val = static_cast<float>(pieces_in_hand_[pl][di]) / 18.0f;
      int plane = hand_start + pl * kNumDropPieceTypes + di;
      for (int r = 0; r < kBoardSize; ++r)
        for (int c = 0; c < kBoardSize; ++c)
          view[{plane, r, c}] = val;
    }
  }
}

// ---------------------------------------------------------------------------
// Clone
// ---------------------------------------------------------------------------

std::unique_ptr<State> ShogiState::Clone() const {
  return std::unique_ptr<State>(new ShogiState(*this));
}

// ---------------------------------------------------------------------------
// DoApplyAction / UndoAction
// ---------------------------------------------------------------------------

void ShogiState::DoApplyAction(Action action) {
  cached_legal_actions_.reset();

  if (action >= kDropOffset) {
    int remainder = action - kDropOffset;
    int piece_index = remainder / kNumSquares;
    int to_sq = remainder % kNumSquares;
    PieceType type = kDropIndexToType[piece_index];

    SPIEL_CHECK_TRUE(board_[to_sq].IsEmpty());
    SPIEL_CHECK_GT(pieces_in_hand_[current_player_][piece_index], 0);

    board_[to_sq] = Piece{type, current_player_, false};
    pieces_in_hand_[current_player_][piece_index]--;

    undo_stack_.push_back(
        UndoInfo{action, Piece{kEmpty, -1, false}, false, -1, to_sq});
  } else {
    bool promote = (action >= kPromotionOffset);
    int raw = promote ? action - kPromotionOffset : action;
    int from_sq = raw / kNumSquares;
    int to_sq = raw % kNumSquares;

    Piece captured = board_[to_sq];
    bool was_promoted = board_[from_sq].promoted;

    board_[to_sq] = board_[from_sq];
    board_[from_sq] = Piece{kEmpty, -1, false};
    if (promote) board_[to_sq].promoted = true;

    // Captured pieces switch ownership and lose promotion status.
    if (!captured.IsEmpty()) {
      int di = PieceTypeToDropIndex(captured.type);
      if (di >= 0) pieces_in_hand_[current_player_][di]++;
      if (captured.type == kKing) outcome_ = current_player_;
    }

    undo_stack_.push_back(
        UndoInfo{action, captured, was_promoted, from_sq, to_sq});
  }

  current_player_ = 1 - current_player_;
  num_moves_++;
}

void ShogiState::UndoAction(Player player, Action action) {
  SPIEL_CHECK_FALSE(undo_stack_.empty());
  UndoInfo info = undo_stack_.back();
  undo_stack_.pop_back();
  cached_legal_actions_.reset();

  current_player_ = player;
  num_moves_--;
  outcome_ = -1;

  if (action >= kDropOffset) {
    int remainder = action - kDropOffset;
    int piece_index = remainder / kNumSquares;
    int to_sq = remainder % kNumSquares;
    board_[to_sq] = Piece{kEmpty, -1, false};
    pieces_in_hand_[player][piece_index]++;
  } else {
    bool promote = (action >= kPromotionOffset);
    int raw = promote ? action - kPromotionOffset : action;
    int from_sq = raw / kNumSquares;
    int to_sq = raw % kNumSquares;

    board_[from_sq] = board_[to_sq];
    board_[from_sq].promoted = info.was_promoted;
    board_[to_sq] = info.captured;

    if (!info.captured.IsEmpty()) {
      int di = PieceTypeToDropIndex(info.captured.type);
      if (di >= 0) pieces_in_hand_[player][di]--;
    }
  }

  history_.pop_back();
  --move_number_;
}

// ===========================================================================
// ShogiGame
// ===========================================================================

ShogiGame::ShogiGame(const GameParameters& params)
    : Game(kGameType, params) {}

}  // namespace shogi
}  // namespace open_spiel
