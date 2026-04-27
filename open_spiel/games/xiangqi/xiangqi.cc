// Copyright 2024 DeepMind Technologies Limited
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

#include "open_spiel/games/xiangqi/xiangqi.h"

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/abseil-cpp/absl/types/span.h"
#include "open_spiel/observer.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_globals.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/utils/tensor_view.h"

namespace open_spiel {
namespace xiangqi {
namespace {

const GameType kGameType{/*short_name=*/"xiangqi",
                         /*long_name=*/"Chinese Chess (Xiangqi)",
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
  return std::shared_ptr<const Game>(new XiangqiGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);

RegisterSingleTensorObserver single_tensor(kGameType.short_name);

constexpr int kOrthoDirs[4][2] = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
constexpr int kDiagDirs[4][2] = {{-1, -1}, {-1, 1}, {1, -1}, {1, 1}};

// Horse moves: (ortho_dr, ortho_dc, final_dr, final_dc)
// The horse first moves one step orthogonally, then one step diagonally
// outward. The orthogonal leg can be blocked.
constexpr int kHorseMoves[8][4] = {
    {-1, 0, -2, -1}, {-1, 0, -2, 1},  // north leg
    {1, 0, 2, -1},   {1, 0, 2, 1},    // south leg
    {0, -1, -1, -2}, {0, -1, 1, -2},  // west leg
    {0, 1, -1, 2},   {0, 1, 1, 2},    // east leg
};

bool InBounds(int row, int col) {
  return row >= 0 && row < kNumRows && col >= 0 && col < kNumCols;
}

}  // namespace

// Palace: rows 0-2, cols 3-5 for Black; rows 7-9, cols 3-5 for Red.
bool IsInPalace(int row, int col, int player) {
  if (col < 3 || col > 5) return false;
  if (player == 0) return row >= 7 && row <= 9;  // Red
  return row >= 0 && row <= 2;                   // Black
}

// Own side of the river: Red = rows 5-9, Black = rows 0-4.
bool IsOnOwnSide(int row, int player) {
  if (player == 0) return row >= 5;
  return row <= 4;
}

char PieceTypeToChar(PieceType type) {
  switch (type) {
    case kGeneral:
      return 'G';
    case kAdvisor:
      return 'A';
    case kElephant:
      return 'E';
    case kHorse:
      return 'H';
    case kChariot:
      return 'R';
    case kCannon:
      return 'C';
    case kSoldier:
      return 'S';
    default:
      return '.';
  }
}

// ---------------------------------------------------------------------------
// XiangqiState
// ---------------------------------------------------------------------------

XiangqiState::XiangqiState(std::shared_ptr<const Game> game) : State(game) {
  SetupInitialBoard();
}

void XiangqiState::SetupInitialBoard() {
  board_.fill(kEmptyPiece);

  // Black back rank (row 0).
  auto place = [&](int row, int col, PieceType type, int player) {
    board_[SquareIndex(row, col)] = {type, player};
  };

  const PieceType back_rank[] = {kChariot,  kHorse,   kElephant,
                                 kAdvisor,  kGeneral, kAdvisor,
                                 kElephant, kHorse,   kChariot};
  for (int c = 0; c < 9; ++c) {
    place(0, c, back_rank[c], 1);  // Black
    place(9, c, back_rank[c], 0);  // Red
  }

  // Cannons.
  place(2, 1, kCannon, 1);
  place(2, 7, kCannon, 1);
  place(7, 1, kCannon, 0);
  place(7, 7, kCannon, 0);

  // Soldiers.
  for (int c = 0; c < 9; c += 2) {
    place(3, c, kSoldier, 1);
    place(6, c, kSoldier, 0);
  }
}

std::vector<Action> XiangqiState::LegalActions() const {
  if (IsTerminal()) return {};
  std::vector<Action> pseudo;
  GeneratePseudoLegalMoves(&pseudo);

  std::vector<Action> legal;
  legal.reserve(pseudo.size());
  for (Action action : pseudo) {
    auto [from, to] = DecodeMove(action);
    if (!WouldLeaveInCheck(from, to)) {
      legal.push_back(action);
    }
  }
  std::sort(legal.begin(), legal.end());
  return legal;
}

void XiangqiState::GeneratePseudoLegalMoves(std::vector<Action>* moves) const {
  for (int sq = 0; sq < kNumCells; ++sq) {
    const Piece& p = board_[sq];
    if (p.IsEmpty() || p.player != current_player_) continue;
    switch (p.type) {
      case kGeneral:
        GenerateGeneralMoves(sq, moves);
        break;
      case kAdvisor:
        GenerateAdvisorMoves(sq, moves);
        break;
      case kElephant:
        GenerateElephantMoves(sq, moves);
        break;
      case kHorse:
        GenerateHorseMoves(sq, moves);
        break;
      case kChariot:
        GenerateChariotMoves(sq, moves);
        break;
      case kCannon:
        GenerateCannonMoves(sq, moves);
        break;
      case kSoldier:
        GenerateSoldierMoves(sq, moves);
        break;
      default:
        break;
    }
  }
}

void XiangqiState::GenerateGeneralMoves(int sq,
                                        std::vector<Action>* moves) const {
  int row = SquareRow(sq);
  int col = SquareCol(sq);
  int player = board_[sq].player;
  for (auto& d : kOrthoDirs) {
    int nr = row + d[0], nc = col + d[1];
    if (!IsInPalace(nr, nc, player)) continue;
    int to = SquareIndex(nr, nc);
    if (board_[to].IsEmpty() || board_[to].player != player) {
      moves->push_back(EncodeMove(sq, to));
    }
  }
}

void XiangqiState::GenerateAdvisorMoves(int sq,
                                        std::vector<Action>* moves) const {
  int row = SquareRow(sq);
  int col = SquareCol(sq);
  int player = board_[sq].player;
  for (auto& d : kDiagDirs) {
    int nr = row + d[0], nc = col + d[1];
    if (!IsInPalace(nr, nc, player)) continue;
    int to = SquareIndex(nr, nc);
    if (board_[to].IsEmpty() || board_[to].player != player) {
      moves->push_back(EncodeMove(sq, to));
    }
  }
}

void XiangqiState::GenerateElephantMoves(int sq,
                                         std::vector<Action>* moves) const {
  int row = SquareRow(sq);
  int col = SquareCol(sq);
  int player = board_[sq].player;
  for (auto& d : kDiagDirs) {
    int mr = row + d[0], mc = col + d[1];  // midpoint (eye)
    int nr = row + 2 * d[0], nc = col + 2 * d[1];
    if (!InBounds(nr, nc)) continue;
    if (!IsOnOwnSide(nr, player)) continue;                // cannot cross river
    if (!board_[SquareIndex(mr, mc)].IsEmpty()) continue;  // blocked
    int to = SquareIndex(nr, nc);
    if (board_[to].IsEmpty() || board_[to].player != player) {
      moves->push_back(EncodeMove(sq, to));
    }
  }
}

void XiangqiState::GenerateHorseMoves(int sq,
                                      std::vector<Action>* moves) const {
  int row = SquareRow(sq);
  int col = SquareCol(sq);
  int player = board_[sq].player;
  for (auto& m : kHorseMoves) {
    int block_r = row + m[0], block_c = col + m[1];
    int nr = row + m[2], nc = col + m[3];
    if (!InBounds(nr, nc)) continue;
    // Blocked if the orthogonal adjacent square is occupied.
    if (!board_[SquareIndex(block_r, block_c)].IsEmpty()) continue;
    int to = SquareIndex(nr, nc);
    if (board_[to].IsEmpty() || board_[to].player != player) {
      moves->push_back(EncodeMove(sq, to));
    }
  }
}

void XiangqiState::GenerateChariotMoves(int sq,
                                        std::vector<Action>* moves) const {
  int row = SquareRow(sq);
  int col = SquareCol(sq);
  int player = board_[sq].player;
  for (auto& d : kOrthoDirs) {
    for (int step = 1;; ++step) {
      int nr = row + d[0] * step, nc = col + d[1] * step;
      if (!InBounds(nr, nc)) break;
      int to = SquareIndex(nr, nc);
      if (board_[to].IsEmpty()) {
        moves->push_back(EncodeMove(sq, to));
      } else {
        if (board_[to].player != player) {
          moves->push_back(EncodeMove(sq, to));
        }
        break;
      }
    }
  }
}

void XiangqiState::GenerateCannonMoves(int sq,
                                       std::vector<Action>* moves) const {
  int row = SquareRow(sq);
  int col = SquareCol(sq);
  int player = board_[sq].player;
  for (auto& d : kOrthoDirs) {
    bool jumped = false;
    for (int step = 1;; ++step) {
      int nr = row + d[0] * step, nc = col + d[1] * step;
      if (!InBounds(nr, nc)) break;
      int to = SquareIndex(nr, nc);
      if (!jumped) {
        if (board_[to].IsEmpty()) {
          moves->push_back(EncodeMove(sq, to));  // non-capture move
        } else {
          jumped = true;  // found the screen piece
        }
      } else {
        if (!board_[to].IsEmpty()) {
          if (board_[to].player != player) {
            moves->push_back(EncodeMove(sq, to));  // capture over screen
          }
          break;
        }
      }
    }
  }
}

void XiangqiState::GenerateSoldierMoves(int sq,
                                        std::vector<Action>* moves) const {
  int row = SquareRow(sq);
  int col = SquareCol(sq);
  int player = board_[sq].player;

  // Forward direction: Red moves toward row 0 (dr=-1), Black toward row 9
  // (dr=+1).
  int forward = (player == 0) ? -1 : 1;
  bool crossed_river = !IsOnOwnSide(row, player);

  auto try_move = [&](int nr, int nc) {
    if (!InBounds(nr, nc)) return;
    int to = SquareIndex(nr, nc);
    if (board_[to].IsEmpty() || board_[to].player != player) {
      moves->push_back(EncodeMove(sq, to));
    }
  };

  // Always can move forward.
  try_move(row + forward, col);

  // After crossing the river, can also move sideways.
  if (crossed_river) {
    try_move(row, col - 1);
    try_move(row, col + 1);
  }
}

int XiangqiState::FindGeneral(int player) const {
  for (int sq = 0; sq < kNumCells; ++sq) {
    if (board_[sq].type == kGeneral && board_[sq].player == player) {
      return sq;
    }
  }
  return -1;  // general was captured
}

// Check if square `sq` is attacked by any piece belonging to `attacker`.
bool XiangqiState::IsAttackedBy(int sq, int attacker) const {
  int row = SquareRow(sq);
  int col = SquareCol(sq);

  // Attacked by General? (relevant for flying general check)
  for (auto& d : kOrthoDirs) {
    int nr = row + d[0], nc = col + d[1];
    if (InBounds(nr, nc)) {
      const Piece& p = board_[SquareIndex(nr, nc)];
      if (p.type == kGeneral && p.player == attacker) return true;
    }
  }

  // Attacked by Advisor?
  for (auto& d : kDiagDirs) {
    int nr = row + d[0], nc = col + d[1];
    if (InBounds(nr, nc)) {
      const Piece& p = board_[SquareIndex(nr, nc)];
      if (p.type == kAdvisor && p.player == attacker &&
          IsInPalace(nr, nc, attacker))
        return true;
    }
  }

  // Attacked by Elephant?
  for (auto& d : kDiagDirs) {
    int nr = row + 2 * d[0], nc = col + 2 * d[1];
    if (!InBounds(nr, nc)) continue;
    const Piece& p = board_[SquareIndex(nr, nc)];
    if (p.type != kElephant || p.player != attacker) continue;
    if (!IsOnOwnSide(nr, attacker)) continue;
    int mr = row + d[0], mc = col + d[1];
    if (!board_[SquareIndex(mr, mc)].IsEmpty()) continue;
    return true;
  }

  // Attacked by Horse? (reverse the horse move to find attacking horses)
  // A horse at (hr, hc) reaches (hr+m[2], hc+m[3]) if (hr+m[0], hc+m[1]) is
  // empty. So a horse attacking (row, col) sits at (row-m[2], col-m[3]).
  for (auto& m : kHorseMoves) {
    int hr = row - m[2], hc = col - m[3];
    if (!InBounds(hr, hc)) continue;
    const Piece& p = board_[SquareIndex(hr, hc)];
    if (p.type != kHorse || p.player != attacker) continue;
    int block_r = hr + m[0], block_c = hc + m[1];
    if (!board_[SquareIndex(block_r, block_c)].IsEmpty()) continue;
    return true;
  }

  // Attacked by Chariot or Cannon? (orthogonal rays)
  for (auto& d : kOrthoDirs) {
    bool found_screen = false;
    for (int step = 1;; ++step) {
      int nr = row + d[0] * step, nc = col + d[1] * step;
      if (!InBounds(nr, nc)) break;
      const Piece& p = board_[SquareIndex(nr, nc)];
      if (!found_screen) {
        if (!p.IsEmpty()) {
          if (p.player == attacker && p.type == kChariot) return true;
          found_screen = true;
        }
      } else {
        if (!p.IsEmpty()) {
          if (p.player == attacker && p.type == kCannon) return true;
          break;
        }
      }
    }
  }

  // Attacked by Soldier?
  // A soldier attacks the squares it could move to. Red soldiers attack the
  // square directly "forward" (toward row 0) and sideways if across the river.
  // Since we're checking if `attacker` attacks `sq`, we reverse the logic:
  // the attacker soldier could be at positions from which it could reach sq.
  int fwd = (attacker == 0) ? -1 : 1;
  // Soldier directly behind sq (from attacker's perspective):
  {
    int sr = row - fwd, sc = col;
    if (InBounds(sr, sc)) {
      const Piece& p = board_[SquareIndex(sr, sc)];
      if (p.type == kSoldier && p.player == attacker) return true;
    }
  }
  // Soldier to the left or right (only if the soldier has crossed the river):
  for (int dc : {-1, 1}) {
    int sr = row, sc = col + dc;
    if (!InBounds(sr, sc)) continue;
    const Piece& p = board_[SquareIndex(sr, sc)];
    if (p.type == kSoldier && p.player == attacker &&
        !IsOnOwnSide(sr, attacker)) {
      return true;
    }
  }

  return false;
}

bool XiangqiState::IsInCheck(int player) const {
  int gen = FindGeneral(player);
  if (gen < 0) return true;  // general captured = "in check"
  return IsAttackedBy(gen, 1 - player);
}

// Check if the two generals face each other on the same column with no pieces
// between them. Called after the board has been temporarily modified.
bool XiangqiState::ViolatesFlyingGeneral() const {
  int red_gen = FindGeneral(0);
  int black_gen = FindGeneral(1);
  if (red_gen < 0 || black_gen < 0) return false;

  int red_col = SquareCol(red_gen);
  int black_col = SquareCol(black_gen);
  if (red_col != black_col) return false;

  int red_row = SquareRow(red_gen);
  int black_row = SquareRow(black_gen);

  // Check for intervening pieces between the two generals.
  int min_row = std::min(red_row, black_row);
  int max_row = std::max(red_row, black_row);
  for (int r = min_row + 1; r < max_row; ++r) {
    if (!board_[SquareIndex(r, red_col)].IsEmpty()) return false;
  }
  return true;  // generals face each other with nothing between
}

bool XiangqiState::WouldLeaveInCheck(int from, int to) const {
  // Make a temporary modification to check legality.
  auto& mutable_board = const_cast<std::array<Piece, kNumCells>&>(board_);
  Piece from_piece = mutable_board[from];
  Piece captured = mutable_board[to];
  mutable_board[to] = from_piece;
  mutable_board[from] = kEmptyPiece;

  bool illegal = IsInCheck(current_player_) || ViolatesFlyingGeneral();

  mutable_board[from] = from_piece;
  mutable_board[to] = captured;
  return illegal;
}

void XiangqiState::DoApplyAction(Action action) {
  auto [from, to] = DecodeMove(action);
  SPIEL_CHECK_GE(from, 0);
  SPIEL_CHECK_LT(from, kNumCells);
  SPIEL_CHECK_GE(to, 0);
  SPIEL_CHECK_LT(to, kNumCells);
  SPIEL_CHECK_EQ(board_[from].player, current_player_);

  Piece captured = board_[to];
  undo_stack_.push_back({from, to, captured});

  board_[to] = board_[from];
  board_[from] = kEmptyPiece;

  // Check if a general was captured.
  if (captured.type == kGeneral) {
    outcome_ = current_player_;
  }

  current_player_ = 1 - current_player_;

  // Check if the new current player has no legal moves (stalemate = loss).
  if (outcome_ == kInvalidPlayer) {
    std::vector<Action> next_legal = LegalActions();
    if (next_legal.empty()) {
      // The player who just moved wins because opponent has no legal moves.
      outcome_ = 1 - current_player_;
      no_legal_moves_ = true;
    }
  }
}

void XiangqiState::UndoAction(Player player, Action action) {
  auto [from, to] = DecodeMove(action);
  SPIEL_CHECK_FALSE(undo_stack_.empty());
  MoveHistoryEntry entry = undo_stack_.back();
  undo_stack_.pop_back();

  board_[from] = board_[to];
  board_[to] = entry.captured;
  current_player_ = player;
  outcome_ = kInvalidPlayer;
  no_legal_moves_ = false;
  history_.pop_back();
  --move_number_;
}

bool XiangqiState::IsTerminal() const {
  return outcome_ != kInvalidPlayer || move_number_ >= kMaxGameLength;
}

std::vector<double> XiangqiState::Returns() const {
  if (outcome_ == 0) return {1.0, -1.0};  // Red wins
  if (outcome_ == 1) return {-1.0, 1.0};  // Black wins
  return {0.0, 0.0};
}

std::string XiangqiState::ActionToString(Player player,
                                         Action action_id) const {
  auto [from, to] = DecodeMove(action_id);
  int fr = SquareRow(from), fc = SquareCol(from);
  int tr = SquareRow(to), tc = SquareCol(to);
  char piece_char = PieceTypeToChar(board_[from].type);
  if (board_[from].player == 1) {
    piece_char = piece_char - 'A' + 'a';  // lowercase for Black
  }
  return absl::StrCat(std::string(1, piece_char), "(", fr, ",", fc, ")-(", tr,
                      ",", tc, ")");
}

std::string XiangqiState::ToString() const {
  std::string str;
  for (int r = 0; r < kNumRows; ++r) {
    for (int c = 0; c < kNumCols; ++c) {
      const Piece& p = board_[SquareIndex(r, c)];
      if (p.IsEmpty()) {
        absl::StrAppend(&str, ".");
      } else {
        char ch = PieceTypeToChar(p.type);
        if (p.player == 1) ch = ch - 'A' + 'a';  // lowercase for Black
        absl::StrAppend(&str, std::string(1, ch));
      }
    }
    if (r < kNumRows - 1) absl::StrAppend(&str, "\n");
  }
  return str;
}

std::string XiangqiState::InformationStateString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  return HistoryString();
}

std::string XiangqiState::ObservationString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  return ToString();
}

void XiangqiState::ObservationTensor(Player player,
                                     absl::Span<float> values) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);

  TensorView<3> view(values, {kNumObservationPlanes, kNumRows, kNumCols}, true);

  for (int sq = 0; sq < kNumCells; ++sq) {
    const Piece& p = board_[sq];
    if (p.IsEmpty()) continue;
    int r = SquareRow(sq);
    int c = SquareCol(sq);
    // Planes 0-6: Red piece types (kGeneral=1 .. kSoldier=7, so index =
    // type-1). Planes 7-13: Black piece types.
    int plane = (p.type - 1) + p.player * kNumPieceTypes;
    view[{plane, r, c}] = 1.0;
  }

  // Plane 14: current player (1.0 if Red to move, 0.0 if Black).
  float player_val = (current_player_ == 0) ? 1.0f : 0.0f;
  for (int r = 0; r < kNumRows; ++r) {
    for (int c = 0; c < kNumCols; ++c) {
      view[{kNumObservationPlanes - 1, r, c}] = player_val;
    }
  }
}

std::unique_ptr<State> XiangqiState::Clone() const {
  return std::unique_ptr<State>(new XiangqiState(*this));
}

// ---------------------------------------------------------------------------
// XiangqiGame
// ---------------------------------------------------------------------------

XiangqiGame::XiangqiGame(const GameParameters& params)
    : Game(kGameType, params) {}

}  // namespace xiangqi
}  // namespace open_spiel
