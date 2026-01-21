
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

#include "open_spiel/games/crazyhouse/crazyhouse.h"

#include <cmath>
#include <map>
#include <vector>

#include "open_spiel/abseil-cpp/absl/algorithm/container.h"
#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/abseil-cpp/absl/strings/str_split.h"
#include "open_spiel/games/chess/chess_common.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace crazyhouse {
namespace {

using chess::Color;
using chess::Move;
using chess::Piece;
using chess::PieceType;
using chess::Square;

// Facts about the game
const GameType kGameType{/*short_name=*/"crazyhouse",
                         /*long_name=*/"Crazyhouse Chess",
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
                         /*provides_observation_tensor=*/true};

std::shared_ptr<const Game> Factory(const GameParameters &params) {
  return std::shared_ptr<const Game>(new CrazyhouseGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);

// Drop order for encoding: P, N, B, R, Q
const std::array<PieceType, 5> kDropRefList = {
    PieceType::kPawn, PieceType::kKnight, PieceType::kBishop, PieceType::kRook,
    PieceType::kQueen};

// Helper to map PieceType to index 0-4
int DropTypeToIndex(PieceType pt) {
  switch (pt) {
  case PieceType::kPawn:
    return 0;
  case PieceType::kKnight:
    return 1;
  case PieceType::kBishop:
    return 2;
  case PieceType::kRook:
    return 3;
  case PieceType::kQueen:
    return 4;
  default:
    SpielFatalError(
        absl::StrCat("Invalid drop type: ", chess::PieceTypeToString(pt)));
  }
}

PieceType IndexToDropType(int idx) {
  SPIEL_CHECK_GE(idx, 0);
  SPIEL_CHECK_LT(idx, 5);
  return kDropRefList[idx];
}

int EncodeDrop(Color color, PieceType pt, Square to_sq) {
  // Reflect square if black?
  // Chess moves are encoded from player's perspective.
  // Standard chess action encoding does rotation.
  // We should probably follow suit for consistency if we want the policy to be
  // invariant? Yes, let's rotate.
  int board_size = 8;
  int rank = to_sq.y;
  if (color == Color::kBlack) {
    rank = board_size - 1 - rank;
  }

  int sq_idx = rank * board_size + to_sq.x;
  int type_idx = DropTypeToIndex(pt);

  // Base offset is standard chess actions
  return chess::NumDistinctActions() + (type_idx * 64 + sq_idx);
}

// Decode drop action. Returns (PieceType, Destination index)
std::pair<PieceType, int> DecodeDrop(int action) {
  int drop_action = action - chess::NumDistinctActions();
  int type_idx = drop_action / 64;
  int sq_idx = drop_action % 64;
  return {IndexToDropType(type_idx), sq_idx};
}

// Copied from chess.cc helper
template <typename T>
void AddScalarPlane(T val, T min, T max,
                    absl::Span<float>::iterator &value_it) {
  double normalized_val = static_cast<double>(val - min) / (max - min);
  for (int i = 0; i < chess::k2dMaxBoardSize; ++i)
    *value_it++ = normalized_val;
}

void AddPieceTypePlane(Color color, PieceType piece_type,
                       const chess::ChessBoard &board,
                       absl::Span<float>::iterator &value_it) {
  for (int8_t y = 0; y < chess::kMaxBoardSize; ++y) {
    for (int8_t x = 0; x < chess::kMaxBoardSize; ++x) {
      Piece piece_on_board = board.at(Square{x, y});
      *value_it++ =
          (piece_on_board.color == color && piece_on_board.type == piece_type
               ? 1.0
               : 0.0);
    }
  }
}
void AddBinaryPlane(bool val, absl::Span<float>::iterator &value_it) {
  AddScalarPlane<int>(val ? 1 : 0, 0, 1, value_it);
}

} // namespace

const std::array<chess::PieceType, 5> &GetDropTypes() { return kDropRefList; }

CrazyhouseState::CrazyhouseState(std::shared_ptr<const Game> game)
    : State(game), start_board_(chess::MakeDefaultBoard()),
      current_board_(start_board_) {
  repetitions_[current_board_.HashValue()] = 1;
}

CrazyhouseState::CrazyhouseState(std::shared_ptr<const Game> game,
                                 const std::string &fen)
    : State(game) {
  specific_initial_fen_ = fen;
  auto maybe_board = CrazyhouseBoard::BoardFromFEN(fen);
  if (!maybe_board) {
    SpielFatalError(absl::StrCat("Invalid FEN: ", fen));
  }
  start_board_ = *maybe_board;
  current_board_ = start_board_;
  repetitions_[current_board_.HashValue()] = 1;
}

Player CrazyhouseState::CurrentPlayer() const {
  return IsTerminal() ? kTerminalPlayerId
                      : chess::ColorToPlayer(Board().ToPlay());
}

std::vector<Action> CrazyhouseState::LegalActions() const {
  MaybeGenerateLegalActions();
  if (IsTerminal())
    return {};
  return *cached_legal_actions_;
}

void CrazyhouseState::MaybeGenerateLegalActions() const {
  if (!cached_legal_actions_) {
    cached_legal_actions_ = std::vector<Action>();
    Board().GenerateLegalMoves([this](const Move &move) -> bool {
      // If Drop Move
      if (IsDropMove(move)) {
        cached_legal_actions_->push_back(
            EncodeDrop(move.piece.color, move.piece.type, move.to));
      } else {
        // Standard Move
        // We use chess::MoveToAction but we need to ensure it works with our
        // Move struct Our Move is identical to chess::Move (aliased or same
        // struct). However, chess::MoveToAction needs board_size
        cached_legal_actions_->push_back(chess::MoveToAction(move, 8));
      }
      return true;
    });
    absl::c_sort(*cached_legal_actions_);
  }
}

std::string CrazyhouseState::ActionToString(Player player,
                                            Action action) const {
  if (action >= chess::NumDistinctActions()) {
    // Drop Move
    auto [pt, sq_idx] = DecodeDrop(action);
    // Decode square from perspective?
    // EncodeDrop rotated it. So we need to un-rotate to get board coordinate.
    int board_size = 8;
    int rank = sq_idx / board_size;
    int file = sq_idx % board_size;

    // Player 0 (Black) perspective: rank 0 is actually rank 7 on board
    Color color = chess::PlayerToColor(player);
    if (color == Color::kBlack) {
      rank = board_size - 1 - rank;
    }

    Square sq{static_cast<int8_t>(file), static_cast<int8_t>(rank)};
    std::string pt_str = chess::PieceTypeToString(pt, true);
    std::string sq_str = chess::SquareToString(sq);
    return absl::StrCat(pt_str, "@", sq_str);
  } else {
    // Standard Move
    // We can delegate to chess logic to format SAN
    Move move = chess::ActionToMove(action, Board());
    return move.ToLAN();
  }
}

std::string CrazyhouseState::ToString() const { return Board().ToFEN(); }

std::string CrazyhouseState::InformationStateString(Player player) const {
  return HistoryString();
}

std::string CrazyhouseState::ObservationString(Player player) const {
  return ToString();
}

void CrazyhouseState::ObservationTensor(Player player,
                                        absl::Span<float> values) const {
  // Similar to ChessState::ObservationTensor but with Pockets
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);

  auto value_it = values.begin();

  // 1. Piece configuration on board (standard chess)
  for (const auto &piece_type : chess::kPieceTypes) {
    AddPieceTypePlane(Color::kWhite, piece_type, Board(), value_it);
    AddPieceTypePlane(Color::kBlack, piece_type, Board(), value_it);
  }
  AddPieceTypePlane(Color::kEmpty, PieceType::kEmpty, Board(), value_it);

  // 2. Repetitions
  const auto entry = repetitions_.find(Board().HashValue());
  int rep = (entry != repetitions_.end()) ? entry->second : 0;
  AddScalarPlane(rep, 1, 3, value_it);

  // 3. Side to play
  AddScalarPlane(chess::ColorToPlayer(Board().ToPlay()), 0, 1, value_it);

  // 4. Irreversible move counter
  AddScalarPlane(Board().IrreversibleMoveCounter(), 0, 101, value_it);

  // 5. Castling rights
  AddBinaryPlane(
      Board().CastlingRight(Color::kWhite, chess::CastlingDirection::kLeft),
      value_it);
  AddBinaryPlane(
      Board().CastlingRight(Color::kWhite, chess::CastlingDirection::kRight),
      value_it);
  AddBinaryPlane(
      Board().CastlingRight(Color::kBlack, chess::CastlingDirection::kLeft),
      value_it);
  AddBinaryPlane(
      Board().CastlingRight(Color::kBlack, chess::CastlingDirection::kRight),
      value_it);

  // 6. [NEW] Pockets
  // Add planes for pocket counts.
  // We have 5 types allowed in pocket: P, N, B, R, Q
  // For each color.
  // Max count? 8 Pawns, 2 R, 2 B, 2 N, 1 Q each side.
  // Actually in Crazyhouse, piece counts can exceed standard limits (e.g. 16
  // pawns). A safe max normalization constant might be 16?
  const int kMaxPocket = 16;

  for (Color color : {Color::kWhite, Color::kBlack}) {
    const auto &pocket = Board().pocket(color);
    for (PieceType pt : kDropRefList) {
      int count = pocket[static_cast<int>(pt)];
      AddScalarPlane(count, 0, kMaxPocket, value_it);
    }
  }

  SPIEL_CHECK_EQ(value_it, values.end());
}

absl::optional<std::vector<double>> CrazyhouseState::MaybeFinalReturns() const {
  // Check sufficient material?
  // In crazyhouse, material can reappear.
  // So insufficient material is extremely rare (only K vs K? but even then...)
  // Let's assume K vs K is draw. But K+P vs K is winnable.
  // Just delegate to standard checks if implementation allows.
  // But standard chess HasSufficientMaterial implementation might be
  // restrictive. We can skip material check for MVP.

  const auto entry = repetitions_.find(Board().HashValue());
  if (entry != repetitions_.end() && entry->second >= 3) {
    return std::vector<double>{0.0, 0.0};
  }

  MaybeGenerateLegalActions();
  if (!cached_legal_actions_ || cached_legal_actions_->empty()) {
    if (Board().InCheck()) {
      std::vector<double> returns(NumPlayers());
      auto next_to_play = chess::ColorToPlayer(Board().ToPlay());
      returns[next_to_play] = -1.0;
      returns[chess::OtherPlayer(next_to_play)] = 1.0;
      return returns;
    } else {
      return std::vector<double>{0.0, 0.0}; // Stalemate
    }
  }

  // 50-move rule (100 reversible PLY)
  if (Board().IrreversibleMoveCounter() >= 100) {
    return std::vector<double>{0.0, 0.0};
  }

  return absl::nullopt;
}

std::string CrazyhouseState::Serialize() const { return ToString(); }

void CrazyhouseState::UndoAction(Player player, Action action) {
  // Simple undo not supported yet efficiently.
  // Reconstruct from start.
  // This is expensive but correct.
  SPIEL_CHECK_GE(moves_history_.size(), 1);
  --repetitions_[current_board_.HashValue()];
  moves_history_.pop_back();
  history_.pop_back(); // State::history_

  current_board_ = start_board_;
  for (const Move &move : moves_history_) {
    current_board_.ApplyMove(move);
  }
}

std::unique_ptr<State> CrazyhouseState::Clone() const {
  return std::make_unique<CrazyhouseState>(*this);
}

void CrazyhouseState::DoApplyAction(Action action) {
  Move move;
  if (action >= chess::NumDistinctActions()) {
    // Decode Drop
    auto [pt, sq_idx] = DecodeDrop(action);
    // Un-rotate
    int board_size = 8;
    int rank = sq_idx / board_size;
    int file = sq_idx % board_size;
    Color color = Board().ToPlay();
    if (color == Color::kBlack) {
      rank = board_size - 1 - rank;
    }
    Square sq{static_cast<int8_t>(file), static_cast<int8_t>(rank)};
    Piece piece{color, pt};
    move = Move(chess::kInvalidSquare, sq, piece);
  } else {
    move = chess::ActionToMove(action, Board());
  }

  moves_history_.push_back(move);
  Board().ApplyMove(move);
  ++repetitions_[current_board_.HashValue()];
  cached_legal_actions_.reset();
}

std::vector<double> CrazyhouseState::Returns() const {
  auto maybe = MaybeFinalReturns();
  return maybe ? *maybe : std::vector<double>{0.0, 0.0};
}

CrazyhouseGame::CrazyhouseGame(const GameParameters &params)
    : Game(kGameType, params) {}

std::unique_ptr<State>
CrazyhouseGame::DeserializeState(const std::string &str) const {
  return NewInitialState(str);
}

std::vector<int> CrazyhouseGame::ObservationTensorShape() const {
  // Standard Chess: 13 (piece types) + 1 (rep) + 1 (side) + 1 (rev) + 4
  // (castle) = 20 planes. Crazyhouse Adds: 10 planes (5 drop types * 2 colors).
  // Total 30 planes.
  return {30, 8, 8};
}

} // namespace crazyhouse
} // namespace open_spiel
