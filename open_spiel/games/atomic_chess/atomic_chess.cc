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

#include "open_spiel/games/atomic_chess/atomic_chess.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "open_spiel/abseil-cpp/absl/algorithm/container.h"
#include "open_spiel/abseil-cpp/absl/strings/match.h"
#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/abseil-cpp/absl/strings/str_join.h"
#include "open_spiel/abseil-cpp/absl/strings/str_split.h"
#include "open_spiel/abseil-cpp/absl/types/optional.h"
#include "open_spiel/abseil-cpp/absl/types/span.h"
#include "open_spiel/game_parameters.h"
#include "open_spiel/games/chess/chess.h"
#include "open_spiel/games/chess/chess_board.h"
#include "open_spiel/games/chess/chess_common.h"
#include "open_spiel/observer.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_globals.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace atomic_chess {
namespace {

using chess::ChessBoard;
using chess::Color;
using chess::ColorToPlayer;
using chess::DrawUtility;
using chess::kEmptyPiece;
using chess::kInvalidSquare;
using chess::kPieceTypes;
using chess::LossUtility;
using chess::Move;
using chess::OtherPlayer;
using chess::Piece;
using chess::PieceType;
using chess::Square;
using chess::WinUtility;

constexpr int kNumReversibleMovesToDraw = 100;
constexpr int kNumRepetitionsToDraw = 3;
// Standard chess start position; atomic chess keeps castling rights.
inline const std::string kDefaultStandardFEN =
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

ChessBoard MakeDefaultBoard() {
  auto maybe_board =
      ChessBoard::BoardFromFEN(kDefaultStandardFEN, kBoardSize,
                               /*king_in_check_allowed=*/true,
                               /*allow_pass_move=*/false,
                               /*allow_king_promotion=*/false);
  SPIEL_CHECK_TRUE(maybe_board);
  return *maybe_board;
}

inline Color OtherColor(Color c) {
  return c == Color::kWhite ? Color::kBlack : Color::kWhite;
}

// True if `move` is an en passant capture on `board` (a pawn moving to the
// current en passant target square, which is empty).
bool IsEnPassant(const ChessBoard& board, const Move& move) {
  return move.piece.type == PieceType::kPawn &&
         board.EpSquare() != kInvalidSquare && move.to == board.EpSquare();
}

// True if `move` captures a piece on `board` (including en passant).
bool IsCapture(const ChessBoard& board, const Move& move) {
  return board.at(move.to).type != PieceType::kEmpty ||
         IsEnPassant(board, move);
}

// Applies the atomic explosion for a capture that has ALREADY been applied to
// `board` via ApplyMove. `move` is the move that was applied and
// `was_en_passant` records whether it was an en passant capture (in which case
// the captured pawn sits one rank behind the destination square and must be
// removed explicitly).
void ApplyExplosion(ChessBoard* board, const Move& move, bool was_en_passant) {
  if (was_en_passant) {
    board->set_square(Square{move.to.x, move.from.y}, kEmptyPiece);
  }
  // The capturing piece is destroyed.
  board->set_square(move.to, kEmptyPiece);
  // All non-pawn pieces on the eight surrounding squares are destroyed.
  for (int8_t dx = -1; dx <= 1; ++dx) {
    for (int8_t dy = -1; dy <= 1; ++dy) {
      if (dx == 0 && dy == 0) continue;
      Square sq{static_cast<int8_t>(move.to.x + dx),
                static_cast<int8_t>(move.to.y + dy)};
      if (!board->InBoardArea(sq)) continue;
      Piece piece = board->at(sq);
      if (piece.type != PieceType::kEmpty && piece.type != PieceType::kPawn) {
        board->set_square(sq, kEmptyPiece);
      }
    }
  }
}

// True if the two kings are on adjacent squares (Chebyshev distance 1).
bool KingsAdjacent(const ChessBoard& board) {
  Square wk = board.find(Piece{Color::kWhite, PieceType::kKing});
  Square bk = board.find(Piece{Color::kBlack, PieceType::kKing});
  if (!board.InBoardArea(wk) || !board.InBoardArea(bk)) return false;
  return std::max(std::abs(wk.x - bk.x), std::abs(wk.y - bk.y)) == 1;
}

// Atomic-chess check: `color`'s king is attacked, EXCEPT that adjacent kings
// nullify all checks (the opponent could never capture the king without also
// exploding its own king).
bool InAtomicCheck(const ChessBoard& board, Color color) {
  Square ksq = board.find(Piece{color, PieceType::kKing});
  if (!board.InBoardArea(ksq)) return false;
  if (KingsAdjacent(board)) return false;
  return board.UnderAttack(ksq, color);
}

// Returns true if `color`'s king is not on the board (was exploded).
bool KingMissing(const ChessBoard& board, Color color) {
  return !board.InBoardArea(board.find(Piece{color, PieceType::kKing}));
}

// Castling legality for atomic chess: king may not start in, pass through, or
// land on an attacked square. Castling is never a capture, so no explosion.
bool CastlingLegal(const ChessBoard& board, const Move& move) {
  Color me = move.piece.color;
  int8_t y = move.from.y;
  int8_t lo = std::min(move.from.x, move.to.x);
  int8_t hi = std::max(move.from.x, move.to.x);
  for (int8_t x = lo; x <= hi; ++x) {
    if (board.UnderAttack(Square{x, y}, me)) return false;
  }
  return true;
}

// Facts about the game.
const GameType kGameType{
    /*short_name=*/"atomic_chess",
    /*long_name=*/"Atomic Chess",
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
    /*parameter_specification=*/
    {{"fen", GameParameter(GameParameter::Type::kString, false)}}};

std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::shared_ptr<const Game>(new AtomicChessGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);

RegisterSingleTensorObserver single_tensor(kGameType.short_name);

// Adds a plane to the observation vector corresponding to the presence and
// absence of the given piece type and colour at each square.
void AddPieceTypePlane(Color color, PieceType piece_type,
                       const ChessBoard& board,
                       absl::Span<float>::iterator& value_it) {
  for (int8_t y = 0; y < kBoardSize; ++y) {
    for (int8_t x = 0; x < kBoardSize; ++x) {
      Piece piece_on_board = board.at(Square{x, y});
      *value_it++ =
          (piece_on_board.color == color && piece_on_board.type == piece_type
               ? 1.0
               : 0.0);
    }
  }
}

// Adds a uniform scalar plane scaled with min and max.
template <typename T>
void AddScalarPlane(T val, T min, T max,
                    absl::Span<float>::iterator& value_it) {
  double normalized_val = static_cast<double>(val - min) / (max - min);
  for (int i = 0; i < k2dBoardSize; ++i) *value_it++ = normalized_val;
}

}  // namespace

AtomicChessState::AtomicChessState(std::shared_ptr<const Game> game)
    : State(game),
      start_board_(MakeDefaultBoard()),
      current_board_(start_board_) {
  repetitions_[current_board_.HashValue()] = 1;
}

AtomicChessState::AtomicChessState(std::shared_ptr<const Game> game,
                                   const std::string& fen)
    : State(game) {
  specific_initial_fen_ = fen;
  auto maybe_board =
      ChessBoard::BoardFromFEN(fen, kBoardSize,
                               /*king_in_check_allowed=*/true,
                               /*allow_pass_move=*/false,
                               /*allow_king_promotion=*/false);
  SPIEL_CHECK_TRUE(maybe_board);
  start_board_ = *maybe_board;
  current_board_ = start_board_;
  repetitions_[current_board_.HashValue()] = 1;
}

Action AtomicChessState::ParseMoveToAction(const std::string& move_str) const {
  absl::optional<Move> move = Board().ParseMove(move_str, false);
  if (!move.has_value()) {
    return kInvalidAction;
  }
  return chess::MoveToAction(*move, BoardSize());
}

void AtomicChessState::DoApplyAction(Action action) {
  Move move = chess::ActionToMove(action, Board());
  moves_history_.push_back(move);
  bool was_en_passant = IsEnPassant(Board(), move);
  bool is_capture = IsCapture(Board(), move);
  Board().ApplyMove(move);
  if (is_capture) {
    ApplyExplosion(&Board(), move, was_en_passant);
  }
  ++repetitions_[current_board_.HashValue()];
  cached_legal_actions_.reset();
}

void AtomicChessState::MaybeGenerateLegalActions() const {
  if (cached_legal_actions_) return;
  cached_legal_actions_.emplace();

  Color me = Board().ToPlay();
  Board().GenerateLegalMoves([this, me](const Move& move) -> bool {
    // Kings may never capture (they would explode themselves).
    if (move.piece.type == PieceType::kKing && IsCapture(Board(), move)) {
      return true;
    }

    // Castling is never a capture; validate it under atomic-check semantics.
    if (move.is_castling()) {
      if (CastlingLegal(Board(), move)) {
        cached_legal_actions_->push_back(chess::MoveToAction(move, kBoardSize));
      }
      return true;
    }

    // Simulate the move (and any resulting explosion) on a copy.
    ChessBoard board = Board();
    bool was_en_passant = IsEnPassant(board, move);
    bool is_capture = IsCapture(board, move);
    board.ApplyMove(move);
    if (is_capture) {
      ApplyExplosion(&board, move, was_en_passant);
    }

    // A move that removes our own king is illegal.
    if (KingMissing(board, me)) return true;

    // Exploding the enemy king wins immediately, even while in check.
    if (KingMissing(board, OtherColor(me))) {
      cached_legal_actions_->push_back(chess::MoveToAction(move, kBoardSize));
      return true;
    }

    // Otherwise the move is legal iff it does not leave our king in check.
    if (!InAtomicCheck(board, me)) {
      cached_legal_actions_->push_back(chess::MoveToAction(move, kBoardSize));
    }
    return true;
  });

  absl::c_sort(*cached_legal_actions_);
}

std::vector<Action> AtomicChessState::LegalActions() const {
  MaybeGenerateLegalActions();
  if (IsTerminal()) return {};
  return *cached_legal_actions_;
}

std::string AtomicChessState::ActionToString(Player player,
                                             Action action) const {
  Move move = chess::ActionToMove(action, Board());
  return move.ToSAN(Board());
}

std::string AtomicChessState::DebugString() const {
  return current_board_.DebugString(false);
}

std::string AtomicChessState::ToString() const { return Board().ToFEN(); }

std::vector<double> AtomicChessState::Returns() const {
  auto maybe_final_returns = MaybeFinalReturns();
  if (maybe_final_returns) {
    return *maybe_final_returns;
  } else {
    return {0.0, 0.0};
  }
}

std::string AtomicChessState::InformationStateString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  return HistoryString();
}

std::string AtomicChessState::ObservationString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  return ToString();
}

void AtomicChessState::ObservationTensor(Player player,
                                         absl::Span<float> values) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);

  auto value_it = values.begin();

  // Piece configuration.
  for (const auto& piece_type : kPieceTypes) {
    AddPieceTypePlane(Color::kWhite, piece_type, Board(), value_it);
    AddPieceTypePlane(Color::kBlack, piece_type, Board(), value_it);
  }

  AddPieceTypePlane(Color::kEmpty, PieceType::kEmpty, Board(), value_it);

  const auto entry = repetitions_.find(Board().HashValue());
  SPIEL_CHECK_FALSE(entry == repetitions_.end());
  int repetitions = entry->second;

  // Num repetitions for the current board.
  AddScalarPlane(repetitions, 1, 3, value_it);

  // Side to play.
  AddScalarPlane(ColorToPlayer(Board().ToPlay()), 0, 1, value_it);

  // Irreversible move counter.
  AddScalarPlane(Board().IrreversibleMoveCounter(), 0, 101, value_it);

  SPIEL_CHECK_EQ(value_it, values.end());
}

std::unique_ptr<State> AtomicChessState::Clone() const {
  return std::unique_ptr<State>(new AtomicChessState(*this));
}

void AtomicChessState::UndoAction(Player player, Action action) {
  SPIEL_CHECK_GE(moves_history_.size(), 1);
  --repetitions_[current_board_.HashValue()];
  moves_history_.pop_back();
  history_.pop_back();
  --move_number_;
  current_board_ = start_board_;
  for (const Move& move : moves_history_) {
    bool was_en_passant = IsEnPassant(current_board_, move);
    bool is_capture = IsCapture(current_board_, move);
    current_board_.ApplyMove(move);
    if (is_capture) {
      ApplyExplosion(&current_board_, move, was_en_passant);
    }
  }
}

bool AtomicChessState::IsRepetitionDraw() const {
  const auto entry = repetitions_.find(Board().HashValue());
  SPIEL_CHECK_FALSE(entry == repetitions_.end());
  return entry->second >= kNumRepetitionsToDraw;
}

std::pair<std::string, std::vector<std::string>>
AtomicChessState::ExtractFenAndMaybeMoves() const {
  SPIEL_CHECK_FALSE(IsChanceNode());
  std::string initial_fen = start_board_.ToFEN();
  std::vector<std::string> move_lans;
  std::unique_ptr<State> state = ParentGame()->NewInitialState(initial_fen);
  ChessBoard board = down_cast<const AtomicChessState&>(*state).Board();
  for (const Move& move : moves_history_) {
    move_lans.push_back(move.ToLAN(false, &board));
    bool was_en_passant = IsEnPassant(board, move);
    bool is_capture = IsCapture(board, move);
    board.ApplyMove(move);
    if (is_capture) {
      ApplyExplosion(&board, move, was_en_passant);
    }
  }
  return std::make_pair(initial_fen, move_lans);
}

absl::optional<std::vector<double>> AtomicChessState::MaybeFinalReturns()
    const {
  std::vector<double> returns(NumPlayers(), DrawUtility());

  // A king was exploded: the other player wins immediately.
  bool white_king_gone = KingMissing(Board(), Color::kWhite);
  bool black_king_gone = KingMissing(Board(), Color::kBlack);
  if (white_king_gone || black_king_gone) {
    // Both kings cannot be missing simultaneously (a move that would explode
    // one's own king is illegal).
    SPIEL_CHECK_FALSE(white_king_gone && black_king_gone);
    int winner = ColorToPlayer(white_king_gone ? Color::kBlack : Color::kWhite);
    returns[winner] = WinUtility();
    returns[OtherPlayer(winner)] = LossUtility();
    return returns;
  }

  if (IsRepetitionDraw()) return returns;

  if (Board().IrreversibleMoveCounter() >= kNumReversibleMovesToDraw) {
    return returns;
  }

  // Compute and cache the legal actions.
  MaybeGenerateLegalActions();
  SPIEL_CHECK_TRUE(cached_legal_actions_);

  if (cached_legal_actions_->empty()) {
    int to_play = ColorToPlayer(Board().ToPlay());
    if (InAtomicCheck(Board(), Board().ToPlay())) {
      // Checkmate: the player to move loses.
      returns[to_play] = LossUtility();
      returns[OtherPlayer(to_play)] = WinUtility();
    }
    // Otherwise stalemate: draw (returns already all DrawUtility()).
    return returns;
  }

  return absl::nullopt;
}

std::string AtomicChessState::Serialize() const {
  std::string state_str = "";
  // If the specific_initial_fen is empty, the deserializer will use the
  // default NewInitialState(). Otherwise, the deserializer will specify
  // the specific initial fen by calling NewInitialState(string).
  absl::StrAppend(&state_str, "FEN: ", specific_initial_fen_, "\n");
  std::vector<Action> history = History();
  absl::StrAppend(&state_str, absl::StrJoin(history, "\n"), "\n");
  return state_str;
}

std::string AtomicChessState::StartFEN() const {
  return start_board_.ToFEN();
}

AtomicChessGame::AtomicChessGame(const GameParameters& params)
    : Game(kGameType, params) {}

std::unique_ptr<State> AtomicChessGame::DeserializeState(
    const std::string& str) const {
  const std::string prefix("FEN: ");
  if (!absl::StartsWith(str, prefix)) {
    // Backward compatibility.
    return Game::DeserializeState(str);
  }
  std::vector<std::string> lines = absl::StrSplit(str, '\n');
  int line_num = 0;
  std::string fen = lines[line_num].substr(prefix.length());
  std::unique_ptr<State> state = nullptr;
  if (fen.empty()) {
    state = NewInitialState();
  } else {
    state = NewInitialState(fen);
  }
  line_num += 1;
  for (int i = line_num; i < lines.size(); ++i) {
    if (lines[i].empty()) {
      break;
    }
    Action action = static_cast<Action>(std::stol(lines[i]));
    state->ApplyAction(action);
  }
  return state;
}

}  // namespace atomic_chess
}  // namespace open_spiel
