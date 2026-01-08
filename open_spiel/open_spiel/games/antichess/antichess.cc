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

#include "open_spiel/games/antichess/antichess.h"

#include <sys/types.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <iterator>
#include <memory>
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
#include "open_spiel/games/chess/chess_common.h"
#include "open_spiel/games/chess/chess_board.h"
#include "open_spiel/observer.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_globals.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace antichess {
namespace {

using chess::CastlingDirection;
using chess::ChessBoard;
using chess::Color;
using chess::DrawUtility;
using chess::kKnightOffsets;
using chess::kPassMove;
using chess::kPieceTypes;
using chess::kUnderPromotionDirectionToOffset;
using chess::LossUtility;
using chess::Move;
using chess::Offset;
using chess::OtherPlayer;
using chess::Piece;
using chess::PieceType;
using chess::Square;
using chess::WinUtility;

constexpr int kNumReversibleMovesToDraw = 100;
constexpr int kNumRepetitionsToDraw = 3;
// No castling in antichess.
inline const std::string kDefaultStandardFEN =
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1";

ChessBoard MakeDefaultBoard() {
  auto maybe_board = ChessBoard::BoardFromFEN(kDefaultStandardFEN, kBoardSize,
                                              /*king_in_check_allowed=*/true,
                                              /*allow_pass_move=*/false,
                                              /*allow_king_promotion=*/true);
  SPIEL_CHECK_TRUE(maybe_board);
  return *maybe_board;
}

// Facts about the game
const GameType kGameType{
    /*short_name=*/"antichess",
    /*long_name=*/"Antichess (Losing Chess)",
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
  return std::shared_ptr<const Game>(new AntichessGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);

RegisterSingleTensorObserver single_tensor(kGameType.short_name);

// Adds a plane to the information state vector corresponding to the presence
// and absence of the given piece type and colour at each square.
void AddPieceTypePlane(Color color, PieceType piece_type,
                       const ChessBoard& board,
                       absl::Span<float>::iterator& value_it) {
  for (int8_t y = 0; y < kBoardSize; ++y) {
    for (int8_t x = 0; x < kBoardSize; ++x) {
      Piece piece_on_board = board.at(Square{x, y});
      *value_it++ = (
          piece_on_board.color == color && piece_on_board.type == piece_type ?
          1.0 : 0.0);
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

Action MoveToAction(const Move& move) {
  // Pass move is not allowed in antichess.
  bool is_pass_move = move == kPassMove;
  SPIEL_CHECK_FALSE(is_pass_move);
  // Castling is not allowed in antichess.
  SPIEL_CHECK_FALSE(move.is_castling());

  Color color = move.piece.color;
  // We rotate the move to be from player p's perspective.
  Move player_move(move);

  // Rotate move to be from player p's perspective.
  player_move.from.y = ReflectRank(color, kBoardSize, player_move.from.y);
  player_move.to.y = ReflectRank(color, kBoardSize, player_move.to.y);

  // For each starting square, we enumerate 76 actions:
  // - 12 = 3 (squares for promotion) * 4 (pieces to underpromote to) possible
  // underpromotions
  // - 56 queen moves
  // - 8 knight moves
  // In total, this results in 64 * 76 = 4864 indices.
  // This is the union of all possible moves, which can be further a little
  // reduced to the number of moves actually available from each starting
  // square.
  //

  int8_t x_diff = player_move.to.x - player_move.from.x;
  int8_t y_diff = player_move.to.y - player_move.from.y;
  Offset offset{x_diff, y_diff};
  int destination_action_index;
  // Check for underpromotion.
  if (move.promotion_type != PieceType::kEmpty &&
      move.promotion_type != PieceType::kQueen) {
    // We have to indicate underpromotions as special moves, because in terms of
    // from/to they are identical to queen promotions.
    // For a given starting square, an underpromotion can have 3 possible
    // destination squares (straight, left diagonal, right diagonal) and 4
    // possible piece types (3 from standard chess + King piece).
    SPIEL_CHECK_EQ(move.piece.type, PieceType::kPawn);
    SPIEL_CHECK_TRUE(player_move.from.y == kBoardSize - 2 &&
                     player_move.to.y == kBoardSize - 1);

    int promotion_index;
    {
      auto itr = absl::c_find(kUnderPromotionIndexToType, move.promotion_type);
      SPIEL_CHECK_TRUE(itr != kUnderPromotionIndexToType.end());
      promotion_index = std::distance(kUnderPromotionIndexToType.begin(), itr);
    }

    int direction_index;
    {
      auto itr = absl::c_find_if(
          kUnderPromotionDirectionToOffset,
          [offset](Offset o) { return o.x_offset == offset.x_offset; });
      SPIEL_CHECK_TRUE(itr != kUnderPromotionDirectionToOffset.end());
      direction_index =
          std::distance(kUnderPromotionDirectionToOffset.begin(), itr);
    }
    destination_action_index =
        kUnderPromotionDirectionToOffset.size() * promotion_index +
        direction_index;
  } else {
    // For the normal moves, we simply encode starting and destination square.
    int destination_index =
        OffsetToDestinationIndex(offset, kKnightOffsets, kBoardSize);
    destination_action_index = kNumUnderPromotions + destination_index;
    SPIEL_CHECK_TRUE(kNumUnderPromotions <= destination_action_index &&
                     destination_action_index < kNumActionDestinations);
  }

  return chess_common::EncodeNetworkTarget(player_move.from,
                                           destination_action_index, kBoardSize,
                                           kNumActionDestinations);
}

Move ActionToMove(const Action& action, const ChessBoard& board) {
  SPIEL_CHECK_GE(action, 0);
  SPIEL_CHECK_LT(action, NumDistinctActions());

  // The encoded action represents an action encoded from color's perspective.
  Color color = board.ToPlay();
  PieceType promotion_type = PieceType::kEmpty;
  CastlingDirection castle_dir = CastlingDirection::kNone;

  auto [from_square, destination_index] = chess_common::DecodeNetworkTarget(
      action, kBoardSize, kNumActionDestinations);

  Offset offset;
  bool is_under_promotion = destination_index < kNumUnderPromotions;
  if (is_under_promotion) {
    // Underpromotion.
    int promotion_index = destination_index / 3,
        direction_index = destination_index % 3;
    promotion_type = kUnderPromotionIndexToType[promotion_index];
    offset = kUnderPromotionDirectionToOffset[direction_index];
  } else {
    destination_index -= kNumUnderPromotions;
    offset =
        DestinationIndexToOffset(destination_index, kKnightOffsets, kBoardSize);
  }
  Square to_square = from_square + offset;

  from_square.y = ReflectRank(color, kBoardSize, from_square.y);
  to_square.y = ReflectRank(color, kBoardSize, to_square.y);

  // This uses the current state to infer the piece type.
  Piece piece = {board.ToPlay(), board.at(from_square).type};

  // Check for queen promotion.
  if (!is_under_promotion && piece.type == PieceType::kPawn &&
      ReflectRank(color, kBoardSize, from_square.y) == kBoardSize - 2 &&
      ReflectRank(color, kBoardSize, to_square.y) == kBoardSize - 1) {
    promotion_type = PieceType::kQueen;
  }

  return Move(from_square, to_square, piece, promotion_type, castle_dir);
}

AntichessState::AntichessState(std::shared_ptr<const Game> game)
    : State(game),
      start_board_(MakeDefaultBoard()),
      current_board_(start_board_) {
  repetitions_[current_board_.HashValue()] = 1;
}

AntichessState::AntichessState(std::shared_ptr<const Game> game,
                               const std::string& fen)
    : State(game) {
  specific_initial_fen_ = fen;
  auto maybe_board = ChessBoard::BoardFromFEN(fen, kBoardSize,
                                              /*king_in_check_allowed=*/true,
                                              /*allow_pass_move=*/false,
                                              /*allow_king_promotion=*/true);
  SPIEL_CHECK_TRUE(maybe_board);
  start_board_ = *maybe_board;
  // Castling not allowed in antichess.
  for (Color color : {Color::kWhite, Color::kBlack}) {
    for (CastlingDirection dir :
         {CastlingDirection::kLeft, CastlingDirection::kRight}) {
      SPIEL_CHECK_FALSE(start_board_.CastlingRight(color, dir));
    }
  }
  current_board_ = start_board_;
  repetitions_[current_board_.HashValue()] = 1;
}

Action AntichessState::ParseMoveToAction(const std::string& move_str) const {
  absl::optional<Move> move = Board().ParseMove(move_str, false);
  if (!move.has_value()) {
    return kInvalidAction;
  }
  return MoveToAction(*move, BoardSize());
}

void AntichessState::DoApplyAction(Action action) {
  Move move = antichess::ActionToMove(action, Board());
  moves_history_.push_back(move);
  Board().ApplyMove(move);
  ++repetitions_[current_board_.HashValue()];
  cached_legal_actions_.reset();
}

void AntichessState::MaybeGenerateLegalActions() const {
  if (cached_legal_actions_) return;

  // In Antichess, captures are mandatory
  absl::optional<std::vector<Action>> capture_moves, non_capture_moves;
  capture_moves.emplace();
  non_capture_moves.emplace();

  Board().GenerateLegalMoves(
      [this, &capture_moves, &non_capture_moves](const Move& move) -> bool {
        // Castling should be disabled in antichess.
        SPIEL_CHECK_FALSE(move.is_castling());
        Piece target_piece = Board().at(move.to);
        Action action = antichess::MoveToAction(move);
        // Check if this is a capture move
        if (target_piece.type == PieceType::kEmpty) {
          non_capture_moves->push_back(action);
        } else {
          capture_moves->push_back(action);
        }
        return true;
      });

  // If there are capture moves, only those are legal.
  cached_legal_actions_.swap(capture_moves->empty() ? non_capture_moves
                                                    : capture_moves);
  absl::c_sort(*cached_legal_actions_);
}

std::vector<Action> AntichessState::LegalActions() const {
  MaybeGenerateLegalActions();
  if (IsTerminal()) return {};
  return *cached_legal_actions_;
}

std::string AntichessState::ActionToString(Player player, Action action) const {
  Move move = antichess::ActionToMove(action, Board());
  return move.ToSAN(Board());
}

std::string AntichessState::DebugString() const {
  return current_board_.DebugString(false);
}

std::string AntichessState::ToString() const { return Board().ToFEN(); }

std::vector<double> AntichessState::Returns() const {
  auto maybe_final_returns = MaybeFinalReturns();
  if (maybe_final_returns) {
    return *maybe_final_returns;
  } else {
    return {0.0, 0.0};
  }
}

std::string AntichessState::InformationStateString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  return HistoryString();
}

std::string AntichessState::ObservationString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  return ToString();
}

void AntichessState::ObservationTensor(Player player,
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

std::unique_ptr<State> AntichessState::Clone() const {
  return std::unique_ptr<State>(new AntichessState(*this));
}

void AntichessState::UndoAction(Player player, Action action) {
  SPIEL_CHECK_GE(moves_history_.size(), 1);
  --repetitions_[current_board_.HashValue()];
  moves_history_.pop_back();
  history_.pop_back();
  --move_number_;
  current_board_ = start_board_;
  for (const Move& move : moves_history_) {
    current_board_.ApplyMove(move);
  }
}

bool AntichessState::IsRepetitionDraw() const {
  const auto entry = repetitions_.find(Board().HashValue());
  SPIEL_CHECK_FALSE(entry == repetitions_.end());
  return entry->second >= kNumRepetitionsToDraw;
}

std::pair<std::string, std::vector<std::string>>
AntichessState::ExtractFenAndMaybeMoves() const {
  SPIEL_CHECK_FALSE(IsChanceNode());
  std::string initial_fen = start_board_.ToFEN();
  std::vector<std::string> move_lans;
  std::unique_ptr<State> state = ParentGame()->NewInitialState(initial_fen);
  ChessBoard board = down_cast<const AntichessState&>(*state).Board();
  for (const Move& move : moves_history_) {
    move_lans.push_back(move.ToLAN(false, &board));
    board.ApplyMove(move);
  }
  return std::make_pair(initial_fen, move_lans);
}

absl::optional<std::vector<double>> AntichessState::MaybeFinalReturns() const {
  std::vector<double> returns(NumPlayers(), DrawUtility());
  if (IsRepetitionDraw()) return returns;

  // Check whether a player has lost all their pieces or is stalemated.
  std::array<bool, 2> has_pieces = {false, false};

  for (int8_t y = 0; y < Board().BoardSize(); ++y) {
    for (int8_t x = 0; x < Board().BoardSize(); ++x) {
      Piece piece = Board().at(Square{x, y});
      if (piece.type != PieceType::kEmpty) {
        has_pieces[ColorToPlayer(piece.color)] = true;
      }
    }
  }

  {
    auto winner_it = std::find(has_pieces.begin(), has_pieces.end(), false);
    if (winner_it != has_pieces.end()) {
      int winner = std::distance(has_pieces.begin(), winner_it);
      int loser = OtherPlayer(winner);
      // Only one player can be out of pieces.
      SPIEL_CHECK_TRUE(has_pieces[loser]);
      returns[winner] = WinUtility();
      returns[loser] = LossUtility();
      return returns;
    }
  }

  // Both players have pieces, check for stalemate.

  // Compute and cache the legal actions
  MaybeGenerateLegalActions();
  SPIEL_CHECK_TRUE(cached_legal_actions_);

  if (cached_legal_actions_->empty()) {
    // Stalemate, stalemated player wins.
    auto next_to_play = ColorToPlayer(Board().ToPlay());
    returns[next_to_play] = WinUtility();
    returns[OtherPlayer(next_to_play)] = LossUtility();
    return returns;
  }

  if (Board().IrreversibleMoveCounter() >= kNumReversibleMovesToDraw) {
    return returns;
  }

  return absl::nullopt;
}

std::string AntichessState::Serialize() const {
  std::string state_str = "";
  // If the specific_initial_fen is empty, the deserializer will use the
  // default NewInitialState(). Otherwise, the deserializer will specify
  // the specific initial fen by calling NewInitialState(string).
  absl::StrAppend(&state_str, "FEN: ", specific_initial_fen_, "\n");
  std::vector<Action> history = History();
  absl::StrAppend(&state_str, absl::StrJoin(history, "\n"), "\n");
  return state_str;
}

std::string AntichessState::StartFEN() const { return start_board_.ToFEN(); }

AntichessGame::AntichessGame(const GameParameters& params)
    : Game(kGameType, params) {}

std::unique_ptr<State> AntichessGame::DeserializeState(
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

}  // namespace antichess
}  // namespace open_spiel
