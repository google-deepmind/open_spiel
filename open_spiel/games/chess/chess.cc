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

#include "open_spiel/games/chess/chess.h"
#include <sys/types.h>

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
#include "open_spiel/games/chess/chess_board.h"
#include "open_spiel/games/chess/chess_common.h"
#include "open_spiel/observer.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_globals.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace chess {
namespace {

constexpr int kNumReversibleMovesToDraw = 100;
constexpr int kNumRepetitionsToDraw = 3;

// Facts about the game
const GameType kGameType{/*short_name=*/"chess",
                         /*long_name=*/"Chess",
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
                         {{"chess960", GameParameter(kDefaultChess960)}}};

std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::shared_ptr<const Game>(new ChessGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);

RegisterSingleTensorObserver single_tensor(kGameType.short_name);

// Adds a plane to the information state vector corresponding to the presence
// and absence of the given piece type and colour at each square.
void AddPieceTypePlane(Color color, PieceType piece_type,
                       const ChessBoard& board,
                       absl::Span<float>::iterator& value_it) {
  for (int8_t y = 0; y < kMaxBoardSize; ++y) {
    for (int8_t x = 0; x < kMaxBoardSize; ++x) {
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
  for (int i = 0; i < k2dMaxBoardSize; ++i) *value_it++ = normalized_val;
}

// Adds a binary scalar plane.
void AddBinaryPlane(bool val, absl::Span<float>::iterator& value_it) {
  AddScalarPlane<int>(val ? 1 : 0, 0, 1, value_it);
}
}  // namespace

ChessState::ChessState(std::shared_ptr<const Game> game)
    : State(game),
      start_board_(MakeDefaultBoard()),
      current_board_(start_board_) {
  repetitions_[current_board_.HashValue()] = 1;
}

ChessState::ChessState(std::shared_ptr<const Game> game, const std::string& fen)
    : State(game) {
  specific_initial_fen_ = fen;
  auto maybe_board = ChessBoard::BoardFromFEN(fen);
  SPIEL_CHECK_TRUE(maybe_board);
  start_board_ = *maybe_board;
  current_board_ = start_board_;
  repetitions_[current_board_.HashValue()] = 1;
}

Player ChessState::CurrentPlayer() const {
  if (ParentGame()->IsChess960() && specific_initial_fen_.empty() &&
      move_number_ == 0) {
    return kChancePlayerId;
  }
  return IsTerminal() ? kTerminalPlayerId : ColorToPlayer(Board().ToPlay());
}

ActionsAndProbs ChessState::ChanceOutcomes() const {
  SPIEL_CHECK_TRUE(ParentGame()->IsChess960());
  // One chance outcome for each initial position in chess960.
  ActionsAndProbs outcomes;
  outcomes.reserve(960);
  for (int i = 0; i < 960; ++i) {
    outcomes.push_back({i, 1.0 / 960});
  }
  return outcomes;
}

Action ChessState::ParseMoveToAction(const std::string& move_str) const {
  bool chess960 = ParentGame()->IsChess960();
  absl::optional<Move> move = Board().ParseMove(move_str, chess960);
  if (!move.has_value()) {
    return kInvalidAction;
  }
  return MoveToAction(*move, BoardSize());
}

void ChessState::DoApplyAction(Action action) {
  if (IsChanceNode()) {
    SPIEL_CHECK_TRUE(ParentGame()->IsChess960());
    // In chess960, there could be a chance node at the top of the game if the
    // initial FEN is not passed in. So here we apply the initial position.
    // First, reset the repetitions table.
    repetitions_ = RepetitionTable();

    // Then get the initial fen and set the board.
    std::string fen = ParentGame()->Chess960LookupFEN(action);
    auto maybe_board = ChessBoard::BoardFromFEN(fen);
    SPIEL_CHECK_TRUE(maybe_board);
    start_board_ = *maybe_board;
    current_board_ = start_board_;
    repetitions_[current_board_.HashValue()] = 1;
    cached_legal_actions_.reset();
    return;
  }

  Move move = ActionToMove(action, Board());
  moves_history_.push_back(move);
  Board().ApplyMove(move);
  ++repetitions_[current_board_.HashValue()];
  cached_legal_actions_.reset();
}

void ChessState::MaybeGenerateLegalActions() const {
  if (!cached_legal_actions_) {
    cached_legal_actions_ = std::vector<Action>();
    Board().GenerateLegalMoves([this](const Move& move) -> bool {
      cached_legal_actions_->push_back(MoveToAction(move, kMaxBoardSize));
      return true;
    });
    absl::c_sort(*cached_legal_actions_);
  }
}

std::vector<Action> ChessState::LegalActions() const {
  if (IsChanceNode()) {
    return LegalChanceOutcomes();
  }  // chess960.
  MaybeGenerateLegalActions();
  if (IsTerminal()) return {};
  return *cached_legal_actions_;
}

int EncodeMove(const Square& from_square, int destination_index, int board_size,
               int num_actions_destinations) {
  return (from_square.x * board_size + from_square.y) *
             num_actions_destinations +
         destination_index;
}

int8_t ReflectRank(Color to_play, int board_size, int8_t rank) {
  return to_play == Color::kBlack ? board_size - 1 - rank : rank;
}

Color PlayerToColor(Player p) {
  SPIEL_CHECK_NE(p, kInvalidPlayer);
  return static_cast<Color>(p);
}

Action MoveToAction(const Move& move, int board_size) {
  // Special-case for pass move.
  if (move == kPassMove) return kPassAction;

  if (move.is_castling()) {
    if (move.castle_dir == CastlingDirection::kLeft) {
      return kLeftCastlingAction;
    } else if (move.castle_dir == CastlingDirection::kRight) {
      return kRightCastlingAction;
    } else {
      SpielFatalError("Invalid castling move.");
    }
  }

  Color color = move.piece.color;
  // We rotate the move to be from player p's perspective.
  Move player_move(move);

  // Rotate move to be from player p's perspective.
  player_move.from.y = ReflectRank(color, board_size, player_move.from.y);
  player_move.to.y = ReflectRank(color, board_size, player_move.to.y);

  // For each starting square, we enumerate 73 actions:
  // - 9 possible underpromotions
  // - 56 queen moves
  // - 8 knight moves
  // In total, this results in 64 * 73 = 4672 indices.
  // This is the union of all possible moves, by reducing this to the number of
  // moves actually available from each starting square this could still be
  // reduced a little to 1816 indices.
  int starting_index =
      EncodeMove(player_move.from, 0, kMaxBoardSize, kNumActionDestinations);
  int8_t x_diff = player_move.to.x - player_move.from.x;
  int8_t y_diff = player_move.to.y - player_move.from.y;
  Offset offset{x_diff, y_diff};
  bool is_under_promotion = move.promotion_type != PieceType::kEmpty &&
                            move.promotion_type != PieceType::kQueen;
  if (is_under_promotion) {
    // We have to indicate underpromotions as special moves, because in terms of
    // from/to they are identical to queen promotions.
    // For a given starting square, an underpromotion can have 3 possible
    // destination squares (straight, left diagonal, right diagonal) and 3
    // possible piece types.
    SPIEL_CHECK_EQ(move.piece.type, PieceType::kPawn);
    SPIEL_CHECK_TRUE((move.piece.color == color &&
                      player_move.from.y == board_size - 2 &&
                      player_move.to.y == board_size - 1) ||
                     (move.piece.color == OppColor(color) &&
                      player_move.from.y == 1 && player_move.to.y == 0));

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
    return starting_index +
           kUnderPromotionDirectionToOffset.size() * promotion_index +
           direction_index;
  } else {
    // For the normal moves, we simply encode starting and destination square.
    int destination_index =
        OffsetToDestinationIndex(offset, kKnightOffsets, kMaxBoardSize);
    SPIEL_CHECK_TRUE(destination_index >= 0 && destination_index < 64);
    return starting_index + kNumUnderPromotions + destination_index;
  }
}

std::pair<Square, int> ActionToDestination(int action, int board_size,
                                           int num_actions_destinations) {
  const int xy = action / num_actions_destinations;
  SPIEL_CHECK_GE(xy, 0);
  SPIEL_CHECK_LT(xy, board_size * board_size);
  const int8_t x = xy / board_size;
  const int8_t y = xy % board_size;
  const int destination_index = action % num_actions_destinations;
  SPIEL_CHECK_GE(destination_index, 0);
  SPIEL_CHECK_LT(destination_index, num_actions_destinations);
  return {Square{x, y}, destination_index};
}

Move ActionToMove(const Action& action, const ChessBoard& board) {
  SPIEL_CHECK_GE(action, 0);
  SPIEL_CHECK_LT(action, NumDistinctActions());

  // Some chess variants (e.g. RBC) allow pass moves.
  if (board.AllowPassMove() && action == kPassAction) {
    return kPassMove;
  }

  // Castle actions.
  if (action == kLeftCastlingAction || action == kRightCastlingAction) {
    Square king_square = board.find(Piece{board.ToPlay(), PieceType::kKing});
    if (action == kLeftCastlingAction) {
      return Move(king_square, Square{2, king_square.y},
                  Piece{board.ToPlay(), PieceType::kKing}, PieceType::kEmpty,
                  CastlingDirection::kLeft);
    } else if (action == kRightCastlingAction) {
      return Move(king_square, Square{6, king_square.y},
                  Piece{board.ToPlay(), PieceType::kKing}, PieceType::kEmpty,
                  CastlingDirection::kRight);
    } else {
      SpielFatalError("Invalid castling move.");
    }
  }

  // The encoded action represents an action encoded from color's perspective.
  Color color = board.ToPlay();
  int board_size = board.BoardSize();
  PieceType promotion_type = PieceType::kEmpty;
  CastlingDirection castle_dir = CastlingDirection::kNone;

  auto [from_square, destination_index] =
      ActionToDestination(action, kMaxBoardSize, kNumActionDestinations);
  SPIEL_CHECK_LT(destination_index, kNumActionDestinations);

  bool is_under_promotion = destination_index < kNumUnderPromotions;
  Offset offset;
  if (is_under_promotion) {
    int promotion_index = destination_index / 3;
    int direction_index = destination_index % 3;
    promotion_type = kUnderPromotionIndexToType[promotion_index];
    offset = kUnderPromotionDirectionToOffset[direction_index];
  } else {
    destination_index -= kNumUnderPromotions;
    offset = DestinationIndexToOffset(destination_index, kKnightOffsets,
                                      kMaxBoardSize);
  }
  Square to_square = from_square + offset;

  from_square.y = ReflectRank(color, board_size, from_square.y);
  to_square.y = ReflectRank(color, board_size, to_square.y);

  // This uses the current state to infer the piece type.
  Piece piece = {board.ToPlay(), board.at(from_square).type};

  // Check for queen promotion.
  if (!is_under_promotion && piece.type == PieceType::kPawn &&
      ReflectRank(color, board_size, from_square.y) == board_size - 2 &&
      ReflectRank(color, board_size, to_square.y) == board_size - 1) {
    promotion_type = PieceType::kQueen;
  }

  Move move(from_square, to_square, piece, promotion_type, castle_dir);
  return move;
}

std::string ChessState::ActionToString(Player player, Action action) const {
  if (player == kChancePlayerId) {
    // Chess960 has an initial chance node.
    SPIEL_CHECK_GE(action, 0);
    SPIEL_CHECK_LT(action, 960);
    return absl::StrCat("ChanceNodeOutcome_", action);
  }
  Move move = ActionToMove(action, Board());
  return move.ToSAN(Board());
}

std::string ChessState::DebugString() const {
    return current_board_.DebugString(ParentGame()->IsChess960());
}

std::string ChessState::ToString() const {
  return Board().ToFEN(ParentGame()->IsChess960());
}

std::vector<double> ChessState::Returns() const {
  auto maybe_final_returns = MaybeFinalReturns();
  if (maybe_final_returns) {
    return *maybe_final_returns;
  } else {
    return {0.0, 0.0};
  }
}

std::string ChessState::InformationStateString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  return HistoryString();
}

std::string ChessState::ObservationString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  return ToString();
}

void ChessState::ObservationTensor(Player player,
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

  // Castling rights.
  AddBinaryPlane(Board().CastlingRight(Color::kWhite, CastlingDirection::kLeft),
                 value_it);

  AddBinaryPlane(
      Board().CastlingRight(Color::kWhite, CastlingDirection::kRight),
      value_it);

  AddBinaryPlane(Board().CastlingRight(Color::kBlack, CastlingDirection::kLeft),
                 value_it);

  AddBinaryPlane(
      Board().CastlingRight(Color::kBlack, CastlingDirection::kRight),
      value_it);

  SPIEL_CHECK_EQ(value_it, values.end());
}

std::unique_ptr<State> ChessState::Clone() const {
  return std::unique_ptr<State>(new ChessState(*this));
}

void ChessState::UndoAction(Player player, Action action) {
  // TODO: Make this fast by storing undo info in another stack.
  // Note: only supported after the chance node in Chess960.
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

bool ChessState::IsRepetitionDraw() const {
  const auto entry = repetitions_.find(Board().HashValue());
  SPIEL_CHECK_FALSE(entry == repetitions_.end());
  return entry->second >= kNumRepetitionsToDraw;
}

int ChessState::NumRepetitions(const ChessState& state) const {
  uint64_t state_hash_value = state.Board().HashValue();
  const auto entry = repetitions_.find(state_hash_value);
  if (entry == repetitions_.end()) {
    return 0;
  } else {
    return entry->second;
  }
}

std::pair<std::string, std::vector<std::string>>
ChessState::ExtractFenAndMaybeMoves() const {
  SPIEL_CHECK_FALSE(IsChanceNode());
  std::string initial_fen = start_board_.ToFEN(ParentGame()->IsChess960());
  std::vector<std::string> move_lans;
  std::unique_ptr<State> state = ParentGame()->NewInitialState(initial_fen);
  ChessBoard board = down_cast<const ChessState&>(*state).Board();
  for (const Move& move : moves_history_) {
    move_lans.push_back(move.ToLAN(ParentGame()->IsChess960(), &board));
    board.ApplyMove(move);
  }
  return std::make_pair(initial_fen, move_lans);
}

absl::optional<std::vector<double>> ChessState::MaybeFinalReturns() const {
  if (!Board().HasSufficientMaterial()) {
    return std::vector<double>{DrawUtility(), DrawUtility()};
  }

  if (IsRepetitionDraw()) {
    return std::vector<double>{DrawUtility(), DrawUtility()};
  }
  // Compute and cache the legal actions.
  MaybeGenerateLegalActions();
  SPIEL_CHECK_TRUE(cached_legal_actions_);
  bool have_legal_moves = !cached_legal_actions_->empty();

  // If we don't have legal moves we are either stalemated or checkmated,
  // depending on whether we are in check or not.
  if (!have_legal_moves) {
    if (!Board().InCheck()) {
      return std::vector<double>{DrawUtility(), DrawUtility()};
    } else {
      std::vector<double> returns(NumPlayers());
      auto next_to_play = ColorToPlayer(Board().ToPlay());
      returns[next_to_play] = LossUtility();
      returns[OtherPlayer(next_to_play)] = WinUtility();
      return returns;
    }
  }

  if (Board().IrreversibleMoveCounter() >= kNumReversibleMovesToDraw) {
    // This is theoretically a draw that needs to be claimed, but we implement
    // it as a forced draw for now.
    return std::vector<double>{DrawUtility(), DrawUtility()};
  }

  return absl::nullopt;
}

std::string ChessState::Serialize() const {
  std::string state_str = "";
  // If the specific_initial_fen is empty, the deserializer will use the
  // default NewInitialState(). Otherwise, the deserializer will specify
  // the specific initial fen by calling NewInitialState(string).
  absl::StrAppend(&state_str, "FEN: ", specific_initial_fen_, "\n");
  std::vector<Action> history = History();
  absl::StrAppend(&state_str, absl::StrJoin(history, "\n"), "\n");
  return state_str;
}

std::string ChessState::StartFEN() const {
  return start_board_.ToFEN(ParentGame()->IsChess960());
}


ChessGame::ChessGame(const GameParameters& params)
    : Game(kGameType, params), chess960_(ParameterValue<bool>("chess960")) {
  if (chess960_) {
    initial_fens_ = Chess960StartingPositions();
    SPIEL_CHECK_EQ(initial_fens_.size(), 960);
  }
}

std::unique_ptr<State> ChessGame::DeserializeState(
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

int ChessGame::MaxChanceOutcomes() const {
  if (IsChess960()) {
    return 960;
  } else {
    return 0;
  }
}

}  // namespace chess
}  // namespace open_spiel
