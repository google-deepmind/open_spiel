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

#include "open_spiel/games/chess.h"

#include "open_spiel/abseil-cpp/absl/algorithm/container.h"
#include "open_spiel/abseil-cpp/absl/types/optional.h"
#include "open_spiel/games/chess/chess_board.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_globals.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace chess {
namespace {

constexpr int kNumReversibleMovesToDraw = 100;
constexpr int kNumRepetitionsToDraw = 3;

// Facts about the game
const GameType kGameType{
    /*short_name=*/"chess",
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
    /*parameter_specification=*/{}  // no parameters
};

std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::shared_ptr<const Game>(new ChessGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);

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
  auto maybe_board = ChessBoard::BoardFromFEN(fen);
  SPIEL_CHECK_TRUE(maybe_board);
  start_board_ = *maybe_board;
  current_board_ = start_board_;
  repetitions_[current_board_.HashValue()] = 1;
}

Action ChessState::ParseMoveToAction(const std::string& move_str) const {
  absl::optional<Move> move = Board().ParseMove(move_str);
  if (!move.has_value()) {
    return kInvalidAction;
  }
  return MoveToAction(*move, BoardSize());
}

void ChessState::DoApplyAction(Action action) {
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

  // The encoded action represents an action encoded from color's perspective.
  Color color = board.ToPlay();
  int board_size = board.BoardSize();
  PieceType promotion_type = PieceType::kEmpty;
  bool is_castling = false;
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

  // Check for castling which is defined here just as king moves horizontally
  // by 2 spaces.
  // TODO(b/149092677): Chess no longer supports chess960. Distinguish between
  // left/right castle.
  if (piece.type == PieceType::kKing && std::abs(offset.x_offset) == 2) {
    is_castling = true;
  }
  Move move(from_square, to_square, piece, promotion_type, is_castling);
  return move;
}

std::string ChessState::ActionToString(Player player, Action action) const {
  Move move = ActionToMove(action, Board());
  return move.ToSAN(Board());
}

std::string ChessState::ToString() const { return Board().ToFEN(); }

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

  // Piece cconfiguration.
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

ChessGame::ChessGame(const GameParameters& params) : Game(kGameType, params) {}

}  // namespace chess
}  // namespace open_spiel
