// Copyright 2019 DeepMind Technologies Ltd. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "open_spiel/games/dark_chess.h"

#include <algorithm>
#include <memory>
#include <utility>
#include <vector>

#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace dark_chess {
namespace {

constexpr int kNumReversibleMovesToDraw = 100;
constexpr int kNumRepetitionsToDraw = 3;

// Facts about the game.
const GameType kGameType{
    /*short_name=*/"dark_chess",
    /*long_name=*/"Dark Chess",
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

std::shared_ptr<const Game> Factory(const GameParameters &params) {
  return std::shared_ptr<const Game>(new DarkChessGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory)

// Adds a plane to the information state vector corresponding to the presence
// and absence of the given piece type and colour at each square.
void AddPieceTypePlane(chess::Color color, chess::PieceType piece_type,
                       const chess::StandardDarkChessBoard& board,
                       absl::Span<float>::iterator& value_it) {
  for (int8_t y = 0; y < BoardSize(); ++y) {
    for (int8_t x = 0; x < BoardSize(); ++x) {
      chess::Piece piece_on_board = board.at(chess::Square{x, y});
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
  for (int i = 0; i < BoardSize() * BoardSize(); ++i)
    *value_it++ = normalized_val;
}

// Adds a binary scalar plane.
void AddBinaryPlane(bool val, absl::Span<float>::iterator& value_it) {
  AddScalarPlane<int>(val ? 1 : 0, 0, 1, value_it);
}

}  // namespace


DarkChessState::DarkChessState(std::shared_ptr<const Game> game)
    : State(game),
      start_board_(chess::MakeDefaultDarkChessBoard()),
      current_board_(start_board_) {
  repetitions_[current_board_.HashValue()] = 1;
}

DarkChessState::DarkChessState(std::shared_ptr<const Game> game, const std::string& fen)
    : State(game) {
  auto maybe_board = chess::StandardDarkChessBoard::BoardFromFEN(fen);
  SPIEL_CHECK_TRUE(maybe_board);
  start_board_ = *maybe_board;
  current_board_ = start_board_;
  repetitions_[current_board_.HashValue()] = 1;
}

void DarkChessState::DoApplyAction(Action action) {
  chess::Move move = ActionToMove(action, Board());
  moves_history_.push_back(move);
  Board().ApplyMove(move);
  ++repetitions_[current_board_.HashValue()];
  cached_legal_actions_.reset();
}

void DarkChessState::MaybeGenerateLegalActions() const {
  if (!cached_legal_actions_) {
    cached_legal_actions_ = std::vector<Action>();
    Board().GenerateLegalMoves([this](const chess::Move& move) -> bool {
      cached_legal_actions_->push_back(MoveToAction(move));
      return true;
    });
    absl::c_sort(*cached_legal_actions_);
  }
}

std::vector<Action> DarkChessState::LegalActions() const {
  MaybeGenerateLegalActions();
  if (IsTerminal()) return {};
  return *cached_legal_actions_;
}

std::pair<chess::Square, int> ActionToDestination(int action, int board_size,
                                           int num_actions_destinations) {
  const int xy = action / num_actions_destinations;
  SPIEL_CHECK_GE(xy, 0);
  SPIEL_CHECK_LT(xy, board_size * board_size);
  const int8_t x = xy / board_size;
  const int8_t y = xy % board_size;
  const int destination_index = action % num_actions_destinations;
  SPIEL_CHECK_GE(destination_index, 0);
  SPIEL_CHECK_LT(destination_index, num_actions_destinations);
  return {chess::Square{x, y}, destination_index};
}

chess::Move ActionToMove(const Action& action, const chess::StandardDarkChessBoard& board) {
  SPIEL_CHECK_GE(action, 0);
  SPIEL_CHECK_LT(action, NumDistinctActions());

  // The encoded action represents an action encoded from color's perspective.
  chess::Color color = board.ToPlay();
  chess::PieceType promotion_type = chess::PieceType::kEmpty;
  bool is_castling = false;
  auto [from_square, destination_index] =
  ActionToDestination(action, BoardSize(), kNumActionDestinations);
  SPIEL_CHECK_LT(destination_index, kNumActionDestinations);

  bool is_under_promotion = destination_index < kNumUnderPromotions;
  chess::Offset offset;
  if (is_under_promotion) {
    int promotion_index = destination_index / 3;
    int direction_index = destination_index % 3;
    promotion_type = kUnderPromotionIndexToType[promotion_index];
    offset = kUnderPromotionDirectionToOffset[direction_index];
  } else {
    destination_index -= kNumUnderPromotions;
    offset = DestinationIndexToOffset(destination_index, chess::kKnightOffsets,
                                      BoardSize());
  }
  chess::Square to_square = from_square + offset;

  from_square.y = ReflectRank(color, BoardSize(), from_square.y);
  to_square.y = ReflectRank(color, BoardSize(), to_square.y);

  // This uses the current state to infer the piece type.
  chess::Piece piece = {board.ToPlay(), board.at(from_square).type};

  // Check for queen promotion.
  if (!is_under_promotion && piece.type == chess::PieceType::kPawn &&
      ReflectRank(color, BoardSize(), from_square.y) == BoardSize() - 2 &&
      ReflectRank(color, BoardSize(), to_square.y) == BoardSize() - 1) {
    promotion_type = chess::PieceType::kQueen;
  }

  // Check for castling which is defined here just as king moves horizontally
  // by 2 spaces.
  // TODO(b/149092677): Chess no longer supports chess960. Distinguish between
  // left/right castle.
  if (piece.type == chess::PieceType::kKing && std::abs(offset.x_offset) == 2) {
    is_castling = true;
  }
  chess::Move move(from_square, to_square, piece, promotion_type, is_castling);
  return move;
}

std::string DarkChessState::ActionToString(Player player, Action action) const {
  chess::Move move = ActionToMove(action, Board());
  return "a";
}

std::string DarkChessState::ToString() const { return Board().ToFEN(); }

std::vector<double> DarkChessState::Returns() const {
  auto maybe_final_returns = MaybeFinalReturns();
  if (maybe_final_returns) {
    return *maybe_final_returns;
  } else {
    return {0.0, 0.0};
  }
}

std::string DarkChessState::InformationStateString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  return HistoryString();
}

std::string DarkChessState::ObservationString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  return ToString();
}

void DarkChessState::ObservationTensor(Player player,
                                   absl::Span<float> values) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);

  auto value_it = values.begin();

  // Piece cconfiguration.
  for (const auto& piece_type : chess::kPieceTypes) {
    AddPieceTypePlane(chess::Color::kWhite, piece_type, Board(), value_it);
    AddPieceTypePlane(chess::Color::kBlack, piece_type, Board(), value_it);
  }

  AddPieceTypePlane(chess::Color::kEmpty, chess::PieceType::kEmpty, Board(), value_it);

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
  AddBinaryPlane(Board().CastlingRight(chess::Color::kWhite, chess::CastlingDirection::kLeft),
                 value_it);

  AddBinaryPlane(
      Board().CastlingRight(chess::Color::kWhite, chess::CastlingDirection::kRight),
      value_it);

  AddBinaryPlane(Board().CastlingRight(chess::Color::kBlack, chess::CastlingDirection::kLeft),
                 value_it);

  AddBinaryPlane(
      Board().CastlingRight(chess::Color::kBlack, chess::CastlingDirection::kRight),
      value_it);

  SPIEL_CHECK_EQ(value_it, values.end());
}

std::unique_ptr<State> DarkChessState::Clone() const {
  return std::unique_ptr<State>(new DarkChessState(*this));
}

void DarkChessState::UndoAction(Player player, Action action) {
  // TODO: Make this fast by storing undo info in another stack.
  SPIEL_CHECK_GE(moves_history_.size(), 1);
  --repetitions_[current_board_.HashValue()];
  moves_history_.pop_back();
  history_.pop_back();
  current_board_ = start_board_;
  for (const chess::Move& move : moves_history_) {
    current_board_.ApplyMove(move);
  }
}

bool DarkChessState::IsRepetitionDraw() const {
  const auto entry = repetitions_.find(Board().HashValue());
  SPIEL_CHECK_FALSE(entry == repetitions_.end());
  return entry->second >= kNumRepetitionsToDraw;
}

absl::optional<std::vector<double>> DarkChessState::MaybeFinalReturns() const {

  auto to_play_color = Board().ToPlay();
  auto opp_color = chess::OppColor(to_play_color);

  auto to_play_king = chess::Piece{to_play_color, chess::PieceType::kKing};
  auto opp_king = chess::Piece{opp_color, chess::PieceType::kKing};

  if (Board().find(to_play_king) == chess::InvalidSquare()) {

    std::vector<double> returns(NumPlayers());
    returns[chess::ColorToPlayer(to_play_color)] = LossUtility();
    returns[chess::ColorToPlayer(opp_color)] = WinUtility();
    return returns;
  }
  else if (Board().find(opp_king) == chess::InvalidSquare()) {

    std::vector<double> returns(NumPlayers());
    returns[chess::ColorToPlayer(to_play_color)] = WinUtility();
    returns[chess::ColorToPlayer(opp_color)] = LossUtility();
    return returns;
  }

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

  // If we don't have legal moves we are stalemated
  if (!have_legal_moves) {
    return std::vector<double>{DrawUtility(), DrawUtility()};
  }

  if (Board().IrreversibleMoveCounter() >= kNumReversibleMovesToDraw) {
    // This is theoretically a draw that needs to be claimed, but we implement
    // it as a forced draw for now.
    return std::vector<double>{DrawUtility(), DrawUtility()};
  }

  return std::nullopt;
}

DarkChessGame::DarkChessGame(const GameParameters& params) : Game(kGameType, params) {}



}  // namespace dark_chess
}  // namespace open_spiel
