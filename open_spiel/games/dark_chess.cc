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
#include <open_spiel/fog/observation_history.h>

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
    GameType::Information::kImperfectInformation,
    GameType::Utility::kZeroSum,
    GameType::RewardModel::kTerminal,
    /*max_num_players=*/2,
    /*min_num_players=*/2,
    /*provides_information_state_string=*/true,
    /*provides_information_state_tensor=*/false,
    /*provides_observation_string=*/true,
    /*provides_observation_tensor=*/true,
    /*parameter_specification=*/{
      {"board_size", GameParameter(8)},
      {"fen", GameParameter(GameParameter::Type::kString, false)}}
};

std::shared_ptr<const Game> Factory(const GameParameters &params) {
  return std::shared_ptr<const Game>(new DarkChessGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory)


std::array<bool, chess::kMaxBoardSize * chess::kMaxBoardSize> ComputeObservabilityTable(const chess::ChessBoard& board,
                                                                                        chess::Color color) {
  int board_size = board.BoardSize();
  std::array<bool, chess::kMaxBoardSize * chess::kMaxBoardSize> observability_table{false};
  board.GenerateLegalMoves([&](const chess::Move &move) -> bool {

    size_t to_index = chess::SquareToIndex(move.to, board_size);
    observability_table[to_index] = true;

    if (move.to == board.EpSquare() && move.piece.type == chess::PieceType::kPawn) {
      int8_t reversed_y_direction = color == chess::Color::kWhite ? -1 : 1;
      chess::Square en_passant_capture = move.to + chess::Offset{0, reversed_y_direction};
      size_t index = chess::SquareToIndex(en_passant_capture, board_size);
      observability_table[index] = true;
    }
    return true;
  }, color);

  for (int8_t y = 0; y < board_size; ++y) {
    for (int8_t x = 0; x < board_size; ++x) {
      chess::Square sq{x, y};
      auto &piece = board.at(sq);
      if (piece.color == color) {
        size_t index = chess::SquareToIndex(sq, board_size);
        observability_table[index] = true;
      }
    }
  }
  return observability_table;
}

// Adds a plane to the information state vector corresponding to the presence
// and absence of the given piece type and colour at each square.
void AddPieceTypePlane(chess::Color color, chess::PieceType piece_type,
                       const chess::ChessBoard& board,
                       std::array<bool, chess::kMaxBoardSize * chess::kMaxBoardSize>& observability_table,
                       absl::Span<float>::iterator& value_it) {
  for (int8_t y = 0; y < chess::kMaxBoardSize; ++y) {
    for (int8_t x = 0; x < chess::kMaxBoardSize; ++x) {
      auto square = chess::Square{x, y};
      chess::Piece piece_on_board = board.at(square);
      *value_it++ =
          (piece_on_board.color == color && piece_on_board.type == piece_type && observability_table[chess::SquareToIndex(square, board.BoardSize())]
           ? 1.0
           : 0.0);
    }
  }
}

// Adds a plane to the information state vector corresponding to the observability of given square
void AddUnknownSquarePlane(std::array<bool, chess::kMaxBoardSize * chess::kMaxBoardSize>& observability_table,
                           absl::Span<float>::iterator& value_it) {
  for (int i = 0; i < chess::kMaxBoardSize * chess::kMaxBoardSize; ++i) {
    *value_it++ = observability_table[i] ? 0.0 : 1.0;
  }
}

// Adds a uniform scalar plane scaled with min and max.
template <typename T>
void AddScalarPlane(T val, T min, T max,
                    absl::Span<float>::iterator& value_it) {
  double normalized_val = static_cast<double>(val - min) / (max - min);
  for (int i = 0; i < chess::kMaxBoardSize * chess::kMaxBoardSize; ++i)
    *value_it++ = normalized_val;
}

// Adds a binary scalar plane.
void AddBinaryPlane(bool val, absl::Span<float>::iterator& value_it) {
  AddScalarPlane<int>(val ? 1 : 0, 0, 1, value_it);
}

}  // namespace


DarkChessState::DarkChessState(std::shared_ptr<const Game> game, int boardSize, const std::string& fen)
    : State(game),
      start_board_(*chess::ChessBoard::BoardFromFEN(fen, boardSize, true)),
      current_board_(start_board_) {
  SPIEL_CHECK_TRUE(&current_board_);
  repetitions_[current_board_.HashValue()] = 1;

  aohs_.reserve(2);
  for (Player player = 0; player < NumPlayers(); ++player) {
    std::vector<std::pair<std::optional<Action>, std::string>> aoh;
    aoh.push_back({static_cast<std::optional<Action>>(std::nullopt), ObservationString(player)});
    aohs_.push_back(open_spiel::ActionObservationHistory(player, aoh));
  }
}

void DarkChessState::DoApplyAction(Action action) {
  Player current_player = CurrentPlayer();
  chess::Move move = ActionToMove(action, Board());
  moves_history_.push_back(move);
  Board().ApplyMove(move);
  ++repetitions_[current_board_.HashValue()];
  cached_legal_actions_.reset();

  for (Player player = 0; player < NumPlayers(); ++player) {
    auto a = current_player == player ? action : static_cast<std::optional<Action>>(std::nullopt);
    aohs_[player].Extend(a, ObservationString(player));
  }
}

void DarkChessState::MaybeGenerateLegalActions() const {
  if (!cached_legal_actions_) {
    cached_legal_actions_ = std::vector<Action>();
    Board().GenerateLegalMoves([this](const chess::Move& move) -> bool {
      cached_legal_actions_->push_back(MoveToAction(move, Board().BoardSize()));
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

std::string DarkChessState::ActionToString(Player player, Action action) const {
  chess::Move move = ActionToMove(action, Board());
  return move.ToSAN(Board());
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
  return aohs_[player].ToString();
}

std::string DarkChessState::ObservationString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  chess::Color color = chess::PlayerToColor(player);
  auto observability_table = ComputeObservabilityTable(Board(), color);
  return Board().ToDarkFEN(observability_table, color);
}

void DarkChessState::ObservationTensor(Player player,
                                       absl::Span<float> values) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);

  chess::Color color = chess::PlayerToColor(player);

  auto value_it = values.begin();
  auto observability_table = ComputeObservabilityTable(Board(), color);

  // Piece configuration.
  for (const auto& piece_type : chess::kPieceTypes) {
    AddPieceTypePlane(chess::Color::kWhite, piece_type, Board(), observability_table, value_it);
    AddPieceTypePlane(chess::Color::kBlack, piece_type, Board(), observability_table, value_it);
  }

  AddPieceTypePlane(chess::Color::kEmpty, chess::PieceType::kEmpty, Board(), observability_table, value_it);
  AddUnknownSquarePlane(observability_table, value_it);

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
  if (color == chess::Color::kWhite) {
    AddBinaryPlane(Board().CastlingRight(chess::Color::kWhite, chess::CastlingDirection::kLeft),
                   value_it);

    AddBinaryPlane(
        Board().CastlingRight(chess::Color::kWhite, chess::CastlingDirection::kRight),
        value_it);
  } else {
    AddBinaryPlane(Board().CastlingRight(chess::Color::kBlack, chess::CastlingDirection::kLeft),
                   value_it);

    AddBinaryPlane(
        Board().CastlingRight(chess::Color::kBlack, chess::CastlingDirection::kRight),
        value_it);
  }

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
  for (Player player = 0; player < NumPlayers(); ++player) {
    aohs_[player].RemoveLast();
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

  } else if (Board().find(opp_king) == chess::InvalidSquare()) {

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

std::string DefaultFen(int boardSize) {
  if (boardSize == 8) return chess::kDefaultStandardFEN;
  else if (boardSize == 4) return chess::kDefaultSmallFEN;
  else SpielFatalError("Only board sizes 4 and 8 have their default chessboards. For other sizes, you have to define fen");
}

DarkChessGame::DarkChessGame(const GameParameters &params)
    : Game(kGameType, params),
      board_size_(ParameterValue<int>("board_size")),
      fen_(ParameterValue<std::string>("fen", DefaultFen(board_size_))) {
}



}  // namespace dark_chess
}  // namespace open_spiel
