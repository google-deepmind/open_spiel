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

#include "open_spiel/games/chess.h"

#include <optional>

#include "open_spiel/games/chess/chess_board.h"
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
    /*provides_information_state=*/true,
    /*provides_information_state_as_normalized_vector=*/true,
    /*provides_observation=*/false,
    /*provides_observation_as_normalized_vector=*/false,
    /*parameter_specification=*/{}  // no parameters
};

std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::shared_ptr<const Game>(new ChessGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);

// Adds a plane to the information state vector corresponding to the presence
// and absence of the given piece type and colour at each square.
void AddPieceTypePlane(Color color, PieceType piece_type,
                       const StandardChessBoard& board,
                       std::vector<double>* values) {
  for (int8_t y = 0; y < BoardSize(); ++y) {
    for (int8_t x = 0; x < BoardSize(); ++x) {
      Piece piece_on_board = board.at(Square{x, y});
      values->push_back(piece_on_board.color == color &&
                                piece_on_board.type == piece_type
                            ? 1.0
                            : 0.0);
    }
  }
}

// Adds a uniform scalar plane scaled with min and max.
template <typename T>
void AddScalarPlane(T val, T min, T max, std::vector<double>* values) {
  double normalized_val = static_cast<double>(val - min) / (max - min);
  values->insert(values->end(), BoardSize() * BoardSize(), normalized_val);
}

// Adds a binary scalar plane.
void AddBinaryPlane(bool val, std::vector<double>* values) {
  AddScalarPlane<int>(val ? 1 : 0, 0, 1, values);
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
  auto maybe_board = StandardChessBoard::BoardFromFEN(fen);
  SPIEL_CHECK_TRUE(maybe_board);
  start_board_ = *maybe_board;
  current_board_ = start_board_;
  repetitions_[current_board_.HashValue()] = 1;
}

void ChessState::DoApplyAction(Action action) {
  Move move = ActionToMove(action);
  moves_history_.push_back(move);
  Board().ApplyMove(move);
  ++repetitions_[current_board_.HashValue()];
}

std::vector<Action> ChessState::LegalActions() const {
  std::vector<Action> actions;
  if (IsTerminal()) return actions;
  Board().GenerateLegalMoves([&actions](const Move& move) -> bool {
    actions.push_back(MoveToAction(move));
    return true;
  });
  std::sort(actions.begin(), actions.end());
  return actions;
}

std::string ChessState::ActionToString(Player player, Action action) const {
  Move move = ActionToMove(action);
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

std::string ChessState::InformationState(Player player) const {
  return ToString();
}

void ChessState::InformationStateAsNormalizedVector(
    Player player, std::vector<double>* values) const {
  SPIEL_CHECK_NE(player, kChancePlayerId);

  values->clear();
  values->reserve(game_->InformationStateNormalizedVectorSize());

  // Piece cconfiguration.
  for (const auto& piece_type : kPieceTypes) {
    AddPieceTypePlane(Color::kWhite, piece_type, Board(), values);
    AddPieceTypePlane(Color::kBlack, piece_type, Board(), values);
  }

  AddPieceTypePlane(Color::kEmpty, PieceType::kEmpty, Board(), values);

  const auto entry = repetitions_.find(Board().HashValue());
  SPIEL_CHECK_FALSE(entry == repetitions_.end());
  int repetitions = entry->second;

  // Num repetitions for the current board.
  AddScalarPlane(repetitions, 1, 3, values);

  // Side to play.
  AddScalarPlane(ColorToPlayer(Board().ToPlay()), 0, 1, values);

  // Irreversible move counter.
  AddScalarPlane(Board().IrreversibleMoveCounter(), 0, 101, values);

  // Castling rights.
  AddBinaryPlane(Board().CastlingRight(Color::kWhite, CastlingDirection::kLeft),
                 values);

  AddBinaryPlane(
      Board().CastlingRight(Color::kWhite, CastlingDirection::kRight), values);

  AddBinaryPlane(Board().CastlingRight(Color::kBlack, CastlingDirection::kLeft),
                 values);

  AddBinaryPlane(
      Board().CastlingRight(Color::kBlack, CastlingDirection::kRight), values);
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

std::optional<std::vector<double>> ChessState::MaybeFinalReturns() const {
  if (Board().IrreversibleMoveCounter() >= kNumReversibleMovesToDraw) {
    // This is theoretically a draw that needs to be claimed, but we implement
    // it as a forced draw for now.
    return std::vector<double>{DrawUtility(), DrawUtility()};
  }

  if (!Board().HasSufficientMaterial()) {
    return std::vector<double>{DrawUtility(), DrawUtility()};
  }

  if (IsRepetitionDraw()) {
    return std::vector<double>{DrawUtility(), DrawUtility()};
  }

  bool have_legal_moves = false;
  Board().GenerateLegalMoves([&have_legal_moves](const Move& move) -> bool {
    have_legal_moves = true;
    return false;  // No need to generate more moves.
  });

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

  return std::nullopt;
}

ChessGame::ChessGame(const GameParameters& params) : Game(kGameType, params) {}

}  // namespace chess
}  // namespace open_spiel
