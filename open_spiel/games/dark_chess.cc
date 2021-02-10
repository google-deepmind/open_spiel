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


std::array<bool, chess::kMaxBoardSize * chess::kMaxBoardSize> ComputePrivateInfoTable(const chess::ChessBoard& board,
                                                                                      chess::Color color,
                                                                                      std::array<bool, chess::kMaxBoardSize * chess::kMaxBoardSize> &publicInfoTable) {
  int board_size = board.BoardSize();
  std::array<bool, chess::kMaxBoardSize * chess::kMaxBoardSize> observability_table{false};
  board.GenerateLegalMoves([&](const chess::Move &move) -> bool {

    size_t to_index = chess::SquareToIndex(move.to, board_size);
    if (!publicInfoTable[to_index]) observability_table[to_index] = true;

    if (move.to == board.EpSquare() && move.piece.type == chess::PieceType::kPawn) {
      int8_t reversed_y_direction = color == chess::Color::kWhite ? -1 : 1;
      chess::Square en_passant_capture = move.to + chess::Offset{0, reversed_y_direction};
      size_t index = chess::SquareToIndex(en_passant_capture, board_size);
      if (!publicInfoTable[index]) observability_table[index] = true;
    }
    return true;
  }, color);

  for (int8_t y = 0; y < board_size; ++y) {
    for (int8_t x = 0; x < board_size; ++x) {
      chess::Square sq{x, y};
      auto &piece = board.at(sq);
      if (piece.color == color) {
        size_t index = chess::SquareToIndex(sq, board_size);
        if (!publicInfoTable[index]) observability_table[index] = true;
      }
    }
  }
  return observability_table;
}

// Checks whether the defender is under attack from the attacker when we know that attacker is under attack from the defender
bool IsUnderAttack(const chess::Square defender_square, const chess::Piece defender_piece,
                   const chess::Square attacker_square, const chess::Piece attacker_piece) {

  if (defender_piece.type == attacker_piece.type) {
    return true;
  }
  if (attacker_piece.type == chess::PieceType::kEmpty || defender_piece.type == chess::PieceType::kKnight) {
    return false;
  }

  if (defender_piece.type == chess::PieceType::kPawn) {
    return attacker_piece.type == chess::PieceType::kBishop
           || attacker_piece.type == chess::PieceType::kQueen
           || attacker_piece.type == chess::PieceType::kKing;
  }
  if (defender_piece.type == chess::PieceType::kKing) {
    if (attacker_piece.type == chess::PieceType::kQueen) {
      return true;
    }
    if (attacker_piece.type == chess::PieceType::kBishop) {
      return abs(attacker_square.x - defender_square.x) >= 1 && abs(attacker_square.y - defender_square.y) >= 1;
    }
    if (attacker_piece.type == chess::PieceType::kRook) {
      return abs(attacker_square.x - defender_square.x) == 0 || abs(attacker_square.y - defender_square.y) == 0;
    }
    if (attacker_piece.type == chess::PieceType::kPawn) {
      int8_t y_direction = attacker_piece.color == chess::Color::kWhite ? 1 : -1;
      return defender_square == attacker_square + chess::Offset{1, y_direction}
             || defender_square == attacker_square + chess::Offset{-1, y_direction};
    }
    return false;
  }
  if (defender_piece.type == chess::PieceType::kRook) {
    if (attacker_piece.type == chess::PieceType::kQueen) {
      return true;
    }
    if (attacker_piece.type == chess::PieceType::kKing) {
      return abs(attacker_square.x - defender_square.x) <= 1 && abs(attacker_square.y - defender_square.y) <= 1;
    }
    return false;
  }
  if (defender_piece.type == chess::PieceType::kBishop) {
    if (attacker_piece.type == chess::PieceType::kQueen) {
      return true;
    }
    if (attacker_piece.type == chess::PieceType::kKing) {
      return abs(attacker_square.x - defender_square.x) <= 1 && abs(attacker_square.y - defender_square.y) <= 1;
    }
    if (attacker_piece.type == chess::PieceType::kPawn) {
      int8_t y_direction = attacker_piece.color == chess::Color::kWhite ? 1 : -1;
      return defender_square == attacker_square + chess::Offset{1, y_direction}
             || defender_square == attacker_square + chess::Offset{-1, y_direction};
    }
    return false;
  }
  if (defender_piece.type == chess::PieceType::kQueen) {
    if (attacker_piece.type == chess::PieceType::kBishop) {
      return abs(attacker_square.x - defender_square.x) >= 1 && abs(attacker_square.y - defender_square.y) >= 1;
    }
    if (attacker_piece.type == chess::PieceType::kRook) {
      return abs(attacker_square.x - defender_square.x) == 0 || abs(attacker_square.y - defender_square.y) == 0;
    }
    if (attacker_piece.type == chess::PieceType::kKing) {
      return abs(attacker_square.x - defender_square.x) <= 1 && abs(attacker_square.y - defender_square.y) <= 1;
    }
    if (attacker_piece.type == chess::PieceType::kPawn) {
      int8_t y_direction = attacker_piece.color == chess::Color::kWhite ? 1 : -1;
      return defender_square == attacker_square + chess::Offset{1, y_direction}
             || defender_square == attacker_square + chess::Offset{-1, y_direction};
    }
    return false;
  }
}

// Computes which squares are public information. It does not recognize all of them. Only squares of two opponent
// pieces of the same type attacking each other
std::array<bool, chess::kMaxBoardSize * chess::kMaxBoardSize> ComputePublicInfoTable(const chess::ChessBoard& board) {
  int board_size = board.BoardSize();
  std::array<bool, chess::kMaxBoardSize * chess::kMaxBoardSize> observability_table{false};
  board.GenerateLegalMoves([&](const chess::Move &move) -> bool {

    auto from_piece = board.at(move.from);
    auto to_piece = board.at(move.to);

    if (IsUnderAttack(move.from, from_piece, move.to, to_piece)) {

      size_t from_index = chess::SquareToIndex(move.from, board_size);
      observability_table[from_index] = true;

      size_t to_index = chess::SquareToIndex(move.to, board_size);
      observability_table[to_index] = true;

      if (from_piece.type != chess::PieceType::kKnight) {
        int offset_x = 0;
        int offset_y = 0;

        int diff_x = move.to.x - move.from.x;
        if (diff_x > 0) offset_x = 1;
        else if (diff_x < 0) offset_x = -1;

        int diff_y = move.to.y - move.from.y;
        if (diff_y > 0) offset_y = 1;
        else if (diff_y < 0) offset_y = -1;
        chess::Offset offset_step = {offset_x, offset_y};

        for (chess::Square dest = move.from + offset_step; dest != move.to; dest += offset_step) {
          size_t dest_index = chess::SquareToIndex(dest, board_size);
          observability_table[dest_index] = true;
        }
      }

    }
    return true;
  }, chess::Color::kWhite);

  return observability_table;
}

}  // namespace


class DarkChessObserver : public Observer {
 public:
  explicit DarkChessObserver(IIGObservationType iig_obs_type)
      : Observer(/*has_string=*/true, /*has_tensor=*/true),
        iig_obs_type_(iig_obs_type) {}

  void WriteTensor(const State& observed_state, int player, Allocator* allocator) const override {
    const auto& state = open_spiel::down_cast<const DarkChessState&>(observed_state);
    const auto& game = open_spiel::down_cast<const DarkChessGame&>(*state.GetGame());
    SPIEL_CHECK_GE(player, 0);
    SPIEL_CHECK_LT(player, game.NumPlayers());

    if (iig_obs_type_.perfect_recall) {
      SpielFatalError("Dark chess observer tensor with perfect recall not implemented");
    }

    auto common_knowledge_table = ComputePublicInfoTable(state.Board());

    if (iig_obs_type_.public_info) {
      WritePublicInfoTensor(state, common_knowledge_table, allocator);
    }
    if (iig_obs_type_.private_info == PrivateInfoType::kSinglePlayer) {
      WritePrivateInfoTensor(state, common_knowledge_table, player, allocator);
    } else if (iig_obs_type_.private_info == PrivateInfoType::kAllPlayers) {
      for (int i = 0; i < chess::NumPlayers(); ++i) {
        WritePrivateInfoTensor(state, common_knowledge_table, i, allocator);
      }
    }
  }

  std::string StringFrom(const State& observed_state, int player) const override {
    const auto& state = open_spiel::down_cast<const DarkChessState&>(observed_state);
    const auto& game = open_spiel::down_cast<const DarkChessGame&>(*state.GetGame());
    SPIEL_CHECK_GE(player, 0);
    SPIEL_CHECK_LT(player, game.NumPlayers());
    std::string result;

    if (iig_obs_type_.perfect_recall) {
      if (iig_obs_type_.public_info && iig_obs_type_.private_info == PrivateInfoType::kSinglePlayer) {
        return state.aohs_[player].ToString();
      } else {
        SpielFatalError("Dark chess observer string with perfect recall is implemented only for the default info state observation type");
      }
    }

    if (iig_obs_type_.public_info && iig_obs_type_.private_info == PrivateInfoType::kSinglePlayer) {
      chess::Color color = chess::PlayerToColor(player);
      std::array<bool, chess::kMaxBoardSize * chess::kMaxBoardSize> empty_public_info_table{};
      auto observability_table = ComputePrivateInfoTable(state.Board(), color, empty_public_info_table);
      return state.Board().ToDarkFEN(observability_table, color);
    }
    else {
      SpielFatalError("Dark chess observer string with imperfect recall is implemented only for the default observation type");
    }

  }

 private:

  void WritePieces(chess::Color color,
                   chess::PieceType piece_type,
                   const chess::ChessBoard &board,
                   std::array<bool, chess::kMaxBoardSize * chess::kMaxBoardSize> &observability_table,
                   std::string prefix,
                   Allocator *allocator) const {

    std::string type_string = color == chess::Color::kEmpty ? "empty" : chess::PieceTypeToString(piece_type, color == chess::Color::kWhite);

    auto out = allocator->Get(prefix + "_ " + type_string + "_pieces", {board.BoardSize(), board.BoardSize()});
    for (int8_t y = 0; y < board.BoardSize(); ++y) {
      for (int8_t x = 0; x < board.BoardSize(); ++x) {
        auto square = chess::Square{x, y};
        chess::Piece piece_on_board = board.at(square);
        out.at(x ,y) =
            (piece_on_board.color == color && piece_on_board.type == piece_type && observability_table[chess::SquareToIndex(square, board.BoardSize())]
             ? 1.0f
             : 0.0f);
      }
    }
  }

  void WriteUnknownSquares(const chess::ChessBoard &board,
                           std::array<bool, chess::kMaxBoardSize * chess::kMaxBoardSize> &observability_table,
                           Allocator *allocator) const {

    auto out = allocator->Get("unknown_squares", {board.BoardSize(), board.BoardSize()});
    for (int8_t y = 0; y < board.BoardSize(); ++y) {
      for (int8_t x = 0; x < board.BoardSize(); ++x) {
        auto square = chess::Square{x, y};
        out.at(x, y) = observability_table[chess::SquareToIndex(square, board.BoardSize())] ? 0.0 : 1.0;
      }
    }
  }

  template <typename T>
  void WriteScalar(int board_size, T val, T min, T max, Allocator *allocator, const std::string &name) const {

    double normalized_val = static_cast<double>(val - min) / (max - min);
    auto out = allocator->Get(name, {board_size, board_size});
    for (int8_t y = 0; y < board_size; ++y) {
      for (int8_t x = 0; x < board_size; ++x) {
        out.at(x, y) = normalized_val;
      }
    }
  }

// Adds a binary scalar plane.
  void WriteBinary(int board_size, bool val, Allocator *allocator, const std::string &name) const {
    WriteScalar<int>(board_size, val ? 1 : 0, 0, 1, allocator, name);
  }

  void WritePrivateInfoTensor(const DarkChessState& state,
                              std::array<bool, chess::kMaxBoardSize * chess::kMaxBoardSize> &common_knowledge_table,
                              int player,
                              Allocator* allocator) const {
    chess::Color color = chess::PlayerToColor(player);
    std::string color_string = chess::ColorToString(color);

    auto private_info_table = ComputePrivateInfoTable(state.Board(), color, common_knowledge_table);

    // Piece configuration.
    for (const auto& piece_type : chess::kPieceTypes) {
      WritePieces(chess::Color::kWhite, piece_type, state.Board(), private_info_table, color_string + "_private", allocator);
      WritePieces(chess::Color::kBlack, piece_type, state.Board(), private_info_table, color_string + "_private", allocator);
    }
    WritePieces(chess::Color::kEmpty, chess::PieceType::kEmpty, state.Board(), private_info_table, color_string + "_private", allocator);
    WriteUnknownSquares(state.Board(), private_info_table, allocator);

    // Castling rights.
    WriteBinary(state.Board().BoardSize(), state.Board().CastlingRight(color, chess::CastlingDirection::kLeft),
                allocator, "left_castling");

    WriteBinary(state.Board().BoardSize(), state.Board().CastlingRight(color, chess::CastlingDirection::kRight),
                allocator, "right_castling");
  }

  void WritePublicInfoTensor(const DarkChessState &state,
                             std::array<bool, chess::kMaxBoardSize * chess::kMaxBoardSize> &public_info_table,
                             Allocator *allocator) const {

    const auto entry = state.repetitions_.find(state.Board().HashValue());
    SPIEL_CHECK_FALSE(entry == state.repetitions_.end());
    int repetitions = entry->second;

    // Piece configuration.
    for (const auto& piece_type : chess::kPieceTypes) {
      WritePieces(chess::Color::kWhite, piece_type, state.Board(), public_info_table, "public", allocator);
      WritePieces(chess::Color::kBlack, piece_type, state.Board(), public_info_table, "public", allocator);
    }
    WritePieces(chess::Color::kEmpty, chess::PieceType::kEmpty, state.Board(), public_info_table, "public", allocator);

    // Num repetitions for the current board.
    WriteScalar(state.Board().BoardSize(), repetitions, 1, 3, allocator, "repetitions");

    // Side to play.
    WriteScalar(state.Board().BoardSize(), ColorToPlayer(state.Board().ToPlay()), 0, 1, allocator, "side_to_play");

    // Irreversible move counter.
    WriteScalar(state.Board().BoardSize(), state.Board().IrreversibleMoveCounter(), 0, 101, allocator, "move_number");
  }

  IIGObservationType iig_obs_type_;
};



DarkChessState::DarkChessState(std::shared_ptr<const Game> game, int boardSize, const std::string& fen)
    : State(game),
      start_board_(*chess::ChessBoard::BoardFromFEN(fen, boardSize, true)),
      current_board_(start_board_) {
  SPIEL_CHECK_TRUE(&current_board_);
  repetitions_[current_board_.HashValue()] = 1;

  aohs_.reserve(2);
  for (Player player = 0; player < NumPlayers(); ++player) {
    std::vector<std::pair<std::optional<Action>, std::string>> aoh;
    aoh.push_back({{}, ObservationString(player)});
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
  const auto& game = open_spiel::down_cast<const DarkChessGame&>(*game_);
  return game.info_state_observer_->StringFrom(*this, player);
}

std::string DarkChessState::ObservationString(Player player) const {
  const auto& game = open_spiel::down_cast<const DarkChessGame&>(*game_);
  return game.default_observer_->StringFrom(*this, player);
}

void DarkChessState::ObservationTensor(Player player, absl::Span<float> values) const {
  ContiguousAllocator allocator(values);
  const auto& game = open_spiel::down_cast<const DarkChessGame&>(*game_);
  game.default_observer_->WriteTensor(*this, player, &allocator);
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
  default_observer_ = std::make_shared<DarkChessObserver>(kDefaultObsType);
  info_state_observer_ = std::make_shared<DarkChessObserver>(kInfoStateObsType);
}



}  // namespace dark_chess
}  // namespace open_spiel
