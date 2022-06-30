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
    GameType::Information::kImperfectInformation,
    GameType::Utility::kZeroSum,
    GameType::RewardModel::kTerminal,
    /*max_num_players=*/2,
    /*min_num_players=*/2,
    /*provides_information_state_string=*/false,
    /*provides_information_state_tensor=*/false,
    /*provides_observation_string=*/true,
    /*provides_observation_tensor=*/true,
    /*parameter_specification=*/
    {{"board_size", GameParameter(8)},
     {"fen", GameParameter(GameParameter::Type::kString, false)}}};

std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::shared_ptr<const Game>(new DarkChessGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory)

chess::ObservationTable ComputePrivateInfoTable(
    const chess::ChessBoard& board, chess::Color color,
    const chess::ObservationTable& public_info_table) {
  const int board_size = board.BoardSize();
  chess::ObservationTable observability_table{false};
  board.GenerateLegalMoves(
      [&](const chess::Move& move) -> bool {
        size_t to_index = chess::SquareToIndex(move.to, board_size);
        if (!public_info_table[to_index]) observability_table[to_index] = true;

        if (move.to == board.EpSquare() &&
            move.piece.type == chess::PieceType::kPawn) {
          int8_t reversed_y_direction = color == chess::Color::kWhite ? -1 : 1;
          chess::Square en_passant_capture =
              move.to + chess::Offset{0, reversed_y_direction};
          size_t index = chess::SquareToIndex(en_passant_capture, board_size);
          if (!public_info_table[index]) observability_table[index] = true;
        }
        return true;
      },
      color);

  for (int8_t y = 0; y < board_size; ++y) {
    for (int8_t x = 0; x < board_size; ++x) {
      chess::Square sq{x, y};
      const auto& piece = board.at(sq);
      if (piece.color == color) {
        size_t index = chess::SquareToIndex(sq, board_size);
        if (!public_info_table[index]) observability_table[index] = true;
      }
    }
  }
  return observability_table;
}

// Checks whether the defender is under attack from the attacker,
// for the special case when we already know that attacker is under attack
// from the defender.
// I.e.  D -> A, but D <-? A  (where arrow is the "under attack relation")
// This is used for computation of the public info table.
bool IsUnderAttack(const chess::Square defender_sq,
                   const chess::Piece defender_piece,
                   const chess::Square attacker_sq,
                   const chess::Piece attacker_piece) {
  // Identity: i.e. we only check distinct piece types from now on.
  if (defender_piece.type == attacker_piece.type) {
    return true;
  }
  // No need to check empty attackers from now on.
  if (attacker_piece.type == chess::PieceType::kEmpty) {
    return false;
  }

  const auto pawn_attack = [&]() {
    int8_t y_dir = attacker_piece.color == chess::Color::kWhite ? 1 : -1;
    return defender_sq == attacker_sq + chess::Offset{1, y_dir} ||
           defender_sq == attacker_sq + chess::Offset{-1, y_dir};
  };
  const auto king_attack = [&]() {
    return abs(attacker_sq.x - defender_sq.x) <= 1 &&
           abs(attacker_sq.y - defender_sq.y) <= 1;
  };
  const auto rook_attack = [&]() {
    return abs(attacker_sq.x - defender_sq.x) == 0 ||
           abs(attacker_sq.y - defender_sq.y) == 0;
  };
  const auto bishop_attack = [&]() {
    return abs(attacker_sq.x - defender_sq.x) >= 1 &&
           abs(attacker_sq.y - defender_sq.y) >= 1;
  };

  switch (defender_piece.type) {
    case chess::PieceType::kEmpty:
      SpielFatalError("Empty squares cannot be already attacking.");

    case chess::PieceType::kKing:
      switch (attacker_piece.type) {
        case chess::PieceType::kQueen:
          return true;
        case chess::PieceType::kRook:
          return rook_attack();
        case chess::PieceType::kBishop:
          return bishop_attack();
        case chess::PieceType::kKnight:
          return false;
        case chess::PieceType::kPawn:
          return pawn_attack();
        default:
          SpielFatalError("Exhausted match");
      }

    case chess::PieceType::kQueen:
      switch (attacker_piece.type) {
        case chess::PieceType::kKing:
          return king_attack();
        case chess::PieceType::kRook:
          return rook_attack();
        case chess::PieceType::kBishop:
          return bishop_attack();
        case chess::PieceType::kKnight:
          return false;
        case chess::PieceType::kPawn:
          return pawn_attack();
        default:
          SpielFatalError("Exhausted match");
      }

    case chess::PieceType::kRook:
      switch (attacker_piece.type) {
        case chess::PieceType::kKing:
          return king_attack();
        case chess::PieceType::kQueen:
          return true;
        default:
          return false;
      }

    case chess::PieceType::kBishop:
      switch (attacker_piece.type) {
        case chess::PieceType::kKing:
          return king_attack();
        case chess::PieceType::kQueen:
          return true;
        case chess::PieceType::kPawn:
          return pawn_attack();
        default:
          return false;
      }

    case chess::PieceType::kKnight:
      return false;

    case chess::PieceType::kPawn:
      return attacker_piece.type == chess::PieceType::kKing ||
             attacker_piece.type == chess::PieceType::kQueen ||
             attacker_piece.type == chess::PieceType::kBishop;

    default:
      // This should not happen, we cover all the possibilities.
      SpielFatalError("Exhausted pattern match in dark_chess::IsUnderAttack()");
  }
}

// Computes which squares are public information. It does not recognize all of
// them. Only squares of two opponent pieces of the same type attacking each
// other.
chess::ObservationTable ComputePublicInfoTable(const chess::ChessBoard& board) {
  const int board_size = board.BoardSize();
  std::array<bool, chess::k2dMaxBoardSize> observability_table{false};
  board.GenerateLegalMoves(
      [&](const chess::Move& move) -> bool {
        const chess::Piece& from_piece = board.at(move.from);
        const chess::Piece& to_piece = board.at(move.to);

        if (IsUnderAttack(move.from, from_piece, move.to, to_piece)) {
          size_t from_index = chess::SquareToIndex(move.from, board_size);
          observability_table[from_index] = true;

          size_t to_index = chess::SquareToIndex(move.to, board_size);
          observability_table[to_index] = true;

          // Fill the table also between the indices.
          if (from_piece.type != chess::PieceType::kKnight) {
            int offset_x = 0;
            int offset_y = 0;

            int diff_x = move.to.x - move.from.x;
            if (diff_x > 0)
              offset_x = 1;
            else if (diff_x < 0)
              offset_x = -1;

            int diff_y = move.to.y - move.from.y;
            if (diff_y > 0)
              offset_y = 1;
            else if (diff_y < 0)
              offset_y = -1;
            chess::Offset offset_step = {
              static_cast<int8_t>(offset_x),
              static_cast<int8_t>(offset_y)
            };

            for (chess::Square dest = move.from + offset_step; dest != move.to;
                 dest += offset_step) {
              size_t dest_index = chess::SquareToIndex(dest, board_size);
              observability_table[dest_index] = true;
            }
          }
        }
        return true;
      },
      chess::Color::kWhite);

  return observability_table;
}

bool ObserverHasString(IIGObservationType iig_obs_type) {
  return iig_obs_type.public_info &&
         iig_obs_type.private_info == PrivateInfoType::kSinglePlayer &&
         !iig_obs_type.perfect_recall;
}
bool ObserverHasTensor(IIGObservationType iig_obs_type) {
  return !iig_obs_type.perfect_recall;
}

}  // namespace

class DarkChessObserver : public Observer {
 public:
  explicit DarkChessObserver(IIGObservationType iig_obs_type)
      : Observer(/*has_string=*/ObserverHasString(iig_obs_type),
                 /*has_tensor=*/ObserverHasTensor(iig_obs_type)),
        iig_obs_type_(iig_obs_type) {}

  void WriteTensor(const State& observed_state, int player,
                   Allocator* allocator) const override {
    auto& state = open_spiel::down_cast<const DarkChessState&>(observed_state);
    auto& game = open_spiel::down_cast<const DarkChessGame&>(*state.GetGame());
    SPIEL_CHECK_GE(player, 0);
    SPIEL_CHECK_LT(player, game.NumPlayers());

    if (iig_obs_type_.perfect_recall) {
      SpielFatalError(
          "DarkChessObserver: tensor with perfect recall not implemented.");
    }

    const auto public_info_table = ComputePublicInfoTable(state.Board());

    if (iig_obs_type_.public_info) {
      WritePublicInfoTensor(state, public_info_table, allocator);
    }
    if (iig_obs_type_.private_info == PrivateInfoType::kSinglePlayer) {
      std::string prefix = "private";
      WritePrivateInfoTensor(state, public_info_table, player, prefix,
                             allocator);
    } else if (iig_obs_type_.private_info == PrivateInfoType::kAllPlayers) {
      for (int i = 0; i < chess::NumPlayers(); ++i) {
        chess::Color color = chess::PlayerToColor(player);
        std::string prefix = chess::ColorToString(color);
        WritePrivateInfoTensor(state, public_info_table, i, prefix, allocator);
      }
    }
  }

  std::string StringFrom(const State& observed_state,
                         int player) const override {
    auto& state = open_spiel::down_cast<const DarkChessState&>(observed_state);
    auto& game = open_spiel::down_cast<const DarkChessGame&>(*state.GetGame());
    SPIEL_CHECK_GE(player, 0);
    SPIEL_CHECK_LT(player, game.NumPlayers());

    if (iig_obs_type_.perfect_recall) {
      SpielFatalError(
          "DarkChessObserver: string with perfect recall is not supported");
    }

    if (iig_obs_type_.public_info &&
        iig_obs_type_.private_info == PrivateInfoType::kSinglePlayer) {
      chess::Color color = chess::PlayerToColor(player);
      chess::ObservationTable empty_public_info_table{};
      auto obs_table = ComputePrivateInfoTable(state.Board(), color,
                                               empty_public_info_table);
      return state.Board().ToDarkFEN(obs_table, color);
    } else {
      SpielFatalError(
          "DarkChessObserver: string with imperfect recall is implemented only"
          " for the (default) observation type.");
    }
  }

 private:
  void WritePieces(chess::Color color, chess::PieceType piece_type,
                   const chess::ChessBoard& board,
                   const chess::ObservationTable& observability_table,
                   const std::string& prefix, Allocator* allocator) const {
    const std::string type_string =
        color == chess::Color::kEmpty
            ? "empty"
            : chess::PieceTypeToString(
                  piece_type,
                  /*uppercase=*/color == chess::Color::kWhite);
    const int board_size = board.BoardSize();

    auto out = allocator->Get(prefix + "_" + type_string + "_pieces",
                              {board_size, board_size});
    for (int8_t y = 0; y < board_size; ++y) {
      for (int8_t x = 0; x < board_size; ++x) {
        const chess::Square square{x, y};
        const chess::Piece& piece_on_board = board.at(square);
        const bool write_square =
            piece_on_board.color == color &&
            piece_on_board.type == piece_type &&
            observability_table[chess::SquareToIndex(square, board_size)];
        out.at(x, y) = write_square ? 1.0f : 0.0f;
      }
    }
  }

  void WriteUnknownSquares(const chess::ChessBoard& board,
                           chess::ObservationTable& observability_table,
                           const std::string& prefix,
                           Allocator* allocator) const {
    const int board_size = board.BoardSize();
    auto out = allocator->Get(prefix + "_unknown_squares",
                              {board.BoardSize(), board.BoardSize()});
    for (int8_t y = 0; y < board_size; ++y) {
      for (int8_t x = 0; x < board_size; ++x) {
        const chess::Square square{x, y};
        const bool write_square =
            observability_table[chess::SquareToIndex(square, board_size)];
        out.at(x, y) = write_square ? 0.0f : 1.0f;
      }
    }
  }

  void WriteScalar(int val, int min, int max, const std::string& field_name,
                   Allocator* allocator) const {
    SPIEL_DCHECK_LT(min, max);
    SPIEL_DCHECK_GE(val, min);
    SPIEL_DCHECK_LE(val, max);
    auto out = allocator->Get(field_name, {max - min + 1});
    out.at(val - min) = 1;
  }

  // Adds a binary scalar plane.
  void WriteBinary(bool val, const std::string& field_name,
                   Allocator* allocator) const {
    WriteScalar(val ? 1 : 0, 0, 1, field_name, allocator);
  }

  void WritePrivateInfoTensor(const DarkChessState& state,
                              const chess::ObservationTable& public_info_table,
                              int player, const std::string& prefix,
                              Allocator* allocator) const {
    chess::Color color = chess::PlayerToColor(player);
    chess::ObservationTable private_info_table =
        ComputePrivateInfoTable(state.Board(), color, public_info_table);

    // Piece configuration.
    for (const chess::PieceType& piece_type : chess::kPieceTypes) {
      WritePieces(chess::Color::kWhite, piece_type, state.Board(),
                  private_info_table, prefix, allocator);
      WritePieces(chess::Color::kBlack, piece_type, state.Board(),
                  private_info_table, prefix, allocator);
    }
    WritePieces(chess::Color::kEmpty, chess::PieceType::kEmpty, state.Board(),
                private_info_table, prefix, allocator);
    WriteUnknownSquares(state.Board(), private_info_table, prefix, allocator);

    // Castling rights.
    WriteBinary(
        state.Board().CastlingRight(color, chess::CastlingDirection::kLeft),
        prefix + "_left_castling", allocator);
    WriteBinary(
        state.Board().CastlingRight(color, chess::CastlingDirection::kRight),
        prefix + "_right_castling", allocator);
  }

  void WritePublicInfoTensor(const DarkChessState& state,
                             const chess::ObservationTable& public_info_table,
                             Allocator* allocator) const {
    const auto entry = state.repetitions_.find(state.Board().HashValue());
    SPIEL_CHECK_FALSE(entry == state.repetitions_.end());
    int repetitions = entry->second;

    // Piece configuration.
    std::string prefix = "public";
    for (const chess::PieceType& piece_type : chess::kPieceTypes) {
      WritePieces(chess::Color::kWhite, piece_type, state.Board(),
                  public_info_table, prefix, allocator);
      WritePieces(chess::Color::kBlack, piece_type, state.Board(),
                  public_info_table, prefix, allocator);
    }
    WritePieces(chess::Color::kEmpty, chess::PieceType::kEmpty, state.Board(),
                public_info_table, prefix, allocator);

    // Num repetitions for the current board.
    WriteScalar(/*val=*/repetitions, /*min=*/1, /*max=*/3, "repetitions",
                allocator);

    // Side to play.
    WriteScalar(/*val=*/ColorToPlayer(state.Board().ToPlay()),
                /*min=*/0, /*max=*/1, "side_to_play", allocator);

    // Irreversible move counter.
    auto out = allocator->Get("irreversible_move_counter", {1});
    out.at(0) = state.Board().IrreversibleMoveCounter() / 100.;
  }

  IIGObservationType iig_obs_type_;
};

DarkChessState::DarkChessState(std::shared_ptr<const Game> game, int board_size,
                               const std::string& fen)
    : State(game),
      start_board_(*chess::ChessBoard::BoardFromFEN(fen, board_size, true)),
      current_board_(start_board_) {
  SPIEL_CHECK_TRUE(&current_board_);
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
      cached_legal_actions_->push_back(MoveToAction(move, BoardSize()));
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

std::string DarkChessState::ObservationString(Player player) const {
  const auto& game = open_spiel::down_cast<const DarkChessGame&>(*game_);
  return game.default_observer_->StringFrom(*this, player);
}

void DarkChessState::ObservationTensor(Player player,
                                       absl::Span<float> values) const {
  ContiguousAllocator allocator(values);
  const auto& game = open_spiel::down_cast<const DarkChessGame&>(*game_);
  game.default_observer_->WriteTensor(*this, player, &allocator);
}

std::unique_ptr<State> DarkChessState::Clone() const {
  return std::make_unique<DarkChessState>(*this);
}

void DarkChessState::UndoAction(Player player, Action action) {
  // TODO: Make this fast by storing undo info in another stack.
  SPIEL_CHECK_GE(moves_history_.size(), 1);
  --repetitions_[current_board_.HashValue()];
  moves_history_.pop_back();
  history_.pop_back();
  --move_number_;
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
  const auto to_play_color = Board().ToPlay();
  const auto opp_color = chess::OppColor(to_play_color);

  const auto to_play_king =
      chess::Piece{to_play_color, chess::PieceType::kKing};
  const auto opp_king = chess::Piece{opp_color, chess::PieceType::kKing};

  if (Board().find(to_play_king) == chess::kInvalidSquare) {
    std::vector<double> returns(NumPlayers());
    returns[chess::ColorToPlayer(to_play_color)] = LossUtility();
    returns[chess::ColorToPlayer(opp_color)] = WinUtility();
    return returns;

  } else if (Board().find(opp_king) == chess::kInvalidSquare) {
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
  const bool have_legal_moves = !cached_legal_actions_->empty();

  // If we don't have legal moves we are stalemated
  if (!have_legal_moves) {
    return std::vector<double>{DrawUtility(), DrawUtility()};
  }

  if (Board().IrreversibleMoveCounter() >= kNumReversibleMovesToDraw) {
    // This is theoretically a draw that needs to be claimed, but we implement
    // it as a forced draw for now.
    return std::vector<double>{DrawUtility(), DrawUtility()};
  }

  return absl::nullopt;
}

DarkChessGame::DarkChessGame(const GameParameters& params)
    : Game(kGameType, params),
      board_size_(ParameterValue<int>("board_size")),
      fen_(ParameterValue<std::string>("fen", chess::DefaultFen(board_size_))) {
  default_observer_ = std::make_shared<DarkChessObserver>(kDefaultObsType);
}

std::shared_ptr<Observer> DarkChessGame::MakeObserver(
    absl::optional<IIGObservationType> iig_obs_type,
    const GameParameters& params) const {
  if (!params.empty()) SpielFatalError("Observation params not supported");
  return std::make_shared<DarkChessObserver>(
      iig_obs_type.value_or(kDefaultObsType));
}

}  // namespace dark_chess
}  // namespace open_spiel
