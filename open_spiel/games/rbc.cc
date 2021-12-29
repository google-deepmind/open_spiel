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

#include "open_spiel/games/rbc.h"

#include <algorithm>
#include <memory>
#include <utility>
#include <vector>

#include "open_spiel/abseil-cpp/absl/algorithm/container.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace rbc {
namespace {

constexpr int kNumReversibleMovesToDraw = 100;
constexpr int kNumRepetitionsToDraw = 3;

// Facts about the game.
const GameType kGameType{
    /*short_name=*/"rbc",
    /*long_name=*/"Reconnaisance Blind Chess",
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
     {"sense_size", GameParameter(3)},
     {"fen", GameParameter(GameParameter::Type::kString, false)}}};

std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::make_shared<RbcGame>(params);
}

REGISTER_SPIEL_GAME(kGameType, Factory)

chess::ObservationTable ComputeObservationTable(const chess::ChessBoard& board,
                                                chess::Color color,
                                                int sense_location,
                                                int sense_size) {
  const int board_size = board.BoardSize();
  chess::ObservationTable observability_table{false};

  // Player pieces.
  for (int8_t y = 0; y < board_size; ++y) {
    for (int8_t x = 0; x < board_size; ++x) {
      chess::Square sq{x, y};
      const auto& piece = board.at(sq);
      if (piece.color == color) {
        size_t index = chess::SquareToIndex(sq, board_size);
        observability_table[index] = true;
      }
    }
  }

  // No sense window specified.
  if (sense_location < 0) return observability_table;

  // All pieces under the sense window.
  int inner_size = board_size - sense_size + 1;
  chess::Square sense_sq = chess::IndexToSquare(sense_location, inner_size);
  SPIEL_DCHECK_LE(sense_sq.x + sense_size, board_size);
  SPIEL_DCHECK_LE(sense_sq.y + sense_size, board_size);
  for (int8_t x = sense_sq.x; x < sense_sq.x + sense_size; ++x) {
    for (int8_t y = sense_sq.y; y < sense_sq.y + sense_size; ++y) {
      const chess::Square sq{x, y};
      size_t index = chess::SquareToIndex(sq, board_size);
      observability_table[index] = true;
    }
  }

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

class RbcObserver : public Observer {
 public:
  explicit RbcObserver(IIGObservationType iig_obs_type)
      : Observer(/*has_string=*/ObserverHasString(iig_obs_type),
                 /*has_tensor=*/ObserverHasTensor(iig_obs_type)),
        iig_obs_type_(iig_obs_type) {}

  void WriteTensor(const State& observed_state, int player,
                   Allocator* allocator) const override {
    auto& state = open_spiel::down_cast<const RbcState&>(observed_state);
    auto& game = open_spiel::down_cast<const RbcGame&>(*state.GetGame());
    SPIEL_CHECK_GE(player, 0);
    SPIEL_CHECK_LT(player, game.NumPlayers());

    if (iig_obs_type_.perfect_recall) {
      SpielFatalError(
          "RbcObserver: tensor with perfect recall not implemented.");
    }

    if (iig_obs_type_.public_info) {
      WritePublicInfoTensor(state, allocator);
    }
    if (iig_obs_type_.private_info == PrivateInfoType::kSinglePlayer) {
      std::string prefix = "private";
      WritePrivateInfoTensor(state, player, prefix, allocator);
    } else if (iig_obs_type_.private_info == PrivateInfoType::kAllPlayers) {
      for (int i = 0; i < chess::NumPlayers(); ++i) {
        chess::Color color = chess::PlayerToColor(player);
        std::string prefix = chess::ColorToString(color);
        WritePrivateInfoTensor(state, i, prefix, allocator);
      }
    }
  }

  std::string StringFrom(const State& observed_state,
                         int player) const override {
    auto& state = open_spiel::down_cast<const RbcState&>(observed_state);
    auto& game = open_spiel::down_cast<const RbcGame&>(*state.GetGame());
    SPIEL_CHECK_GE(player, 0);
    SPIEL_CHECK_LT(player, game.NumPlayers());

    if (iig_obs_type_.perfect_recall) {
      SpielFatalError(
          "RbcObserver: string with perfect recall is not supported");
    }

    if (iig_obs_type_.private_info == PrivateInfoType::kSinglePlayer) {
      return StringPrivateInfoObservation(state, game, player);
    } else {
      SpielFatalError(
          "RbcObserver: string with imperfect recall is implemented only"
          " for the (default) observation type.");
    }
  }

 private:
  // Encode the private observations as a FEN-like string.
  std::string StringPrivateInfoObservation(const RbcState& state,
                                           const RbcGame& game,
                                           int player) const {
    chess::Color color = chess::PlayerToColor(player);
    const int sense_location =
        (state.phase_ == MovePhase::kMoving && state.CurrentPlayer() == player)
            ? state.sense_location_[player]
            // Make sure that sense from last round does not
            // reveal a new hidden move: allow players to
            // perceive only results of the last sensing.
            : kSenseLocationNonSpecified;
    const chess::ChessBoard& board = state.Board();
    chess::ObservationTable observability_table = ComputeObservationTable(
        board, color, sense_location, game.sense_size());
    const int board_size = game.board_size();
    std::string str = "";

    // 1. Encode the board based on what can be observed for the player.
    for (int8_t rank = board_size - 1; rank >= 0; --rank) {
      int num_unknown = 0;
      for (int8_t file = 0; file < board_size; ++file) {
        const size_t index =
            chess::SquareToIndex(chess::Square{file, rank}, board_size);
        if (!observability_table[index]) {
          num_unknown++;
        } else {
          if (num_unknown > 0) {
            absl::StrAppend(&str, num_unknown);
            num_unknown = 0;
          }
          const chess::Piece& piece = board.at(chess::Square{file, rank});
          absl::StrAppend(&str, piece.ToString());
        }
      }
      if (num_unknown > 0) {
        absl::StrAppend(&str, num_unknown);
      }
      if (rank > 0) {
        absl::StrAppend(&str, "/");
      }
    }

    // 2. Castling rights of the player.
    absl::StrAppend(&str, " ");
    std::string castling_rights;
    if (board.CastlingRight(color, chess::CastlingDirection::kRight)) {
      absl::StrAppend(&castling_rights, "K");
    }
    if (board.CastlingRight(color, chess::CastlingDirection::kLeft)) {
      castling_rights.push_back('Q');
    }
    absl::StrAppend(&str, castling_rights.empty() ? "-" : castling_rights);

    // 3. Phase.
    absl::StrAppend(&str, " ", state.phase_ == MovePhase::kSensing ? "s" : "m");

    // 4. Capture (but no information about what was captured).
    absl::StrAppend(&str, " ", state.move_captured_ ? "c" : "-");

    // 5. Side to play.
    absl::StrAppend(&str, " ",
                    board.ToPlay() == chess::Color::kWhite ? "w" : "b");

    // 6. Illegal move.
    const bool can_show = state.CurrentPlayer() == player;
    absl::StrAppend(&str, " ",
                    can_show && state.illegal_move_attempted_ ? "i" : "-");
    return str;
  }

  void WritePieces(chess::Color color, chess::PieceType piece_type,
                   const chess::ChessBoard& board, int sense_location,
                   int sense_size, const std::string& prefix,
                   Allocator* allocator) const {
    const std::string type_string = chess::PieceTypeToString(
        piece_type, /*uppercase=*/color == chess::Color::kWhite);
    const int board_size = board.BoardSize();
    int inner_size = board_size - sense_size + 1;
    chess::Square sense_square =
        chess::IndexToSquare(sense_location, inner_size);
    auto out = allocator->Get(prefix + "_" + type_string + "_pieces",
                              {board_size, board_size});

    if (sense_location < 0) return;  // No sense window specified.
    SPIEL_DCHECK_LE(sense_square.x + sense_size, board_size);
    SPIEL_DCHECK_LE(sense_square.y + sense_size, board_size);
    for (int8_t x = sense_square.x; x < sense_square.x + sense_size; ++x) {
      for (int8_t y = sense_square.y; y < sense_square.y + sense_size; ++y) {
        const chess::Square square{x, y};
        const chess::Piece& piece_on_board = board.at(square);
        const bool write_square =
            piece_on_board.color == color && piece_on_board.type == piece_type;
        out.at(x, y) = write_square ? 1.0f : 0.0f;
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

  void WritePrivateInfoTensor(const RbcState& state, int player,
                              const std::string& prefix,
                              Allocator* allocator) const {
    chess::Color color = chess::PlayerToColor(player);

    // Illegal move (pawn attack or pawn forward-move or castle through
    // opponent pieces).
    const bool can_show = state.CurrentPlayer() == player;
    WriteBinary(can_show && state.illegal_move_attempted_, "illegal_move",
                allocator);

    // Piece configuration.
    for (const chess::PieceType& piece_type : chess::kPieceTypes) {
      WritePieces(static_cast<chess::Color>(player), piece_type, state.Board(),
                  0, state.game()->board_size(), prefix, allocator);
    }

    // Castling rights.
    WriteBinary(
        state.Board().CastlingRight(color, chess::CastlingDirection::kLeft),
        prefix + "_left_castling", allocator);
    WriteBinary(
        state.Board().CastlingRight(color, chess::CastlingDirection::kRight),
        prefix + "_right_castling", allocator);

    // Last sensing
    for (const chess::PieceType& piece_type : chess::kPieceTypes) {
      int sense_location = (state.phase_ == MovePhase::kMoving &&
                            state.CurrentPlayer() == player)
                               ? state.sense_location_[player]
                               // Make sure that sense from last round does not
                               // reveal a new hidden move: allow players to
                               // perceive only results of the last sensing.
                               : kSenseLocationNonSpecified;
      WritePieces(static_cast<chess::Color>(1 - player), piece_type,
                  state.Board(), sense_location, state.game()->sense_size(),
                  prefix + "_sense", allocator);
    }
  }

  void WritePublicInfoTensor(const RbcState& state,
                             Allocator* allocator) const {
    // Compute number of pieces of each player.
    const int board_size = state.game()->board_size();
    std::array<int, 2> num_pieces = {0, 0};
    for (int x = 0; x < board_size; ++x) {
      for (int y = 0; y < board_size; ++y) {
        for (int pl = 0; pl < 2; ++pl) {
          num_pieces[pl] +=
              state.Board().IsFriendly(
                  chess::Square{static_cast<int8_t>(x),
                                static_cast<int8_t>(y)},
                  static_cast<chess::Color>(pl));
        }
      }
    }

    WriteScalar(num_pieces[0], 0, board_size * 2, "pieces_black", allocator);
    WriteScalar(num_pieces[1], 0, board_size * 2, "pieces_white", allocator);
    WriteBinary(state.phase_ == MovePhase::kSensing, "phase", allocator);
    WriteBinary(state.move_captured_, "capture", allocator);
    WriteBinary(state.CurrentPlayer(), "side_to_play", allocator);
  }

  IIGObservationType iig_obs_type_;
};

RbcState::RbcState(std::shared_ptr<const Game> game, int board_size,
                   const std::string& fen)
    : State(game),
      start_board_(*chess::ChessBoard::BoardFromFEN(
          fen, board_size,
          /*king_in_check_allowed=*/true,
          /*allow_pass_move=*/true)),
      current_board_(start_board_),
      phase_(MovePhase::kSensing) {
  SPIEL_CHECK_TRUE(&current_board_);
  repetitions_[current_board_.HashValue()] = 1;
}

void RbcState::DoApplyAction(Action action) {
  // Reset common flags.
  illegal_move_attempted_ = false;
  move_captured_ = false;

  if (phase_ == MovePhase::kSensing) {
    sense_location_[CurrentPlayer()] = action;
    phase_ = MovePhase::kMoving;
  } else {
    SPIEL_CHECK_TRUE(phase_ == MovePhase::kMoving);
    chess::Move move = ActionToMove(action, Board());

    // Handle special cases for RBC.

    if (move == chess::kPassMove) {
      // The RBC's pass move is handled via ChessBoard flag allow_pass_move.
      // Nothing here. Values set above.
    } else if (Board().IsBreachingMove(move)) {
      SPIEL_DCHECK_FALSE(Board().IsMoveLegal(move));
      Board().BreachingMoveToCaptureMove(&move);
      // Transformed move must be legal.
      SPIEL_DCHECK_TRUE(Board().IsMoveLegal(move));
      // And it must be a capture, since we breached unseen opponent pieces.
      SPIEL_DCHECK_NE(Board().at(move.from).color, Board().at(move.to).color);
      move_captured_ = true;
    } else if (!Board().IsMoveLegal(move)) {
      // Illegal move was chosen.
      illegal_move_attempted_ = true;

      // Check why the move was illegal: if it is pawn two-squares-forward move,
      // and there is an enemy piece blocking it, the attempt to move only one
      // square forward (if that would be a legal move).
      if (move.piece.type == chess::PieceType::kPawn &&
          abs(move.from.y - move.to.y) == 2) {
        const int dy = move.to.y - move.from.y > 0 ? 1 : -1;
        chess::Move one_forward_move = move;
        one_forward_move.to.y -= dy;
        move = Board().IsMoveLegal(one_forward_move) ? one_forward_move
                                                     : chess::kPassMove;
      } else {
        // Treat the illegal move as a pass.
        move = chess::kPassMove;
      }
    } else {
      // All other moves
      SPIEL_DCHECK_EQ(Board().at(move.from).color, Board().ToPlay());
      move_captured_ =
          Board().at(move.to).color == chess::OppColor(Board().ToPlay());
    }

    SPIEL_DCHECK_TRUE(Board().IsMoveLegal(move));
    moves_history_.push_back(move);
    Board().ApplyMove(move);

    ++repetitions_[current_board_.HashValue()];
    phase_ = MovePhase::kSensing;
  }
  cached_legal_actions_.reset();
}

void RbcState::MaybeGenerateLegalActions() const {
  if (!cached_legal_actions_) {
    cached_legal_actions_ = std::vector<Action>();

    if (phase_ == MovePhase::kSensing) {
      int num_possible_sense_locations =
          game()->inner_size() * game()->inner_size();
      cached_legal_actions_->resize(num_possible_sense_locations);
      absl::c_iota(*cached_legal_actions_, 0);
    } else {
      SPIEL_CHECK_TRUE(phase_ == MovePhase::kMoving);
      Board().GeneratePseudoLegalMoves(
          [this](const chess::Move& move) -> bool {
            cached_legal_actions_->push_back(MoveToAction(move, BoardSize()));
            return true;
          },
          Board().ToPlay(), chess::PseudoLegalMoveSettings::kBreachEnemyPieces);
      absl::c_sort(*cached_legal_actions_);
    }
  }
}

std::vector<Action> RbcState::LegalActions() const {
  if (IsTerminal()) return {};
  MaybeGenerateLegalActions();
  return *cached_legal_actions_;
}

std::string RbcState::ActionToString(Player player, Action action) const {
  if (phase_ == MovePhase::kSensing) {
    std::string from = chess::SquareToString(
        chess::IndexToSquare(action, game()->inner_size()));
    return absl::StrCat("Sense ", from);
  } else {
    if (action == chess::kPassAction) return "pass";
    chess::Move move = ActionToMove(action, Board());
    return move.ToLAN();
  }
}

std::string RbcState::ToString() const { return Board().ToFEN(); }

std::vector<double> RbcState::Returns() const {
  auto maybe_final_returns = MaybeFinalReturns();
  if (maybe_final_returns) {
    return *maybe_final_returns;
  } else {
    return {0.0, 0.0};
  }
}

std::string RbcState::ObservationString(Player player) const {
  const auto& game = open_spiel::down_cast<const RbcGame&>(*game_);
  return game.default_observer_->StringFrom(*this, player);
}

void RbcState::ObservationTensor(Player player,
                                 absl::Span<float> values) const {
  ContiguousAllocator allocator(values);
  const auto& game = open_spiel::down_cast<const RbcGame&>(*game_);
  game.default_observer_->WriteTensor(*this, player, &allocator);
}

std::unique_ptr<State> RbcState::Clone() const {
  return std::make_unique<RbcState>(*this);
}

void RbcState::UndoAction(Player player, Action action) {
  // TODO: Make this fast by storing undo info in another stack.
  SPIEL_CHECK_FALSE(history_.empty());  // Can't undo initial state.
  history_.pop_back();
  --move_number_;

  if (phase_ == MovePhase::kMoving) {
    phase_ = MovePhase::kSensing;
  } else {
    SPIEL_CHECK_GE(moves_history_.size(), 1);
    phase_ = MovePhase::kMoving;
    --repetitions_[current_board_.HashValue()];
    moves_history_.pop_back();
    current_board_ = start_board_;
    for (const chess::Move& move : moves_history_) {
      current_board_.ApplyMove(move);
    }
  }
}

bool RbcState::IsRepetitionDraw() const {
  const auto entry = repetitions_.find(Board().HashValue());
  SPIEL_CHECK_FALSE(entry == repetitions_.end());
  return entry->second >= kNumRepetitionsToDraw;
}

absl::optional<std::vector<double>> RbcState::MaybeFinalReturns() const {
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

RbcGame::RbcGame(const GameParameters& params)
    : Game(kGameType, params),
      board_size_(ParameterValue<int>("board_size")),
      sense_size_(ParameterValue<int>("sense_size")),
      fen_(ParameterValue<std::string>("fen", chess::DefaultFen(board_size_))) {
  default_observer_ = std::make_shared<RbcObserver>(kDefaultObsType);
}

std::shared_ptr<Observer> RbcGame::MakeObserver(
    absl::optional<IIGObservationType> iig_obs_type,
    const GameParameters& params) const {
  if (!params.empty()) SpielFatalError("Observation params not supported");
  return std::make_shared<RbcObserver>(iig_obs_type.value_or(kDefaultObsType));
}

}  // namespace rbc
}  // namespace open_spiel
