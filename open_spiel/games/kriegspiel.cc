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

#include "open_spiel/games/kriegspiel.h"

#include <algorithm>
#include <memory>
#include <utility>
#include <vector>

#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace kriegspiel {
namespace {

// Facts about the game.
const GameType kGameType{
    /*short_name=*/"kriegspiel",
    /*long_name=*/"Kriegspiel",
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
     {"fen", GameParameter(GameParameter::Type::kString, false)},
     {"threefold_repetition", GameParameter(true)},
     {"50_move_rule", GameParameter(true)}}};

std::shared_ptr<const Game> Factory(const GameParameters &params) {
  return std::shared_ptr<const Game>(new KriegspielGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory)

chess::ObservationTable ComputePrivateInfoTable(const chess::ChessBoard &board,
                                                chess::Color color) {
  const int board_size = board.BoardSize();
  chess::ObservationTable observability_table{false};

  for (int8_t y = 0; y < board_size; ++y) {
    for (int8_t x = 0; x < board_size; ++x) {
      chess::Square sq{x, y};
      if (board.IsFriendly({x, y}, color)) {
        size_t index = chess::SquareToIndex(sq, board_size);
        observability_table[index] = true;
      }
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

bool IsValid(chess::Square square) { return square.x >= 0 && square.y >= 0; }

}  // namespace

class KriegspielObserver : public Observer {
 public:
  explicit KriegspielObserver(IIGObservationType iig_obs_type)
      : Observer(/*has_string=*/ObserverHasString(iig_obs_type),
                 /*has_tensor=*/ObserverHasTensor(iig_obs_type)),
        iig_obs_type_(iig_obs_type) {}

  void WriteTensor(const State &observed_state, int player,
                   Allocator *allocator) const override {
    auto &state =
        open_spiel::down_cast<const KriegspielState &>(observed_state);
    auto &game =
        open_spiel::down_cast<const KriegspielGame &>(*state.GetGame());
    SPIEL_CHECK_GE(player, 0);
    SPIEL_CHECK_LT(player, game.NumPlayers());
    chess::Color color = chess::PlayerToColor(player);

    if (iig_obs_type_.perfect_recall) {
      SpielFatalError(
          "KriegspielObserver: tensor with perfect recall not implemented.");
    }

    if (iig_obs_type_.public_info) {
      WritePublicInfoTensor(state, "public", allocator);
    }
    if (iig_obs_type_.private_info == PrivateInfoType::kSinglePlayer) {
      std::string prefix = "private";
      WritePrivateInfoTensor(state, player, prefix, allocator);
    } else if (iig_obs_type_.private_info == PrivateInfoType::kAllPlayers) {
      for (int i = 0; i < chess::NumPlayers(); ++i) {
        std::string prefix = chess::ColorToString(color);
        WritePrivateInfoTensor(state, i, prefix, allocator);
      }
    }
  }

  std::string StringFrom(const State &observed_state,
                         int player) const override {
    auto &state =
        open_spiel::down_cast<const KriegspielState &>(observed_state);
    auto &game =
        open_spiel::down_cast<const KriegspielGame &>(*state.GetGame());
    SPIEL_CHECK_GE(player, 0);
    SPIEL_CHECK_LT(player, game.NumPlayers());

    if (iig_obs_type_.perfect_recall) {
      SpielFatalError(
          "KriegspielObserver: string with perfect recall is unimplemented");
    }

    if (iig_obs_type_.public_info &&
        iig_obs_type_.private_info == PrivateInfoType::kSinglePlayer) {
      // No observation before the first move
      if (state.MoveMsgHistory().empty()) {
        return std::string();
      }

      // Write last umpire message
      return state.last_umpire_msg_->ToString();
    } else {
      SpielFatalError(
          "KriegspielObserver: string with imperfect recall is implemented only"
          " for the (default) observation type.");
    }
  }

 private:
  void WritePieces(chess::Color color, chess::PieceType piece_type,
                   const chess::ChessBoard &board,
                   const chess::ObservationTable &observability_table,
                   const std::string &prefix, Allocator *allocator) const {
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
        const chess::Piece &piece_on_board = board.at(square);
        const bool write_square =
            piece_on_board.color == color &&
            piece_on_board.type == piece_type &&
            observability_table[chess::SquareToIndex(square, board_size)];
        out.at(x, y) = write_square ? 1.0f : 0.0f;
      }
    }
  }

  void WriteUnknownSquares(const chess::ChessBoard &board,
                           chess::ObservationTable &observability_table,
                           const std::string &prefix,
                           Allocator *allocator) const {
    const int board_size = board.BoardSize();
    auto out =
        allocator->Get(prefix + "_unknown_squares", {board_size, board_size});
    for (int8_t y = 0; y < board_size; ++y) {
      for (int8_t x = 0; x < board_size; ++x) {
        const chess::Square square{x, y};
        const bool write_square =
            observability_table[chess::SquareToIndex(square, board_size)];
        out.at(x, y) = write_square ? 0.0f : 1.0f;
      }
    }
  }

  void WriteMove(const chess::Move &move, const chess::ChessBoard &board,
                 const std::string &prefix, Allocator *allocator) const {
    const int board_size = board.BoardSize();
    auto from_out = allocator->Get(prefix + "_from", {board_size, board_size});
    if (IsValid(move.from)) from_out.at(move.from.x, move.from.y) = 1;
    auto to_out = allocator->Get(prefix + "_to", {board_size, board_size});
    if (IsValid(move.to)) to_out.at(move.to.x, move.to.y) = 1;
    // 5 is maximum because we can't promote to a pawn.
    WriteScalar(static_cast<int>(move.promotion_type), 0, 5,
                prefix + "_promotion", allocator);
  }

  void WriteUmpireMessage(const KriegspielUmpireMessage &msg,
                          const chess::ChessBoard &board,
                          const std::string &prefix,
                          Allocator *allocator) const {
    int board_size = board.BoardSize();
    WriteBinary(msg.illegal, prefix + "_illegal", allocator);
    WriteScalar(static_cast<int>(msg.capture_type), 0, 2,
                prefix + "_capture_type", allocator);
    auto square_out =
        allocator->Get(prefix + "_captured_square", {board_size, board_size});
    if (IsValid(msg.square)) square_out.at(msg.square.x, msg.square.y) = 1;
    WriteScalar(static_cast<int>(msg.check_types.first), 0, 5,
                prefix + "_check_one", allocator);
    WriteScalar(static_cast<int>(msg.check_types.second), 0, 5,
                prefix + "_check_two", allocator);
    WriteScalar(static_cast<int>(msg.to_move), 0, 2, prefix + "_to_move",
                allocator);
    WriteScalar(msg.pawn_tries, 0, 16, prefix + "_pawn_tries", allocator);
  }

  void WriteScalar(int val, int min, int max, const std::string &field_name,
                   Allocator *allocator) const {
    SPIEL_DCHECK_LT(min, max);
    SPIEL_DCHECK_GE(val, min);
    SPIEL_DCHECK_LE(val, max);
    auto out = allocator->Get(field_name, {max - min + 1});
    out.at(val - min) = 1;
  }

  // Adds a binary scalar.
  void WriteBinary(bool val, const std::string &field_name,
                   Allocator *allocator) const {
    WriteScalar(val ? 1 : 0, 0, 1, field_name, allocator);
  }

  void WritePrivateInfoTensor(const KriegspielState &state, int player,
                              const std::string &prefix,
                              Allocator *allocator) const {
    chess::Color color = chess::PlayerToColor(player);
    chess::ObservationTable private_info_table =
        ComputePrivateInfoTable(state.Board(), color);

    // Piece configuration.
    for (const chess::PieceType &piece_type : chess::kPieceTypes) {
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

    // Write observer's last move
    chess::Move last_move = {chess::kInvalidSquare, chess::kInvalidSquare,
                             chess::kEmptyPiece};

    for (auto move_msg = state.MoveMsgHistory().rbegin();
         move_msg != state.MoveMsgHistory().rend(); ++move_msg) {
      if (move_msg->first.piece.color == color) {
        last_move = move_msg->first;
        break;
      }
    }
    WriteMove(last_move, state.Board(), prefix + "_last_move", allocator);
  }

  void WritePublicInfoTensor(const KriegspielState &state,
                             const std::string &prefix,
                             Allocator *allocator) const {
    const auto entry = state.repetitions_.find(state.Board().HashValue());
    SPIEL_CHECK_FALSE(entry == state.repetitions_.end());
    int repetitions = entry->second;

    // Num repetitions for the current board.
    WriteScalar(/*val=*/repetitions, /*min=*/1, /*max=*/3,
                prefix + "_repetitions", allocator);

    // Side to play.
    WriteScalar(/*val=*/ColorToPlayer(state.Board().ToPlay()),
                /*min=*/0, /*max=*/1, prefix + "_side_to_play", allocator);

    // Irreversible move counter.
    auto out = allocator->Get(prefix + "_irreversible_move_counter", {1});
    out.at(0) = state.Board().IrreversibleMoveCounter() / 100.f;

    // Write last umpire message
    if (state.last_umpire_msg_) {
      WriteUmpireMessage(*state.last_umpire_msg_, state.Board(), prefix,
                         allocator);
    } else {
      WriteUmpireMessage(KriegspielUmpireMessage(), state.Board(), prefix,
                         allocator);
    }
  }

  IIGObservationType iig_obs_type_;
};

std::string CaptureTypeToString(KriegspielCaptureType capture_type) {
  if (capture_type == KriegspielCaptureType::kNoCapture) {
    return "No Piece";
  }
  if (capture_type == KriegspielCaptureType::kPawn) {
    return "Pawn";
  }
  return "Piece";
}

std::string CheckTypeToString(KriegspielCheckType check_type) {
  switch (check_type) {
    case KriegspielCheckType::kFile:
      return "File";
    case KriegspielCheckType::kRank:
      return "Rank";
    case KriegspielCheckType::kLongDiagonal:
      return "Long-diagonal";
    case KriegspielCheckType::kShortDiagonal:
      return "Short-diagonal";
    case KriegspielCheckType::kKnight:
      return "Knight";
    default:
      SpielFatalError("kNoCheck does not have a string representation");
  }
}

std::pair<KriegspielCheckType, KriegspielCheckType> GetCheckType(
    const chess::ChessBoard &board) {
  chess::Square king_sq =
      board.find(chess::Piece{board.ToPlay(), chess::PieceType::kKing});

  std::pair<KriegspielCheckType, KriegspielCheckType> check_type_pair = {
      KriegspielCheckType::kNoCheck, KriegspielCheckType::kNoCheck};

  board.GeneratePseudoLegalMoves(
      [&king_sq, &check_type_pair, &board](const chess::Move &move) {
        if (move.to != king_sq) {
          return true;
        }
        KriegspielCheckType check_type;
        if (move.piece.type == chess::PieceType::kKnight)
          check_type = KriegspielCheckType::kKnight;
        else if (move.from.x == move.to.x)
          check_type = KriegspielCheckType::kFile;
        else if (move.from.y == move.to.y)
          check_type = KriegspielCheckType::kRank;
        else if (chess::IsLongDiagonal(move.from, move.to, board.BoardSize()))
          check_type = KriegspielCheckType::kLongDiagonal;
        else
          check_type = KriegspielCheckType::kShortDiagonal;

        if (check_type_pair.first != KriegspielCheckType::kNoCheck) {
          // There cannot be more than two checks at the same time
          check_type_pair.second = check_type;
          return false;
        } else {
          check_type_pair.first = check_type;
        }
        return true;
      },
      board.ToPlay(), chess::PseudoLegalMoveSettings::kAcknowledgeEnemyPieces);

  return check_type_pair;
}

std::string KriegspielUmpireMessage::ToString() const {
  if (illegal) {
    return "Illegal move.";
  }

  std::string msg;
  bool put_comma = false;

  if (capture_type != KriegspielCaptureType::kNoCapture) {
    msg += CaptureTypeToString(capture_type) + " at " +
           chess::SquareToString(square) + " captured";
    put_comma = true;
  }
  if (check_types.first != KriegspielCheckType::kNoCheck) {
    if (put_comma) msg += ", ";
    msg += CheckTypeToString(check_types.first) + " check";
    put_comma = true;
  }
  if (check_types.second != KriegspielCheckType::kNoCheck) {
    if (put_comma) msg += ", ";
    msg += CheckTypeToString(check_types.second) + " check";
    put_comma = true;
  }
  if (put_comma) msg += ", ";

  msg += chess::ColorToString(to_move) + "'s move";
  if (pawn_tries > 0) {
    msg += ", ";
    msg += pawn_tries == 1 ? "1 pawn try"
                           : std::to_string(pawn_tries) + " pawn tries";
  }
  msg += ".";
  return msg;
}

KriegspielUmpireMessage GetUmpireMessage(const chess::ChessBoard &chess_board,
                                         const chess::Move &move) {
  KriegspielUmpireMessage msg{};
  if (!chess_board.IsMoveLegal(move)) {
    // If the move is illegal, the player is notified about it and can play
    // again
    msg.illegal = true;
    msg.to_move = chess_board.ToPlay();
    return msg;
  }
  msg.illegal = false;

  chess::PieceType capture_type = chess_board.at(move.to).type;
  switch (capture_type) {
    case chess::PieceType::kEmpty:
      msg.capture_type = KriegspielCaptureType::kNoCapture;
      msg.square = chess::kInvalidSquare;
      break;
    case chess::PieceType::kPawn:
      msg.capture_type = KriegspielCaptureType::kPawn;
      msg.square = move.to;
      break;
    default:
      msg.capture_type = KriegspielCaptureType::kPiece;
      msg.square = move.to;
  }

  // todo optimze when undo is optimized
  chess::ChessBoard board_copy = chess_board;
  board_copy.ApplyMove(move);

  msg.check_types = GetCheckType(board_copy);

  int pawnTries = 0;
  board_copy.GenerateLegalPawnCaptures(
      [&pawnTries](const chess::Move &move) {
        pawnTries++;
        return true;
      },
      board_copy.ToPlay());
  msg.pawn_tries = pawnTries;
  msg.to_move = board_copy.ToPlay();

  return msg;
}

bool GeneratesUmpireMessage(const chess::ChessBoard &chess_board,
                            const chess::Move &move,
                            const KriegspielUmpireMessage &orig_msg) {
  if (!chess_board.IsMoveLegal(move)) {
    // If the move is illegal, the player is notified about it and can play
    // again
    return orig_msg.illegal;
  }

  chess::PieceType capture_type = chess_board.at(move.to).type;
  switch (capture_type) {
    case chess::PieceType::kEmpty:
      if (orig_msg.capture_type != KriegspielCaptureType::kNoCapture) {
        return false;
      }
      break;
    case chess::PieceType::kPawn:
      if (orig_msg.capture_type != KriegspielCaptureType::kPawn) {
        return false;
      }
      break;
    default:
      if (orig_msg.capture_type != KriegspielCaptureType::kPiece) {
        return false;
      }
  }

  // todo optimize when undo is optimized
  chess::ChessBoard board_copy = chess_board;
  board_copy.ApplyMove(move);

  if (orig_msg.check_types != GetCheckType(board_copy)) {
    return false;
  }

  int pawnTries = 0;
  board_copy.GenerateLegalPawnCaptures(
      [&pawnTries](const chess::Move &move) {
        pawnTries++;
        return true;
      },
      board_copy.ToPlay());
  if (orig_msg.pawn_tries != pawnTries) {
    return false;
  }
  if (orig_msg.to_move != board_copy.ToPlay()) {
    return false;
  }

  return true;
}

KriegspielState::KriegspielState(std::shared_ptr<const Game> game,
                                 int board_size, const std::string &fen,
                                 bool threefold_repetition, bool rule_50_move)
    : State(game),
      start_board_(*chess::ChessBoard::BoardFromFEN(fen, board_size, false)),
      current_board_(start_board_),
      threefold_repetition_(threefold_repetition),
      rule_50_move_(rule_50_move) {
  SPIEL_CHECK_TRUE(&current_board_);
  repetitions_[current_board_.HashValue()] = 1;
}

void KriegspielState::DoApplyAction(Action action) {
  cached_legal_actions_.reset();

  chess::Move move = ActionToMove(action, Board());

  KriegspielUmpireMessage msg = GetUmpireMessage(Board(), move);

  move_msg_history_.emplace_back(move, msg);
  last_umpire_msg_ = msg;

  if (msg.illegal) {
    // If the move is illegal, the player is notified about it and can play
    // again
    illegal_tried_moves_.emplace_back(move);
    cached_legal_actions_.reset();
    return;
  }

  Board().ApplyMove(move);
  illegal_tried_moves_.clear();
  ++repetitions_[current_board_.HashValue()];
}

void KriegspielState::MaybeGenerateLegalActions() const {
  if (!cached_legal_actions_) {
    cached_legal_actions_ = std::vector<Action>();
    Board().GeneratePseudoLegalMoves(
        [this](const chess::Move &move) -> bool {
          bool is_illegal_tried = false;
          for (const chess::Move &illegal_tried_move : illegal_tried_moves_) {
            if (illegal_tried_move == move) {
              is_illegal_tried = true;
              break;
            }
          }
          if (!is_illegal_tried) {
            cached_legal_actions_->push_back(MoveToAction(move, BoardSize()));
          }
          return true;
        },
        Board().ToPlay(), chess::PseudoLegalMoveSettings::kBreachEnemyPieces);
    absl::c_sort(*cached_legal_actions_);
  }
}

std::vector<Action> KriegspielState::LegalActions() const {
  MaybeGenerateLegalActions();
  if (IsTerminal()) return {};
  return *cached_legal_actions_;
}

std::string KriegspielState::ActionToString(Player player,
                                            Action action) const {
  chess::Move move = ActionToMove(action, Board());
  return move.ToLAN();
}

std::string KriegspielState::ToString() const { return Board().ToFEN(); }

std::vector<double> KriegspielState::Returns() const {
  return MaybeFinalReturns().value_or(std::vector<double>{0., 0.});
}

std::string KriegspielState::ObservationString(Player player) const {
  const auto &game = open_spiel::down_cast<const KriegspielGame &>(*game_);
  return game.default_observer_->StringFrom(*this, player);
}

void KriegspielState::ObservationTensor(Player player,
                                        absl::Span<float> values) const {
  ContiguousAllocator allocator(values);
  const auto &game = open_spiel::down_cast<const KriegspielGame &>(*game_);
  game.default_observer_->WriteTensor(*this, player, &allocator);
}

std::unique_ptr<State> KriegspielState::Clone() const {
  return std::make_unique<KriegspielState>(*this);
}

void KriegspielState::UndoAction(Player player, Action action) {
  // TODO: Make this fast by storing undo info in another stack.
  SPIEL_CHECK_GE(move_msg_history_.size(), 1);
  --repetitions_[current_board_.HashValue()];
  move_msg_history_.pop_back();
  history_.pop_back();
  current_board_ = start_board_;
  for (const std::pair<chess::Move, KriegspielUmpireMessage> &move_msg_pair :
       move_msg_history_) {
    current_board_.ApplyMove(move_msg_pair.first);
  }
}

bool KriegspielState::IsThreefoldRepetitionDraw() const {
  const auto entry = repetitions_.find(Board().HashValue());
  SPIEL_CHECK_FALSE(entry == repetitions_.end());
  return entry->second >= 3;
}

absl::optional<std::vector<double>> KriegspielState::MaybeFinalReturns() const {
  if (!Board().HasSufficientMaterial()) {
    return std::vector<double>{kDrawUtility, kDrawUtility};
  }

  if (threefold_repetition_ && IsThreefoldRepetitionDraw()) {
    return std::vector<double>{kDrawUtility, kDrawUtility};
  }

  // Compute and cache the legal actions.
  MaybeGenerateLegalActions();
  SPIEL_CHECK_TRUE(cached_legal_actions_);
  const bool have_legal_moves = !cached_legal_actions_->empty();

  // If we don't have legal moves we are stalemated or mated
  if (!have_legal_moves) {
    if (!Board().InCheck()) {
      return std::vector<double>{kDrawUtility, kDrawUtility};
    } else {
      std::vector<double> returns(NumPlayers());
      auto next_to_play = ColorToPlayer(Board().ToPlay());
      returns[next_to_play] = kLossUtility;
      returns[chess::OtherPlayer(next_to_play)] = kWinUtility;
      return returns;
    }
  }

  if (rule_50_move_ && Board().IrreversibleMoveCounter() >= 50) {
    return std::vector<double>{kDrawUtility, kDrawUtility};
  }

  return absl::nullopt;
}

KriegspielGame::KriegspielGame(const GameParameters &params)
    : Game(kGameType, params),
      board_size_(ParameterValue<int>("board_size")),
      fen_(ParameterValue<std::string>("fen", chess::DefaultFen(board_size_))),
      threefold_repetition_(ParameterValue<bool>("threefold_repetition")),
      rule_50_move_(ParameterValue<bool>("50_move_rule")) {
  default_observer_ = std::make_shared<KriegspielObserver>(kDefaultObsType);
}

std::vector<int> KriegspielGame::ObservationTensorShape() const {
  if (observation_tensor_shape_.empty())
    observation_tensor_shape_ =
        ObserverTensorShape(*NewInitialState(), *default_observer_);
  return observation_tensor_shape_;
}

std::shared_ptr<Observer> KriegspielGame::MakeObserver(
    absl::optional<IIGObservationType> iig_obs_type,
    const GameParameters &params) const {
  if (!params.empty()) SpielFatalError("Observation params not supported");
  return std::make_shared<KriegspielObserver>(
      iig_obs_type.value_or(kDefaultObsType));
}

}  // namespace kriegspiel
}  // namespace open_spiel
