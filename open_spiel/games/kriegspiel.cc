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

#include "open_spiel/games/kriegspiel.h"

#include <algorithm>
#include <memory>
#include <utility>
#include <vector>
#include <open_spiel/fog/observation_history.h>

#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace kriegspiel {
namespace {

constexpr int kNumReversibleMovesToDraw = 100;
constexpr int kNumRepetitionsToDraw = 3;

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
    /*provides_information_state_string=*/true,
    /*provides_information_state_tensor=*/false,
    /*provides_observation_string=*/true,
    /*provides_observation_tensor=*/true,
    /*parameter_specification=*/{
      {"board_size", GameParameter(8)},
      {"fen", GameParameter(GameParameter::Type::kString, false)}}
};

std::shared_ptr<const Game> Factory(const GameParameters &params) {
  return std::shared_ptr<const Game>(new KriegspielGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory)

chess::ObservationTable ComputePrivateInfoTable(
    const chess::ChessBoard &board, chess::Color color) {

  const int board_size = board.BoardSize();
  chess::ObservationTable observability_table{false};

  for (int8_t y = 0; y < board_size; ++y) {
    for (int8_t x = 0; x < board_size; ++x) {
      chess::Square sq = {x, y};
      if (board.IsFriendly({x, y}, color)) {
        size_t index = chess::SquareToIndex(sq, board_size);
        observability_table[index] = true;
      }
    }
  }
  return observability_table;
}

bool ObserverHasString(IIGObservationType iig_obs_type) {
  return iig_obs_type.public_info
      && iig_obs_type.private_info == PrivateInfoType::kSinglePlayer;
}
bool ObserverHasTensor(IIGObservationType iig_obs_type) {
  return !iig_obs_type.perfect_recall;
}

bool IsLongDiagonal(const chess::Square from_sq, const chess::Square to_sq) {
  if ((to_sq.y <= 3 && to_sq.x <= 3) || (to_sq.y > 3 && to_sq.x > 3)) {
    return from_sq.y - to_sq.y == from_sq.x - to_sq.x;
  } else {
    return from_sq.y - to_sq.y != from_sq.x - to_sq.x;
  }
}

}  // namespace


class KriegspielObserver : public Observer {
 public:
  explicit KriegspielObserver(IIGObservationType iig_obs_type)
      : Observer(/*has_string=*/ObserverHasString(iig_obs_type),
                 /*has_tensor=*/ObserverHasTensor(iig_obs_type)),
        iig_obs_type_(iig_obs_type) {}

  void WriteTensor(const State &observed_state,
                   int player,
                   Allocator *allocator) const override {
    auto &state = open_spiel::down_cast<const KriegspielState &>(observed_state);
    auto &game = open_spiel::down_cast<const KriegspielGame &>(*state.GetGame());
    SPIEL_CHECK_GE(player, 0);
    SPIEL_CHECK_LT(player, game.NumPlayers());
    chess::Color color = chess::PlayerToColor(player);

    if (iig_obs_type_.perfect_recall) {
      SpielFatalError(
          "DarkChessObserver: tensor with perfect recall not implemented.");
    }

    if (iig_obs_type_.public_info) {
      WritePublicInfoTensor(state, color, allocator);
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
    auto &state = open_spiel::down_cast<const KriegspielState &>(observed_state);
    auto &game = open_spiel::down_cast<const KriegspielGame &>(*state.GetGame());
    SPIEL_CHECK_GE(player, 0);
    SPIEL_CHECK_LT(player, game.NumPlayers());

    if (iig_obs_type_.perfect_recall) {
      if (iig_obs_type_.public_info
          && iig_obs_type_.private_info == PrivateInfoType::kSinglePlayer) {
        return state.aohs_[player].ToString();
      } else {
        SpielFatalError(
            "DarkChessObserver: string with perfect recall is implemented only"
            " for the (default) info state observation type.");
      }
    }

    if (iig_obs_type_.public_info && iig_obs_type_.private_info == PrivateInfoType::kSinglePlayer) {
      // No observation before the first move
      if (state.MoveMsgHistory().empty()) {
        return std::string();
      }
      chess::Color color = chess::PlayerToColor(player);
      chess::Color to_play = state.Board().ToPlay();
      std::string msg;

      // Write public umpire messages since the last turn of the observing player.
      if (color == to_play && state.before_last_public_msg) {
        msg += state.before_last_public_msg->ToString();
      }
      if (state.last_public_msg) {
        if (!msg.empty()) msg += "\n";
        msg += state.last_public_msg->ToString();
      }

      // Write if the observing player's last move was illegal
      auto last_msg = state.MoveMsgHistory().back();
      bool illegal = last_msg.first.piece.color == to_play &&
                     last_msg.second.illegal_;
      if (illegal) {
        if (!msg.empty()) msg += "\n";
        msg += last_msg.second.ToString();
      }
      return msg;
    } else {
      SpielFatalError(
          "DarkChessObserver: string with imperfect recall is implemented only"
          " for the (default) observation type.");
    }
  }

 private:

  void WritePieces(chess::Color color,
                   chess::PieceType piece_type,
                   const chess::ChessBoard& board,
                   const chess::ObservationTable& observability_table,
                   const std::string& prefix,
                   Allocator* allocator) const {

    const std::string type_string = color == chess::Color::kEmpty
        ? "empty"
        : chess::PieceTypeToString(piece_type,
                                   /*uppercase=*/color == chess::Color::kWhite);
    const int board_size = board.BoardSize();

    auto out = allocator->Get(prefix + "_" + type_string + "_pieces",
                              {board_size, board_size});
    for (int8_t y = 0; y < board_size; ++y) {
      for (int8_t x = 0; x < board_size; ++x) {
        const chess::Square square{x, y};
        const chess::Piece& piece_on_board = board.at(square);
        const bool write_square = piece_on_board.color == color
            && piece_on_board.type == piece_type
            && observability_table[chess::SquareToIndex(square, board_size)];
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
                              {board_size, board_size});
    for (int8_t y = 0; y < board_size; ++y) {
      for (int8_t x = 0; x < board_size; ++x) {
        const chess::Square square{x, y};
        const bool write_square =
            observability_table[chess::SquareToIndex(square, board_size)];
        out.at(x, y) = write_square ? 0.0f : 1.0f;
      }
    }
  }

  void WriteUmpireMessage(const KriegspielUmpireMessage &msg,
                          const chess::ChessBoard& board,
                          const std::string &prefix,
                          Allocator *allocator) const {
    int board_size = board.BoardSize();
    WriteScalar(msg.capture_type_, 0, 2, "_capture_type", allocator);
    auto square_out = allocator->Get(prefix + "_captured_square",
                                     {board_size, board_size});
    for (int8_t y = 0; y < board_size; ++y) {
      for (int8_t x = 0; x < board_size; ++x) {
        const chess::Square square{x, y};
        square_out.at(x, y) = square == msg.square_ ? 1.0f : 0.0f;
      }
    }
    WriteScalar(msg.check_types_.first, 0, 5, "_check_one", allocator);
    WriteScalar(msg.check_types_.second, 0, 5, "_check_two", allocator);
    WriteScalar((int8_t) msg.to_move_, 0, 2, "_to_move", allocator);
    WriteScalar(msg.pawn_tries_, 0, 16, "pawn_tries", allocator);
  }

  void WriteScalar(int val, int min, int max, const std::string& field_name,
                   Allocator* allocator) const {
    SPIEL_DCHECK_LT(min, max);
    SPIEL_DCHECK_GE(val, min);
    SPIEL_DCHECK_LE(val, max);
    auto out = allocator->Get(field_name, {max - min + 1});
    out.at( val - min ) = 1;
  }

  // Adds a binary scalar plane.
  void WriteBinary(bool val, const std::string& field_name,
                   Allocator* allocator) const {
    WriteScalar(val ? 1 : 0, 0, 1, field_name, allocator);
  }

  void WritePrivateInfoTensor(const KriegspielState &state,
                              int player, const std::string &prefix,
                              Allocator *allocator) const {
    chess::Color color = chess::PlayerToColor(player);
    chess::ObservationTable private_info_table =
        ComputePrivateInfoTable(state.Board(), color);

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

    // Write if the observing player's last move was illegal
    bool illegal = false;
    if (!state.MoveMsgHistory().empty()) {
      auto last_msg = state.MoveMsgHistory().back();
      illegal = last_msg.first.piece.color == color &&
                     last_msg.second.illegal_;
    }
    WriteBinary(illegal, prefix + "_last_illegal", allocator);
  }

  void WritePublicInfoTensor(const KriegspielState &state,
                             chess::Color color,
                             Allocator *allocator) const {

    const auto entry = state.repetitions_.find(state.Board().HashValue());
    SPIEL_CHECK_FALSE(entry == state.repetitions_.end());
    int repetitions = entry->second;

    // Num repetitions for the current board.
    WriteScalar(/*val=*/repetitions, /*min=*/1, /*max=*/3, "repetitions",
                allocator);

    // Side to play.
    WriteScalar(/*val=*/ColorToPlayer(state.Board().ToPlay()),
                /*min=*/0, /*max=*/1, "side_to_play", allocator);

    // Irreversible move counter.
    auto out = allocator->Get("irreversible_move_counter", {1});
    out.at(0) = state.Board().IrreversibleMoveCounter() / 100.;

    // Write public umpire messages since the last turn of the observing player.
    chess::Color to_play = state.Board().ToPlay();
    if (to_play == color) {
      if (state.before_last_public_msg) {
        WriteUmpireMessage(*state.before_last_public_msg, state.Board(), "first", allocator);
      }
      if (state.last_public_msg) {
        WriteUmpireMessage(*state.last_public_msg, state.Board(), "second", allocator);
      }
    } else {
      if (state.last_public_msg) {
        WriteUmpireMessage(*state.last_public_msg, state.Board(), "first", allocator);
      }
      WriteUmpireMessage({}, state.Board(), "second", allocator);
    }
  }

  IIGObservationType iig_obs_type_;
};

std::string KriegspielUmpireMessage::ToString() const {
  if (illegal_) {
    return "Illegal move.";
  }

  std::string msg;
  bool put_comma = false;

  if (capture_type_ != KriegspielCaptureType::kNoCapture) {
    msg += CaptureTypeToString(capture_type_) + " at " + chess::SquareToString(square_) + " captured";
    put_comma = true;
  }
  if (check_types_.first != KriegspielCheckType::kNoCheck) {
    if (put_comma) msg += ", ";
    msg += CheckTypeToString(check_types_.first) + " check";
    put_comma = true;
  }
  if (check_types_.second != KriegspielCheckType::kNoCheck) {
    if (put_comma) msg += ", ";
    msg += CheckTypeToString(check_types_.second) + " check";
    put_comma = true;
  }
  if (put_comma) msg += ", ";

  msg += chess::ColorToString(to_move_) + "'s move";
  if (pawn_tries_ > 0) {
    msg += ", ";
    msg +=  pawn_tries_ == 1
            ?  "1 pawn try"
            : std::to_string(pawn_tries_) + " pawn try";
  }
  msg += ".";
  return msg;
}

KriegspielState::KriegspielState(std::shared_ptr<const Game> game,
                               int boardSize,
                               const std::string &fen)
    : State(game),
      start_board_(*chess::ChessBoard::BoardFromFEN(fen, boardSize, false)),
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

void KriegspielState::DoApplyAction(Action action) {
  Player current_player = CurrentPlayer();
  chess::Move move = ActionToMove(action, Board());
  KriegspielUmpireMessage msg {};
  cached_legal_actions_.reset();
  if (!Board().IsMoveLegal(move)) {
    // If the move is illegal, the player is notified about it and can play again
    msg.illegal_ = true;
    move_msg_history_.emplace_back(move, msg);
    illegal_tried_moves_.insert(move);
    return;
  }

  msg.illegal_ = false;
  chess::PieceType capture_type = Board().at(move.to).type;
  switch (capture_type) {
    case chess::PieceType::kEmpty:
      msg.capture_type_ = KriegspielCaptureType::kNoCapture;
      msg.square_ = chess::InvalidSquare();
      break;
    case chess::PieceType::kPawn:
      msg.capture_type_ = KriegspielCaptureType::kPawn;
      msg.square_ = move.to;
      break;
    default:
      msg.capture_type_ = KriegspielCaptureType::kPiece;
      msg.square_ = move.to;
  }

  Board().ApplyMove(move);
  illegal_tried_moves_.clear();
  ++repetitions_[current_board_.HashValue()];

  for (Player player = 0; player < NumPlayers(); ++player) {
    std::optional<Action> a = {};
    if (current_player == player) a = action;
    aohs_[player].Extend(a, ObservationString(player));
  }

  chess::Square king_sq = Board().find(
      chess::Piece{Board().ToPlay(), chess::PieceType::kKing});

  std::pair<KriegspielCheckType, KriegspielCheckType> check_type_pair =
    {KriegspielCheckType::kNoCheck, KriegspielCheckType::kNoCheck};

  Board().GeneratePseudoLegalMoves([&king_sq, &check_type_pair](const chess::Move &move) {
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
    else if (IsLongDiagonal(move.from, move.to))
      check_type = KriegspielCheckType::kLongDiagonal;
    else check_type = KriegspielCheckType::kShortDiagonal;

    if (check_type_pair.first != KriegspielCheckType::kNoCheck) {
      // There cannot be more than two checks at the same time
      check_type_pair.second = check_type;
      return false;
    }
    else check_type_pair.first = check_type;

    return true;
  }, chess::OppColor(Board().ToPlay()), false);
  msg.check_types_ = check_type_pair;

  int pawnTries = 0;
  Board().GenerateLegalMoves([this, &pawnTries](const chess::Move &move) {
    if (move.piece.type == chess::PieceType::kPawn
        && Board().at(move.to).type != chess::PieceType::kEmpty) {
      pawnTries++;
    }
    return true;
  });
  msg.pawn_tries_ = pawnTries;
  msg.to_move_ = Board().ToPlay();

  move_msg_history_.emplace_back(move, msg);
  before_last_public_msg = last_public_msg;
  last_public_msg = msg;

}

void KriegspielState::MaybeGenerateLegalActions() const {
  if (!cached_legal_actions_) {
    cached_legal_actions_ = std::vector<Action>();
    Board().GeneratePseudoLegalMoves([this](const chess::Move& move) -> bool {
      if (illegal_tried_moves_.find(move) == illegal_tried_moves_.end()) {
        cached_legal_actions_->push_back(MoveToAction(move, BoardSize()));
      }
      return true;
    }, Board().ToPlay(), true);
    absl::c_sort(*cached_legal_actions_);
  }
}

std::vector<Action> KriegspielState::LegalActions() const {
  MaybeGenerateLegalActions();
  if (IsTerminal()) return {};
  return *cached_legal_actions_;
}

std::string KriegspielState::ActionToString(Player player, Action action) const {
  chess::Move move = ActionToMove(action, Board());
  return move.ToLAN();
}

std::string KriegspielState::ToString() const { return Board().ToFEN(); }

std::vector<double> KriegspielState::Returns() const {
  auto maybe_final_returns = MaybeFinalReturns();
  if (maybe_final_returns) {
    return *maybe_final_returns;
  } else {
    return {0.0, 0.0};
  }
}

std::string KriegspielState::InformationStateString(Player player) const {
  const auto &game = open_spiel::down_cast<const KriegspielGame &>(*game_);
  return game.info_state_observer_->StringFrom(*this, player);
}

std::string KriegspielState::ObservationString(Player player) const {
  const auto& game = open_spiel::down_cast<const KriegspielGame&>(*game_);
  return game.default_observer_->StringFrom(*this, player);
}

void KriegspielState::ObservationTensor(Player player,
                                       absl::Span<float> values) const {
  ContiguousAllocator allocator(values);
  const auto& game = open_spiel::down_cast<const KriegspielGame&>(*game_);
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
  for (const std::pair<chess::Move, KriegspielUmpireMessage> &move_msg_pair : move_msg_history_) {
    current_board_.ApplyMove(move_msg_pair.first);
  }
  for (Player player = 0; player < NumPlayers(); ++player) {
    aohs_[player].RemoveLast();
  }
}

bool KriegspielState::IsRepetitionDraw() const {
  const auto entry = repetitions_.find(Board().HashValue());
  SPIEL_CHECK_FALSE(entry == repetitions_.end());
  return entry->second >= kNumRepetitionsToDraw;
}

absl::optional<std::vector<double>> KriegspielState::MaybeFinalReturns() const {

  const auto to_play_color = Board().ToPlay();
  const auto opp_color = chess::OppColor(to_play_color);

  const auto to_play_king =
      chess::Piece{to_play_color, chess::PieceType::kKing};
  const auto opp_king =
      chess::Piece{opp_color, chess::PieceType::kKing};

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

  return std::nullopt;
}

std::string DefaultFen(int board_size) {
  if (board_size == 8) return chess::kDefaultStandardFEN;
  else if (board_size == 4) return chess::kDefaultSmallFEN;
  else
    SpielFatalError(
        "Only board sizes 4 and 8 have their default chessboards. "
        "For other sizes, you have to pass your own FEN.");
}

KriegspielGame::KriegspielGame(const GameParameters &params)
    : Game(kGameType, params),
      board_size_(ParameterValue<int>("board_size")),
      fen_(ParameterValue<std::string>("fen", DefaultFen(board_size_))) {
  default_observer_ = std::make_shared<KriegspielObserver>(kDefaultObsType);
  info_state_observer_ = std::make_shared<KriegspielObserver>(kInfoStateObsType);
}

std::shared_ptr<Observer> KriegspielGame::MakeObserver(
    absl::optional<IIGObservationType> iig_obs_type,
    const GameParameters& params) const {
  if (!params.empty()) SpielFatalError("Observation params not supported");
  return std::make_shared<KriegspielObserver>(
      iig_obs_type.value_or(kDefaultObsType));
}

}  // namespace kriegspiel
}  // namespace open_spiel
