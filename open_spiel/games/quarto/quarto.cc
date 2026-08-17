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

#include "open_spiel/games/quarto/quarto.h"

#include <algorithm>
#include <array>
#include <iomanip>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/abseil-cpp/absl/strings/str_format.h"
#include "open_spiel/abseil-cpp/absl/types/span.h"
#include "open_spiel/observer.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/utils/tensor_view.h"

namespace open_spiel {
namespace quarto {
namespace {

const GameType kGameType{/*short_name=*/"quarto",
                         /*long_name=*/"Quarto",
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
                         /*parameter_specification=*/{}};

std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::make_shared<const QuartoGame>(params);
}

REGISTER_SPIEL_GAME(kGameType, Factory);

RegisterSingleTensorObserver single_tensor(kGameType.short_name);

std::string PlayerToString(Player player) {
  switch (player) {
    case 0:
      return "player_0";
    case 1:
      return "player_1";
    default:
      SpielFatalError(absl::StrCat("Invalid player: ", player));
  }
}

Player StringToPlayer(const std::string& player) {
  if (player == "player_0") return 0;
  if (player == "player_1") return 1;
  SpielFatalError(absl::StrCat("Invalid player string: ", player));
}

std::string PhaseToString(Phase phase) {
  switch (phase) {
    case Phase::kSelect:
      return "select";
    case Phase::kPlace:
      return "place";
    case Phase::kTerminal:
      return "terminal";
  }
  SpielFatalError("Invalid Quarto phase.");
}

Phase StringToPhase(const std::string& phase) {
  if (phase == "select") return Phase::kSelect;
  if (phase == "place") return Phase::kPlace;
  if (phase == "terminal") return Phase::kTerminal;
  SpielFatalError(absl::StrCat("Invalid phase string: ", phase));
}

std::string OutcomeToString(Phase phase, Player outcome) {
  if (phase != Phase::kTerminal) return "";
  return outcome == kInvalidPlayer ? "draw" : PlayerToString(outcome);
}

std::string PieceToString(int piece) {
  if (piece == kNoPiece) return "-";
  return absl::StrFormat("%X", piece);
}

}  // namespace

bool LineHasQuarto(const std::array<int, 4>& pieces) {
  if (std::any_of(pieces.begin(), pieces.end(),
                  [](int piece) { return piece == kEmptyCell; })) {
    return false;
  }
  int common_ones = pieces[0] & pieces[1] & pieces[2] & pieces[3];
  int common_zeros = (~(pieces[0] | pieces[1] | pieces[2] | pieces[3])) & 0xf;
  return common_ones != 0 || common_zeros != 0;
}

bool BoardHasQuarto(const std::array<int, kNumCells>& board) {
  for (int row = 0; row < kNumRows; ++row) {
    std::array<int, 4> line;
    for (int col = 0; col < kNumCols; ++col) {
      line[col] = board[row * kNumCols + col];
    }
    if (LineHasQuarto(line)) return true;
  }
  for (int col = 0; col < kNumCols; ++col) {
    std::array<int, 4> line;
    for (int row = 0; row < kNumRows; ++row) {
      line[row] = board[row * kNumCols + col];
    }
    if (LineHasQuarto(line)) return true;
  }
  return LineHasQuarto({board[0], board[5], board[10], board[15]}) ||
         LineHasQuarto({board[3], board[6], board[9], board[12]});
}

QuartoState::QuartoState(std::shared_ptr<const Game> game) : State(game) {
  board_.fill(kEmptyCell);
}

void QuartoState::DoApplyAction(Action action) {
  SPIEL_CHECK_GE(action, 0);
  SPIEL_CHECK_LT(action, kNumPieces);
  if (phase_ == Phase::kSelect) {
    SPIEL_CHECK_FALSE(IsPieceUsed(action));
    selected_piece_ = action;
    used_pieces_ |= uint16_t{1} << action;
    current_player_ = 1 - current_player_;
    phase_ = Phase::kPlace;
    return;
  }

  SPIEL_CHECK_EQ(phase_, Phase::kPlace);
  SPIEL_CHECK_EQ(board_[action], kEmptyCell);
  board_[action] = selected_piece_;
  selected_piece_ = kNoPiece;
  ++num_placements_;

  if (HasQuarto()) {
    outcome_ = current_player_;
    phase_ = Phase::kTerminal;
  } else if (num_placements_ == kNumCells) {
    outcome_ = kInvalidPlayer;
    phase_ = Phase::kTerminal;
  } else {
    phase_ = Phase::kSelect;
  }
}

std::vector<Action> QuartoState::LegalActions() const {
  if (IsTerminal()) return {};
  std::vector<Action> actions;
  actions.reserve(kNumPieces - num_placements_);
  if (phase_ == Phase::kSelect) {
    for (int piece = 0; piece < kNumPieces; ++piece) {
      if (!IsPieceUsed(piece)) actions.push_back(piece);
    }
  } else {
    for (int cell = 0; cell < kNumCells; ++cell) {
      if (board_[cell] == kEmptyCell) actions.push_back(cell);
    }
  }
  return actions;
}

std::string QuartoState::ActionToString(Player player, Action action_id) const {
  if (phase_ == Phase::kSelect) {
    return absl::StrCat("select ", PieceToString(action_id));
  }
  if (phase_ == Phase::kPlace) {
    return absl::StrCat("place ", PieceToString(selected_piece_), " at (",
                        action_id / kNumCols, ",", action_id % kNumCols, ")");
  }
  return game_->ActionToString(player, action_id);
}

std::string QuartoState::ToString() const {
  std::ostringstream stream;
  if (IsTerminal()) {
    stream << (outcome_ == kInvalidPlayer
                   ? "Draw"
                   : absl::StrCat(PlayerToString(outcome_), " wins"));
  } else {
    stream << PlayerToString(current_player_) << " to "
           << PhaseToString(phase_);
  }
  stream << "\nSelected piece: " << PieceToString(selected_piece_) << "\n";
  for (int row = 0; row < kNumRows; ++row) {
    for (int col = 0; col < kNumCols; ++col) {
      if (col > 0) stream << " ";
      int piece = BoardAt(row, col);
      stream << (piece == kEmptyCell ? "." : PieceToString(piece));
    }
    if (row + 1 < kNumRows) stream << "\n";
  }
  return stream.str();
}

bool QuartoState::HasQuarto() const { return BoardHasQuarto(board_); }

std::vector<double> QuartoState::Returns() const {
  if (!IsTerminal() || outcome_ == kInvalidPlayer) return {0.0, 0.0};
  return outcome_ == 0 ? std::vector<double>{1.0, -1.0}
                       : std::vector<double>{-1.0, 1.0};
}

std::string QuartoState::InformationStateString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  return HistoryString();
}

std::string QuartoState::ObservationString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  return ToString();
}

void QuartoState::ObservationTensor(Player player,
                                    absl::Span<float> values) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  TensorView<2> view(values, {kNumPieces, kNumCells + 1}, true);
  for (int cell = 0; cell < kNumCells; ++cell) {
    if (board_[cell] != kEmptyCell) {
      view[{board_[cell], cell}] = 1.0;
    }
  }
  if (selected_piece_ != kNoPiece) {
    view[{selected_piece_, kNumCells}] = 1.0;
  }
}

std::unique_ptr<State> QuartoState::Clone() const {
  return std::make_unique<QuartoState>(*this);
}

void QuartoState::UndoAction(Player player, Action action) {
  if (phase_ == Phase::kPlace) {
    SPIEL_CHECK_EQ(selected_piece_, action);
    used_pieces_ &= ~(uint16_t{1} << selected_piece_);
    selected_piece_ = kNoPiece;
    current_player_ = player;
    phase_ = Phase::kSelect;
  } else {
    SPIEL_CHECK_NE(board_[action], kEmptyCell);
    selected_piece_ = board_[action];
    board_[action] = kEmptyCell;
    --num_placements_;
    current_player_ = player;
    outcome_ = kInvalidPlayer;
    phase_ = Phase::kPlace;
  }
  history_.pop_back();
  --move_number_;
}

std::unique_ptr<StateStruct> QuartoState::ToStruct() const {
  auto state_struct = std::make_unique<QuartoStateStruct>();
  state_struct->current_player = PlayerToString(current_player_);
  state_struct->phase = PhaseToString(phase_);
  state_struct->board.assign(board_.begin(), board_.end());
  state_struct->selected_piece = selected_piece_;
  state_struct->outcome = OutcomeToString(phase_, outcome_);
  return state_struct;
}

std::unique_ptr<ObservationStruct> QuartoState::ToObservationStruct(
    Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  return std::make_unique<QuartoObservationStruct>(ToJson());
}

std::unique_ptr<ActionStruct> QuartoState::ActionToStruct(
    Player player, Action action_id) const {
  SPIEL_CHECK_EQ(player, CurrentPlayer());
  auto action_struct = std::make_unique<QuartoActionStruct>();
  if (phase_ == Phase::kSelect) {
    action_struct->action_type = "select";
    action_struct->piece = action_id;
    action_struct->row = -1;
    action_struct->col = -1;
  } else {
    SPIEL_CHECK_EQ(phase_, Phase::kPlace);
    action_struct->action_type = "place";
    action_struct->piece = selected_piece_;
    action_struct->row = action_id / kNumCols;
    action_struct->col = action_id % kNumCols;
  }
  return action_struct;
}

std::vector<Action> QuartoState::StructToActions(
    const ActionStruct& action_struct) const {
  const auto* action = SafeActionCast<QuartoActionStruct>(action_struct);
  if (phase_ == Phase::kSelect) {
    SPIEL_CHECK_EQ(action->action_type, "select");
    SPIEL_CHECK_GE(action->piece, 0);
    SPIEL_CHECK_LT(action->piece, kNumPieces);
    return {action->piece};
  }
  SPIEL_CHECK_EQ(phase_, Phase::kPlace);
  SPIEL_CHECK_EQ(action->action_type, "place");
  SPIEL_CHECK_EQ(action->piece, selected_piece_);
  SPIEL_CHECK_GE(action->row, 0);
  SPIEL_CHECK_LT(action->row, kNumRows);
  SPIEL_CHECK_GE(action->col, 0);
  SPIEL_CHECK_LT(action->col, kNumCols);
  return {action->row * kNumCols + action->col};
}

QuartoState::QuartoState(std::shared_ptr<const Game> game,
                         const QuartoStateStruct& state_struct)
    : State(game) {
  if (state_struct.board.size() != kNumCells) {
    SpielFatalError(absl::StrFormat("Invalid board size: expected %d, got %d",
                                    kNumCells, state_struct.board.size()));
  }

  board_.fill(kEmptyCell);
  for (int cell = 0; cell < kNumCells; ++cell) {
    int piece = state_struct.board[cell];
    if (piece < kEmptyCell || piece >= kNumPieces) {
      SpielFatalError(absl::StrCat("Invalid piece: ", piece));
    }
    if (piece != kEmptyCell) {
      if (IsPieceUsed(piece)) {
        SpielFatalError(absl::StrCat("Duplicate piece: ", piece));
      }
      board_[cell] = piece;
      used_pieces_ |= uint16_t{1} << piece;
      ++num_placements_;
    }
  }

  phase_ = StringToPhase(state_struct.phase);
  current_player_ = StringToPlayer(state_struct.current_player);
  selected_piece_ = state_struct.selected_piece;

  if (phase_ == Phase::kPlace) {
    if (selected_piece_ < 0 || selected_piece_ >= kNumPieces ||
        IsPieceUsed(selected_piece_)) {
      SpielFatalError("Place phase requires a valid unused selected piece.");
    }
    used_pieces_ |= uint16_t{1} << selected_piece_;
  } else if (selected_piece_ != kNoPiece) {
    SpielFatalError("Only the place phase can have a selected piece.");
  }

  Player expected_player = phase_ == Phase::kPlace
                               ? (num_placements_ % 2 == 0 ? 1 : 0)
                               : (num_placements_ % 2 == 0 ? 0 : 1);
  if (current_player_ != expected_player) {
    SpielFatalError(absl::StrCat("Invalid current player: expected ",
                                 PlayerToString(expected_player), ", got ",
                                 state_struct.current_player));
  }

  bool has_quarto = HasQuarto();
  if (phase_ == Phase::kTerminal) {
    if (has_quarto) {
      Player expected_outcome = num_placements_ % 2 == 0 ? 0 : 1;
      outcome_ = StringToPlayer(state_struct.outcome);
      if (outcome_ != expected_outcome) {
        SpielFatalError("Outcome does not match the player who placed last.");
      }
    } else {
      if (num_placements_ != kNumCells || state_struct.outcome != "draw") {
        SpielFatalError("A terminal state must be a win or a full-board draw.");
      }
      outcome_ = kInvalidPlayer;
    }
  } else {
    if (has_quarto || num_placements_ == kNumCells ||
        !state_struct.outcome.empty()) {
      SpielFatalError("Non-terminal Quarto state has a terminal outcome.");
    }
  }

  starting_state_str_ = ToJson();
}

QuartoGame::QuartoGame(const GameParameters& params)
    : Game(kGameType, params) {}

std::string QuartoGame::ActionToString(Player player, Action action_id) const {
  return absl::StrCat("action ", action_id);
}

}  // namespace quarto
}  // namespace open_spiel
