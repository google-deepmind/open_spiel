// Copyright 2022 DeepMind Technologies Limited
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

#include "open_spiel/games/2048.h"

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/utils/tensor_view.h"

namespace open_spiel {
namespace two_zero_four_eight {
namespace {

// Number of rows with pieces for each player
constexpr int kNumRowsWithPieces = 3;
// Types of moves: normal & capture
constexpr int kNumMoveType = 2;
// Number of unique directions each piece can take.
constexpr int kNumDirections = 4;

constexpr int kMoveUp = 0;
constexpr int kMoveRight = 1;
constexpr int kMoveDown = 2;
constexpr int kMoveLeft = 3;
// Index 0: Direction is diagonally up-left.
// Index 1: Direction is diagonally up-right.
// Index 2: Direction is diagonally down-right.
// Index 3: Direction is diagonally down-left.
constexpr std::array<int, kNumDirections> kDirRowOffsets = {{-1, -1, 1, 1}};
constexpr std::array<int, kNumDirections> kDirColumnOffsets = {{-1, 1, 1, -1}};

// Facts about the game.
const GameType kGameType{/*short_name=*/"2048",
                         /*long_name=*/"2048",
                         GameType::Dynamics::kSequential,
                         GameType::ChanceMode::kExplicitStochastic,
                         GameType::Information::kPerfectInformation,
                         GameType::Utility::kGeneralSum,
                         GameType::RewardModel::kTerminal,
                         /*max_num_players=*/1,
                         /*min_num_players=*/1,
                         /*provides_information_state_string=*/false,
                         /*provides_information_state_tensor=*/false,
                         /*provides_observation_string=*/true,
                         /*provides_observation_tensor=*/true,
                         /*parameter_specification=*/
                         {{"rows", GameParameter(kDefaultRows)},
                          {"columns", GameParameter(kDefaultColumns)}}};

std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::shared_ptr<const Game>(new TwoZeroFourEightGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);

int StateToPlayer(CellState state) {
  switch (state) {
    case CellState::kWhite:
      return 0;
    case CellState::kBlack:
      return 1;
    default:
      SpielFatalError("No player id for this cell state");
  }
}

CellState CrownState(CellState state) {
  switch (state) {
    case CellState::kWhite:
      return CellState::kWhiteKing;
    case CellState::kBlack:
      return CellState::kBlackKing;
    default:
      SpielFatalError("Invalid state");
  }
}

PieceType StateToPiece(CellState state) {
  switch (state) {
    case CellState::kWhite:
    case CellState::kBlack:
      return PieceType::kMan;
    case CellState::kWhiteKing:
    case CellState::kBlackKing:
      return PieceType::kKing;
    default:
      SpielFatalError("Invalid state");
  }
}

CellState PlayerToState(Player player) {
  switch (player) {
    case 0:
      return CellState::kWhite;
    case 1:
      return CellState::kBlack;
    default:
      SpielFatalError(absl::StrCat("Invalid player id ", player));
  }
}

std::string StateToString(CellState state) {
  switch (state) {
    case CellState::kEmpty:
      return ".";
    case CellState::kWhite:
      return "o";
    case CellState::kBlack:
      return "+";
    case CellState::kWhiteKing:
      return "8";
    case CellState::kBlackKing:
      return "*";
    default:
      SpielFatalError("Unknown state.");
  }
}

CellState StringToState(char ch) {
  switch (ch) {
    case '.':
      return CellState::kEmpty;
    case 'o':
      return CellState::kWhite;
    case '+':
      return CellState::kBlack;
    case '8':
      return CellState::kWhiteKing;
    case '*':
      return CellState::kBlackKing;
    default:
      std::string error_string = "Unknown state: ";
      error_string.push_back(ch);
      SpielFatalError(error_string);
  }
}

CellState OpponentState(CellState state) {
  return PlayerToState(1 - StateToPlayer(state));
}

std::string RowLabel(int rows, int row) {
  int row_number = rows - row;
  std::string label = std::to_string(row_number);
  return label;
}

std::string ColumnLabel(int column) {
  std::string label = "";
  label += static_cast<char>('a' + column);
  return label;
}
}  // namespace

std::ostream& operator<<(std::ostream& stream, const CellState& state) {
  switch (state) {
    case CellState::kWhite:
      return stream << "White";
    case CellState::kBlack:
      return stream << "Black";
    case CellState::kWhiteKing:
      return stream << "WhiteKing";
    case CellState::kBlackKing:
      return stream << "BlackKing";
    case CellState::kEmpty:
      return stream << "Empty";
    default:
      SpielFatalError("Unknown cell state");
  }
}

TwoZeroFourEightState::TwoZeroFourEightState(std::shared_ptr<const Game> game, int rows,
                             int columns)
    : State(game), rows_(rows), columns_(columns) {
  SPIEL_CHECK_GE(rows_, 1);
  SPIEL_CHECK_GE(columns_, 1);
  SPIEL_CHECK_LE(rows_, 99);     // Only supports 1 and 2 digit row numbers.
  SPIEL_CHECK_LE(columns_, 26);  // Only 26 letters to represent columns.

  board_ = std::vector<int>(rows_ * columns_, 0);
  turn_history_info_ = {};
}

CellState TwoZeroFourEightState::CrownStateIfLastRowReached(int row, CellState state) {
  if (row == 0 && state == CellState::kWhite) {
    return CellState::kWhiteKing;
  }
  if (row == rows_ - 1 && state == CellState::kBlack) {
    return CellState::kBlackKing;
  }
  return state;
}

void TwoZeroFourEightState::SetCustomBoard(const std::string board_string) {
  
}

ChanceAction TwoZeroFourEightState::SpielActionToChanceAction(Action action) const {
  std::vector<int> values = UnrankActionMixedBase(
      action, {rows_, columns_, kNumChanceTiles});
  return ChanceAction(values[0], values[1], values[2]);
}

Action TwoZeroFourEightState::ChanceActionToSpielAction(ChanceAction move) const {
  std::vector<int> action_bases = {rows_, columns_, kNumChanceTiles};
  return RankActionMixedBase(
      action_bases, {move.row, move.column, move.is_four});
}

CheckersAction TwoZeroFourEightState::SpielActionToCheckersAction(Action action) const {
  std::vector<int> values = UnrankActionMixedBase(
      action, {rows_, columns_, kNumDirections, kNumMoveType});
  return CheckersAction(values[0], values[1], values[2], values[3]);
}

Action TwoZeroFourEightState::CheckersActionToSpielAction(CheckersAction move) const {
  std::vector<int> action_bases = {rows_, columns_, kNumDirections,
                                   kNumMoveType};
  return RankActionMixedBase(
      action_bases, {move.row, move.column, move.direction, move.move_type});
}

void TwoZeroFourEightState::DoApplyAction(Action action) {
  if (IsChanceNode()) {
    ChanceAction chance_action = SpielActionToChanceAction(action);
    SetBoard(chance_action.row, chance_action.column, 
        chance_action.is_four ? 4 : 2);
    current_player_ = 0;
    return;
  }
  current_player_ = kChancePlayerId;
}

std::string TwoZeroFourEightState::ActionToString(Player player,
                                          Action action_id) const {
  if (IsChanceNode()) {
    ChanceAction chance_action = SpielActionToChanceAction(action_id);
    return absl::StrCat(std::to_string(chance_action.is_four ? 4 : 2), 
        " added to row ", std::to_string(chance_action.row + 1),
        ", column ", std::to_string(chance_action.column + 1));
  }
  switch (action_id) {
    case kMoveUp:
      return "Up";
      break;
    case kMoveRight:
      return "Right";
      break;
    case kMoveDown:
      return "Down";
      break;
    case kMoveLeft:
      return "Left";
      break;
    default:
      return "Invalid action";
      break;
  }  
}

int TwoZeroFourEightState::AvailableCellCount() const {
  int count = 0;
  for (int r = 0; r < rows_; r++) {
    for (int c = 0; c < columns_; c++) {
      if (BoardAt(r, c) == 0) {
        count++;
      }
    }
  }
  return count;
}

ActionsAndProbs TwoZeroFourEightState::ChanceOutcomes() const {
  ActionsAndProbs action_and_probs;
  int count = AvailableCellCount();
  action_and_probs.reserve(count * 2);
  for (int r = 0; r < rows_; r++) {
    for (int c = 0; c < columns_; c++) {
      if (BoardAt(r, c) == 0) {
        action_and_probs.emplace_back(ChanceActionToSpielAction(
            ChanceAction(r, c, false)), .9 / count);
        action_and_probs.emplace_back(ChanceActionToSpielAction(
            ChanceAction(r, c, true)), .1 / count);
      }      
    }
  }  
  return action_and_probs;
}

std::vector<Action> TwoZeroFourEightState::LegalActions() const {
  if (IsChanceNode()) return LegalChanceOutcomes();
  return {kMoveUp, kMoveRight, kMoveDown, kMoveLeft};
}

bool TwoZeroFourEightState::InBounds(int row, int column) const {
  return (row >= 0 && row < rows_ && column >= 0 && column < columns_);
}

std::string TwoZeroFourEightState::ToString() const {
  std::string str;
  for (int r = 0; r < rows_; ++r) {
    for (int c = 0; c < columns_; ++c) {
      absl::StrAppend(&str, std::to_string(BoardAt(r, c)));
    }
    absl::StrAppend(&str, "\n");
  }
  return str;  
}

int TwoZeroFourEightState::ObservationPlane(CellState state, Player player) const {
  int state_value;
  switch (state) {
    case CellState::kWhite:
      state_value = 0;
      break;
    case CellState::kWhiteKing:
      state_value = 1;
      break;
    case CellState::kBlackKing:
      state_value = 2;
      break;
    case CellState::kBlack:
      state_value = 3;
      break;
    case CellState::kEmpty:
    default:
      return 4;
  }
  if (player == Player{0}) {
    return state_value;
  } else {
    return 3 - state_value;
  }
}

bool TwoZeroFourEightState::IsTerminal() const {
  return AvailableCellCount() == 0;
}

std::vector<double> TwoZeroFourEightState::Returns() const {
  if (outcome_ == kInvalidPlayer ||
      moves_without_capture_ >= kMaxMovesWithoutCapture) {
    return {0., 0.};
  } else if (outcome_ == Player{0}) {
    return {1.0, -1.0};
  } else if (outcome_ == Player{1}) {
    return {-1.0, 1.0};
  }
  return {0., 0.};
}

std::string TwoZeroFourEightState::InformationStateString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  return HistoryString();
}

std::string TwoZeroFourEightState::ObservationString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  return ToString();
}

void TwoZeroFourEightState::ObservationTensor(Player player,
                                      absl::Span<float> values) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);  
}

CellState GetPieceStateFromTurnHistory(Player player, int piece_type) {
  return piece_type == PieceType::kMan ? PlayerToState(player)
                                       : CrownState(PlayerToState(player));
}

void TwoZeroFourEightState::UndoAction(Player player, Action action) {  
  turn_history_info_.pop_back();
  history_.pop_back();
}

TwoZeroFourEightGame::TwoZeroFourEightGame(const GameParameters& params)
    : Game(kGameType, params),
      rows_(ParameterValue<int>("rows")),
      columns_(ParameterValue<int>("columns")) {}

int TwoZeroFourEightGame::NumDistinctActions() const {
  return rows_ * columns_ * kNumDirections * kNumMoveType;
}

}  // namespace two_zero_four_eight
}  // namespace open_spiel
