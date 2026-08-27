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

#include "open_spiel/games/catch/catch.h"

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/abseil-cpp/absl/strings/str_format.h"
#include "open_spiel/abseil-cpp/absl/types/span.h"
#include "open_spiel/json/include/nlohmann/json.hpp"  // IWYU pragma: keep
#include "open_spiel/game_parameters.h"
#include "open_spiel/observer.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_globals.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/utils/tensor_view.h"

namespace open_spiel {
namespace catch_ {
namespace {

// Facts about the game.
const GameType kGameType{/*short_name=*/"catch",
                         /*long_name=*/"Catch",
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
  return std::shared_ptr<const Game>(new CatchGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);

RegisterSingleTensorObserver single_tensor(kGameType.short_name);

// Maps CellState to a single-character string for ToString().
std::string CellStateToString(CellState state) {
  switch (state) {
    case CellState::kEmpty:
      return ".";
    case CellState::kPaddle:
      return "x";
    case CellState::kBall:
      return "o";
    default:
      SpielFatalError("Unknown state.");
      return "This will never return.";
  }
}

// Maps the action id to the human-readable direction string used in
// CatchActionStruct.
std::string ActionIdToDirection(Action action_id) {
  switch (action_id) {
    case kLeft:
      return "left";
    case kStay:
      return "stay";
    case kRight:
      return "right";
    default:
      SpielFatalError(absl::StrCat("Out of range action: ", action_id));
      return "This will never return.";
  }
}

// Inverse of ActionIdToDirection.
Action DirectionToActionId(const std::string& direction) {
  if (direction == "left") return kLeft;
  if (direction == "stay") return kStay;
  if (direction == "right") return kRight;
  SpielFatalError(absl::StrCat("Invalid direction string: '", direction,
                               "'. Expected 'left', 'stay', or 'right'."));
  return kStay;  // unreachable
}

}  // namespace

// ---------------------------------------------------------------------------
// CatchState
// ---------------------------------------------------------------------------

CatchState::CatchState(std::shared_ptr<const Game> game) : State(game) {
  const CatchGame& parent_game = static_cast<const CatchGame&>(*game);
  num_rows_ = parent_game.NumRows();
  num_columns_ = parent_game.NumColumns();
}

// Construct a CatchState from a typed CatchStateStruct.
// Validation rules mirror those of TicTacToeState::TicTacToeState(…, struct).
CatchState::CatchState(std::shared_ptr<const Game> game,
                       const CatchStateStruct& state_struct)
    : State(game) {
  const CatchGame& parent_game = static_cast<const CatchGame&>(*game);
  num_rows_ = parent_game.NumRows();
  num_columns_ = parent_game.NumColumns();

  const int br = state_struct.ball_row;
  const int bc = state_struct.ball_col;
  const int pc = state_struct.paddle_col;

  if (br == -1 && bc == -1 && pc == -1) {
    // Pre-chance node: all fields must be -1.
    initialized_ = false;
    ball_row_ = -1;
    ball_col_ = -1;
    paddle_col_ = -1;
  } else {
    // Post-chance node: validate each field individually.
    if (br < 0 || br >= num_rows_) {
      SpielFatalError(absl::StrFormat(
          "Invalid ball_row %d: must be in [0, %d).", br, num_rows_));
    }
    if (bc < 0 || bc >= num_columns_) {
      SpielFatalError(absl::StrFormat(
          "Invalid ball_col %d: must be in [0, %d).", bc, num_columns_));
    }
    if (pc < 0 || pc >= num_columns_) {
      SpielFatalError(absl::StrFormat(
          "Invalid paddle_col %d: must be in [0, %d).", pc, num_columns_));
    }

    initialized_ = true;
    ball_row_ = br;
    ball_col_ = bc;
    paddle_col_ = pc;
  }

  // Store the JSON of this starting state for serialization support
  // (mirrors the pattern used in TicTacToeState and ConnectFourState).
  starting_state_str_ = this->ToJson();
}

Player CatchState::CurrentPlayer() const {
  if (!initialized_) return kChancePlayerId;
  if (IsTerminal()) return kTerminalPlayerId;
  return 0;
}

std::vector<Action> CatchState::LegalActions() const {
  if (IsTerminal()) return {};
  if (initialized_) {
    return {kLeft, kStay, kRight};
  }
  std::vector<Action> moves;
  moves.reserve(num_columns_);
  for (int i = 0; i < num_columns_; i++) moves.push_back(i);
  return moves;
}

ActionsAndProbs CatchState::ChanceOutcomes() const {
  SPIEL_CHECK_TRUE(!initialized_);
  ActionsAndProbs action_and_probs;
  action_and_probs.reserve(num_columns_);
  for (int c = 0; c < num_columns_; c++) {
    action_and_probs.emplace_back(c, 1. / num_columns_);
  }
  return action_and_probs;
}

CellState CatchState::BoardAt(int row, int column) const {
  if (row == num_rows_ - 1 && column == paddle_col_)
    return CellState::kPaddle;
  else if (row == ball_row_ && column == ball_col_)
    return CellState::kBall;
  return CellState::kEmpty;
}

std::string CatchState::ActionToString(Player player, Action action_id) const {
  if (player == kChancePlayerId)
    return absl::StrCat("Initialized ball to ", action_id);
  SPIEL_CHECK_EQ(player, 0);
  switch (action_id) {
    case kLeft:
      return "LEFT";
    case kStay:
      return "STAY";
    case kRight:
      return "RIGHT";
    default:
      SpielFatalError("Out of range action");
      return "This will never return.";
  }
}

std::string CatchState::ToString() const {
  std::string str;
  for (int r = 0; r < num_rows_; ++r) {
    for (int c = 0; c < num_columns_; ++c) {
      absl::StrAppend(&str, CellStateToString(BoardAt(r, c)));
    }
    absl::StrAppend(&str, "\n");
  }
  return str;
}

bool CatchState::IsTerminal() const {
  return initialized_ && ball_row_ >= num_rows_ - 1;
}

std::vector<double> CatchState::Returns() const {
  if (!IsTerminal()) {
    return {0.0};
  } else if (ball_col_ == paddle_col_) {
    return {1.0};
  } else {
    return {-1.0};
  }
}

std::string CatchState::ObservationString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  return ToString();
}

void CatchState::ObservationTensor(Player player,
                                   absl::Span<float> values) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);

  TensorView<2> view(values, {num_rows_, num_columns_}, true);
  if (initialized_) {
    view[{ball_row_, ball_col_}] = 1.0;
    view[{num_rows_ - 1, paddle_col_}] = 1.0;
  }
}

std::unique_ptr<State> CatchState::Clone() const {
  return std::unique_ptr<State>(new CatchState(*this));
}

void CatchState::DoApplyAction(Action move) {
  if (!initialized_) {
    initialized_ = true;
    ball_col_ = move;
    ball_row_ = 0;
    paddle_col_ = num_columns_ / 2;
  } else {
    ball_row_++;
    int direction = move - 1;
    paddle_col_ =
        std::min(std::max(paddle_col_ + direction, 0), num_columns_ - 1);
  }
}

// ---------------------------------------------------------------------------
// OpenSpiel 2.0 struct API
// ---------------------------------------------------------------------------

std::unique_ptr<StateStruct> CatchState::ToStruct() const {
  auto rv = std::make_unique<CatchStateStruct>();
  rv->ball_row = ball_row_;
  rv->ball_col = ball_col_;
  rv->paddle_col = paddle_col_;
  return rv;
}

// The observation for a single-player perfect-information game is the full
// state. We follow the same pattern as TicTacToeState: delegate to ToJson().
std::unique_ptr<ObservationStruct> CatchState::ToObservationStruct(
    Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  return std::make_unique<CatchObservationStruct>(this->ToJson());
}

std::unique_ptr<ActionStruct> CatchState::ActionToStruct(
    Player player, Action action_id) const {
  // Chance actions (ball column selection) are not represented as
  // CatchActionStruct because they don't have a semantic direction.
  SPIEL_CHECK_EQ(player, 0);
  auto action_struct = std::make_unique<CatchActionStruct>();
  action_struct->direction = ActionIdToDirection(action_id);
  return action_struct;
}

std::vector<Action> CatchState::StructToActions(
    const ActionStruct& action_struct) const {
  const auto* a = SafeActionCast<CatchActionStruct>(action_struct);
  return {DirectionToActionId(a->direction)};
}

// ---------------------------------------------------------------------------
// CatchGame
// ---------------------------------------------------------------------------

CatchGame::CatchGame(const GameParameters& params)
    : Game(kGameType, params),
      num_rows_(ParameterValue<int>("rows")),
      num_columns_(ParameterValue<int>("columns")) {}

}  // namespace catch_
}  // namespace open_spiel
