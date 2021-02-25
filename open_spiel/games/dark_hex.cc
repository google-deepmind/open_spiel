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

#include "open_spiel/games/dark_hex.h"

#include <memory>
#include <utility>
#include <vector>

#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/games/hex.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace dark_hex {
namespace {

using hex::kCellStates;
using hex::kDefaultBoardSize;

using hex::CellState;
using hex::kMinValueCellState;

using hex::PlayerToState;
using hex::StateToString;

// Game Facts
const GameType kGameType{
    /*short_name=*/"dark_hex",
    /*long_name=*/"Dark Hex",
    GameType::Dynamics::kSequential,
    GameType::ChanceMode::kDeterministic,
    GameType::Information::kImperfectInformation,
    GameType::Utility::kZeroSum,
    GameType::RewardModel::kTerminal,
    /*max_num_players=*/2,
    /*min_num_players=*/2,
    /*provides_information_state_string=*/true,
    /*provides_information_state_tensor=*/true,
    /*provides_observation_string=*/true,
    /*provides_observation_tensor=*/true,
    /*parameter_specification=*/
    {
        {"obstype", GameParameter(std::string(kDefaultObsType))},
        {"board_size", GameParameter(hex::kDefaultBoardSize)},
    }};

std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::shared_ptr<const Game>(new DarkHexGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);

}  // namespace

DarkHexState::DarkHexState(std::shared_ptr<const Game> game, int board_size,
                           ObservationType obs_type)
    : State(game),
      state_(game, board_size),
      obs_type_(obs_type),
      board_size_(board_size),
      num_cells_(board_size * board_size),
      bits_per_action_(num_cells_ + 1),
      longest_sequence_(num_cells_ * 2 - 1) {
  black_view_.resize(board_size * board_size, CellState::kEmpty);
  white_view_.resize(board_size * board_size, CellState::kEmpty);
}

void DarkHexState::DoApplyAction(Action move) {
  Player cur_player = CurrentPlayer();  // current player
  auto& cur_view = (cur_player == 0 ? black_view_ : white_view_);

  // Either occupied or not
  if (state_.BoardAt(move) == CellState::kEmpty) {
    state_.ApplyAction(move);
  }

  SPIEL_CHECK_EQ(cur_view[move], CellState::kEmpty);
  cur_view[move] = state_.BoardAt(move);
  action_sequence_.push_back(std::pair<int, Action>(cur_player, move));
}

std::vector<Action> DarkHexState::LegalActions() const {
  if (IsTerminal()) return {};
  std::vector<Action> moves;
  const Player player = CurrentPlayer();
  const auto& cur_view = (player == 0 ? black_view_ : white_view_);

  for (Action move = 0; move < num_cells_; ++move) {
    if (cur_view[move] == CellState::kEmpty) {
      moves.push_back(move);
    }
  }

  return moves;
}

std::string DarkHexState::ViewToString(Player player) const {
  const auto& cur_view = (player == 0 ? black_view_ : white_view_);
  std::string str;

  // TODO: Hex - Dark-Hex Research sometimes experiments on different sized
  // columns and rows for the game board. Therefore we need to add c-r option
  // instead of board_size only.
  int num_rows = board_size_, num_cols = board_size_;
  for (int r = 0; r < num_rows; ++r) {
    for (int c = 0; c < num_cols; ++c) {
      absl::StrAppend(&str, StateToString(cur_view[r * num_cols + c]));
    }
    if (r < (num_rows - 1)) {
      absl::StrAppend(&str, "\n");
    }
  }
  return str;
}

std::string DarkHexState::ActionSequenceToString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);

  std::string str;
  for (const auto& player_with_action : action_sequence_) {
    if (player_with_action.first == player) {
      absl::StrAppend(&str, player_with_action.first, ",");
      absl::StrAppend(&str, player_with_action.second, " ");
    } else if (obs_type_ == ObservationType::kRevealNumTurns) {
      absl::StrAppend(&str, player_with_action.first, ",? ");
    } else {
      SPIEL_CHECK_EQ(obs_type_, ObservationType::kRevealNothing);
    }
  }
  return str;
}

std::string DarkHexState::InformationStateString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  std::string str;
  absl::StrAppend(&str, ViewToString(player), "\n");
  absl::StrAppend(&str, history_.size(), "\n");
  absl::StrAppend(&str, ActionSequenceToString(player));
  return str;
}

void DarkHexState::InformationStateTensor(Player player,
                                          absl::Span<float> values) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);

  const auto& player_view = (player == 0 ? black_view_ : white_view_);

  SPIEL_CHECK_EQ(values.size(), num_cells_ * kCellStates +
                                    longest_sequence_ * (1 + bits_per_action_));
  std::fill(values.begin(), values.end(), 0.);
  for (int cell = 0; cell < num_cells_; ++cell) {
    values[num_cells_ *
               (static_cast<int>(player_view[cell]) - kMinValueCellState) +
           cell] = 1.0;
  }

  // Encoding the sequence
  int offset = num_cells_ * kCellStates;
  for (const auto& player_with_action : action_sequence_) {
    if (player_with_action.first == player) {
      values[offset] = player_with_action.first;
      values[offset + 1 + player_with_action.second] = 1.0;
    } else if (obs_type_ == ObservationType::kRevealNumTurns) {
      values[offset] = player_with_action.first;
      values[offset + 1 + num_cells_ + 1] = 1.0;
    } else {
      SPIEL_CHECK_EQ(obs_type_, ObservationType::kRevealNothing);
    }
    offset += (1 + bits_per_action_);
  }
}

std::string DarkHexState::ObservationString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  std::string observation = ViewToString(player);
  if (obs_type_ == ObservationType::kRevealNumTurns) {
    absl::StrAppend(&observation, "\nTotal turns: ", action_sequence_.size());
  }
  return observation;
}

void DarkHexState::ObservationTensor(Player player,
                                     absl::Span<float> values) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  SPIEL_CHECK_EQ(values.size(), game_->ObservationTensorSize());
  std::fill(values.begin(), values.end(), 0.);

  const auto& player_view = (player == 0 ? black_view_ : white_view_);
  for (int cell = 0; cell < num_cells_; ++cell) {
    values[num_cells_ *
               (static_cast<int>(player_view[cell]) - kMinValueCellState) +
           cell] = 1.0;
  }

  if (obs_type_ == ObservationType::kRevealNumTurns) {
    values[num_cells_ * kCellStates + action_sequence_.size()] = 1.0;
  }
}

std::unique_ptr<State> DarkHexState::Clone() const {
  return std::unique_ptr<State>(new DarkHexState(*this));
}

void DarkHexState::UndoAction(Player player, Action move) {
  Action last_move = action_sequence_.back().second;
  SPIEL_CHECK_EQ(last_move, move);

  if (state_.BoardAt(move) == PlayerToState(player)) {
    state_.UndoAction(player, move);
  } else {
  }

  auto& player_view = (player == 0 ? black_view_ : white_view_);
  player_view[move] = CellState::kEmpty;
  action_sequence_.pop_back();

  history_.pop_back();
}

DarkHexGame::DarkHexGame(const GameParameters& params)
    : Game(kGameType, params),
      game_(std::static_pointer_cast<const hex::HexGame>(LoadGame(
          "hex",
          {{"board_size", GameParameter(ParameterValue<int>("board_size"))}}))),
      board_size_(ParameterValue<int>("board_size")),
      num_cells_(board_size_ * board_size_),
      bits_per_action_(num_cells_ + 1),
      longest_sequence_(num_cells_ * 2 - 1) {
  std::string obs_type = ParameterValue<std::string>("obstype");
  if (obs_type == "reveal-nothing") {
    obs_type_ = ObservationType::kRevealNothing;
  } else if (obs_type == "reveal-numturns") {
    obs_type_ = ObservationType::kRevealNumTurns;
  } else {
    SpielFatalError(absl::StrCat("Unrecognized observation type: ", obs_type));
  }
}

std::vector<int> DarkHexGame::InformationStateTensorShape() const {
  return {num_cells_ * kCellStates +
          longest_sequence_ * (1 + bits_per_action_)};
}

std::vector<int> DarkHexGame::ObservationTensorShape() const {
  if (obs_type_ == ObservationType::kRevealNothing) {
    return {num_cells_ * kCellStates};
  } else if (obs_type_ == ObservationType::kRevealNumTurns) {
    return {num_cells_ * kCellStates + longest_sequence_};
  } else {
    SpielFatalError("Uknown observation type");
  }
}

}  // namespace dark_hex
}  // namespace open_spiel
