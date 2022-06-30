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

using hex::CellState;
using hex::kMinValueCellState;

using hex::PlayerToState;
using hex::StateToString;

// Game Facts
const GameType kGameType{/*short_name=*/"dark_hex",
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
                         {{"obstype", GameParameter(kDefaultObsType)},
                          {"gameversion", GameParameter(kDefaultGameVersion)},
                          {"board_size", GameParameter(kDefaultBoardSize)},
                          {"num_cols", GameParameter(kDefaultNumCols)},
                          {"num_rows", GameParameter(kDefaultNumRows)}}};

const GameType kImperfectRecallGameType{
    /*short_name=*/"dark_hex_ir",
    /*long_name=*/"Dark Hex with Imperfect Recall",
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
    {{"obstype", GameParameter(kDefaultObsType)},
     {"gameversion", GameParameter(kDefaultGameVersion)},
     {"board_size", GameParameter(kDefaultBoardSize)},
     {"num_cols", GameParameter(kDefaultNumCols)},
     {"num_rows", GameParameter(kDefaultNumRows)}}};

std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::shared_ptr<const Game>(new DarkHexGame(params, kGameType));
}

std::shared_ptr<const Game> ImperfectRecallFactory(
    const GameParameters& params) {
  return std::shared_ptr<const Game>(new ImperfectRecallDarkHexGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);
REGISTER_SPIEL_GAME(kImperfectRecallGameType, ImperfectRecallFactory);

}  // namespace

ImperfectRecallDarkHexGame::ImperfectRecallDarkHexGame(
    const GameParameters& params)
    : DarkHexGame(params, kImperfectRecallGameType) {}

DarkHexState::DarkHexState(std::shared_ptr<const Game> game, int num_cols,
                           int num_rows, GameVersion game_version,
                           ObservationType obs_type)
    : State(game),
      state_(game, num_cols, num_rows),
      obs_type_(obs_type),
      game_version_(game_version),
      num_cols_(num_cols),
      num_rows_(num_rows),
      num_cells_(num_cols * num_rows),
      bits_per_action_(num_cells_ + 1),
      longest_sequence_(num_cells_ * 2 - 1) {
  black_view_.resize(num_cols * num_rows, CellState::kEmpty);
  white_view_.resize(num_cols * num_rows, CellState::kEmpty);
}

void DarkHexState::DoApplyAction(Action move) {
  Player cur_player = CurrentPlayer();  // current player
  auto& cur_view = (cur_player == 0 ? black_view_ : white_view_);

  // Either occupied or not
  if (game_version_ == GameVersion::kClassicalDarkHex) {
    if (state_.BoardAt(move) == CellState::kEmpty) {
      state_.ApplyAction(move);
    }
  } else {
    SPIEL_CHECK_EQ(game_version_, GameVersion::kAbruptDarkHex);
    if (state_.BoardAt(move) == CellState::kEmpty) {
      state_.ApplyAction(move);
    } else {
      // switch the current player
      state_.ChangePlayer();
    }
  }

  SPIEL_CHECK_EQ(cur_view[move], CellState::kEmpty);
  // Update the view - only using CellState::kBlack and CellState::kWhite
  if (state_.BoardAt(move) == CellState::kBlack ||
      state_.BoardAt(move) == CellState::kBlackNorth ||
      state_.BoardAt(move) == CellState::kBlackSouth) {
    cur_view[move] = CellState::kBlack;
  } else if (state_.BoardAt(move) == CellState::kWhite ||
             state_.BoardAt(move) == CellState::kWhiteEast ||
             state_.BoardAt(move) == CellState::kWhiteWest) {
    cur_view[move] = CellState::kWhite;
  } else if (state_.BoardAt(move) == CellState::kBlackWin ||
             state_.BoardAt(move) == CellState::kWhiteWin) {
    cur_view[move] = state_.BoardAt(move);
  } else {
    SPIEL_CHECK_TRUE(false);
  }
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

  for (int r = 0; r < num_rows_; ++r) {
    for (int c = 0; c < num_cols_; ++c) {
      absl::StrAppend(&str, StateToString(cur_view[r * num_cols_ + c]));
    }
    if (r < (num_rows_ - 1)) {
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
    values[cell * kCellStates +
           (static_cast<int>(player_view[cell]) - kMinValueCellState)] = 1.0;
  }

  // Encoding the sequence
  int offset = num_cells_ * kCellStates;
  for (const auto& player_with_action : action_sequence_) {
    if (player_with_action.first == player) {
      // Always include the observing player's actions.
      values[offset] = player_with_action.first;
      values[offset + 1 + player_with_action.second] = 1.0;
    } else if (obs_type_ == ObservationType::kRevealNumTurns) {
      // If the number of turns are revealed, then each of the other player's
      // actions will show up as unknowns. Here, num_cells_ + 1 is used to
      // encode "unknown".
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
    values[cell * kCellStates +
           (static_cast<int>(player_view[cell]) - kMinValueCellState)] = 1.0;
  }

  if (obs_type_ == ObservationType::kRevealNumTurns) {
    values[num_cells_ * kCellStates + action_sequence_.size()] = 1.0;
  }
}

std::unique_ptr<State> DarkHexState::Clone() const {
  return std::unique_ptr<State>(new DarkHexState(*this));
}

DarkHexGame::DarkHexGame(const GameParameters& params, GameType game_type)
    : Game(game_type, params),
      game_(std::static_pointer_cast<const hex::HexGame>(LoadGame(
          "hex",
          {{"num_cols", GameParameter(ParameterValue<int>(
                            "num_cols", ParameterValue<int>("board_size")))},
           {"num_rows",
            GameParameter(ParameterValue<int>(
                "num_rows", ParameterValue<int>("board_size")))}}))),
      num_cols_(
          ParameterValue<int>("num_cols", ParameterValue<int>("board_size"))),
      num_rows_(
          ParameterValue<int>("num_rows", ParameterValue<int>("board_size"))),
      num_cells_(num_cols_ * num_rows_),
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

  std::string game_version = ParameterValue<std::string>("gameversion");
  if (game_version == "cdh") {
    game_version_ = GameVersion::kClassicalDarkHex;
  } else if (game_version == "adh") {
    game_version_ = GameVersion::kAbruptDarkHex;
  } else {
    SpielFatalError(absl::StrCat("Unrecognized game version: ", game_version));
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
