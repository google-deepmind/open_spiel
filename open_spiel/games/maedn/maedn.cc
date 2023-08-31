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

#include "open_spiel/games/maedn/maedn.h"

#include <algorithm>
#include <cstdlib>
#include <set>
#include <utility>
#include <vector>

#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/game_parameters.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace maedn {
namespace {

const std::vector<std::pair<Action, double>> kChanceOutcomes = {
    std::pair<Action, double>(0, 1.0 / 6),
    std::pair<Action, double>(1, 1.0 / 6),
    std::pair<Action, double>(2, 1.0 / 6),
    std::pair<Action, double>(3, 1.0 / 6),
    std::pair<Action, double>(4, 1.0 / 6),
    std::pair<Action, double>(5, 1.0 / 6),
};

const std::vector<int> kChanceOutcomeValues = {1, 2, 3, 4, 5, 6};

// Facts about the game
const GameType kGameType{/*short_name=*/"maedn",
                         /*long_name=*/"Mensch-Aergere-Dich-Nicht",
                         GameType::Dynamics::kSequential,
                         GameType::ChanceMode::kExplicitStochastic,
                         GameType::Information::kPerfectInformation,
                         GameType::Utility::kZeroSum,
                         GameType::RewardModel::kTerminal,
                         /*max_num_players=*/4,
                         /*min_num_players=*/2,
                         /*provides_information_state_string=*/false,
                         /*provides_information_state_tensor=*/false,
                         /*provides_observation_string=*/true,
                         /*provides_observation_tensor=*/true,
                         /*parameter_specification=*/
                         {
                             {"players", GameParameter(2)},
                             {"twoPlayersOpposite", GameParameter(true)},
                         }};

static std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::shared_ptr<const Game>(new MaednGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);

RegisterSingleTensorObserver single_tensor(kGameType.short_name);
}  // namespace

std::string CurPlayerToString(Player cur_player) {
  switch (cur_player) {
    case kRedPlayerId:
      return "1";
    case kBluePlayerId:
      return "2";
    case kGreenPlayerId:
      return "3";
    case kYellowPlayerId:
      return "4";
    case kChancePlayerId:
      return "*";
    case kTerminalPlayerId:
      return "T";
    default:
      SpielFatalError(absl::StrCat("Unrecognized player id: ", cur_player));
  }
}

std::string MaednState::ActionToString(Player player, Action move_id) const {
  if (player == kChancePlayerId) {
    // Normal chance roll.
    return absl::StrCat("chance outcome ", move_id,
                        " (roll: ", kChanceOutcomeValues[move_id], ")");
  } else {
    // Assemble a human-readable string representation of the move.
    if (move_id == kBringInAction) {
      return absl::StrCat(move_id, " - brings in new piece");
    } else if (move_id == kPassAction) {
      return absl::StrCat(move_id, " - passes");
    } else {
      return absl::StrCat(move_id, " - moves piece on field ",
                          move_id - kFieldActionsOffset);
    }
  }
}

std::string MaednState::ObservationString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  return ToString();
}

void MaednState::ObservationTensor(Player player,
                                   absl::Span<float> values) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  SPIEL_CHECK_EQ(values.size(), kStateEncodingSize);
  auto value_it = values.begin();

  // Tensor should contain state from the player's PoV, so relative
  // positions are used and converted to absolute positions.
  int position = PlayerToPosition(player);
  for (int i = 0; i < kNumCommonFields; i++) {
    int abs_pos = RelPosToAbsPos(i, position);
    int piece = board_[abs_pos];
    *value_it++ = ((piece == 1) ? 1 : 0);
    *value_it++ = ((piece == 2) ? 1 : 0);
    *value_it++ = ((piece == 3) ? 1 : 0);
    *value_it++ = ((piece == 4) ? 1 : 0);
  }

  // Rotated goal fields to one hot encoded tensor.
  for (int p = 0; p < kMaxNumPlayers; p++) {
    int ply_position = PlayerToPosition((player + p) % kMaxNumPlayers);
    for (int i = 0; i < kNumGoalFieldsPerPlayer; i++) {
      int abs_pos = RelPosToAbsPos(kNumCommonFields + i, ply_position);
      int piece = board_[abs_pos];
      *value_it++ = ((piece == 1) ? 1 : 0);
      *value_it++ = ((piece == 2) ? 1 : 0);
      *value_it++ = ((piece == 3) ? 1 : 0);
      *value_it++ = ((piece == 4) ? 1 : 0);
    }
  }

  // Rotated number of pieces outside of field per player.
  for (int p = 0; p < kMaxNumPlayers; p++) {
    *value_it++ = (out_[(player + p) % kMaxNumPlayers]);
  }

  if (cur_player_ == kChancePlayerId) {
    // Encode chance player with all zeros.
    for (int i = 0; i < kMaxNumPlayers; i++) {
      *value_it++ = 0;
    }
  } else {
    int rotated_current_player =
        (num_players_ + cur_player_ - player) % num_players_;
    // Rotated current player id to one hot encoded tensor.
    for (int i = 0; i < kMaxNumPlayers; i++) {
      *value_it++ = (rotated_current_player == i) ? 1 : 0;
    }
  }

  *value_it++ = ((dice_ == 1) ? 1 : 0);
  *value_it++ = ((dice_ == 2) ? 1 : 0);
  *value_it++ = ((dice_ == 3) ? 1 : 0);
  *value_it++ = ((dice_ == 4) ? 1 : 0);
  *value_it++ = ((dice_ == 5) ? 1 : 0);
  *value_it++ = ((dice_ == 6) ? 1 : 0);

  SPIEL_CHECK_EQ(value_it, values.end());
}

void MaednState::FromObservationTensor(Player player, absl::Span<float> values,
                                       Player prev_player, int prev_dice) {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  SPIEL_CHECK_EQ(values.size(), kStateEncodingSize);

  prev_player_ = prev_player;
  prev_dice_ = prev_dice;

  auto value_it = values.begin();

  // Tensor should contain state from the player's PoV, so relative
  // positions are used and converted to absolute positions.
  int position = PlayerToPosition(player);
  for (int i = 0; i < kNumCommonFields; i++) {
    int abs_pos = RelPosToAbsPos(i, position);
    int one = *value_it++;
    int two = *value_it++;
    int three = *value_it++;
    int four = *value_it++;
    int piece = one ? 1 : (two ? 2 : (three ? 3 : (four ? 4 : 0)));
    board_[abs_pos] = piece;
  }

  // rotated goal fields to one hot encoded tensor
  for (int p = 0; p < kMaxNumPlayers; p++) {
    int ply_position = PlayerToPosition((player + p) % kMaxNumPlayers);
    for (int i = 0; i < kNumGoalFieldsPerPlayer; i++) {
      int abs_pos = RelPosToAbsPos(kNumCommonFields + i, ply_position);
      int one = *value_it++;
      int two = *value_it++;
      int three = *value_it++;
      int four = *value_it++;
      int piece = one ? 1 : (two ? 2 : (three ? 3 : (four ? 4 : 0)));
      board_[abs_pos] = piece;
    }
  }

  // rotated number of pieces outside of field per player
  for (int p = 0; p < kMaxNumPlayers; p++) {
    out_[(player + p) % kMaxNumPlayers] = *value_it++;
  }

  int zero = *value_it++;
  int one = *value_it++;
  int two = *value_it++;
  int three = *value_it++;

  if (zero + one + two + three == 0) {
    cur_player_ = kChancePlayerId;
  } else {
    int rotated_current_player = zero ? 0 : (one ? 1 : (two ? 2 : 3));

    cur_player_ = (rotated_current_player + player) % num_players_;
  }

  int dice_1 = *value_it++;
  int dice_2 = *value_it++;
  int dice_3 = *value_it++;
  int dice_4 = *value_it++;
  int dice_5 = *value_it++;
  int dice_6 = *value_it++;

  dice_ = dice_1 ? 1
                 : (dice_2 ? 2
                           : (dice_3   ? 3
                              : dice_4 ? 4
                                       : (dice_5 ? 5 : (dice_6 ? 6 : 0))));

  SPIEL_CHECK_EQ(value_it, values.end());
}

MaednState::MaednState(std::shared_ptr<const Game> game,
                       bool two_players_opposite)
    : State(game),
      cur_player_(kChancePlayerId),
      prev_player_(game->NumPlayers() - 1),
      two_players_opposite_(two_players_opposite),
      turns_(0),
      dice_(0),
      prev_dice_(0),
      board_(std::vector<int>(kNumFields, 0)),
      turn_history_info_({}) {
  int i = 0;
  for (; i < num_players_; i++) {
    out_.push_back(4);
  }
  for (; i < kMaxNumPlayers; i++) {
    out_.push_back(0);
  }
}

Player MaednState::CurrentPlayer() const {
  return IsTerminal() ? kTerminalPlayerId : Player{cur_player_};
}

void MaednState::DoApplyAction(Action move) {
  if (IsChanceNode()) {
    // Chance action.
    turn_history_info_.push_back(TurnHistoryInfo(kChancePlayerId, prev_player_,
                                                 dice_, prev_dice_, move, 0));

    SPIEL_CHECK_TRUE(dice_ == 0);
    dice_ = kChanceOutcomeValues[move];
    if (prev_dice_ == 6) {
      // if last dice roll was a 6, same player moves again
      cur_player_ = prev_player_;
    } else {
      // next player
      cur_player_ = (prev_player_ + 1) % num_players_;
      turns_++;
    }
    return;
  }

  // Normal move action.
  int thrown_out_player = -1;

  if (move != kPassAction) {
    if (move == kBringInAction) {
      // Bring in new piece.
      int players_first_field = GetPlayersFirstField(cur_player_);

      thrown_out_player = board_[players_first_field] - 1;
      board_[players_first_field] = cur_player_ + 1;
      out_[cur_player_]--;
    } else {
      // Normal piece move.
      std::pair<int, int> fields =
          GetFieldsFromAction(move, cur_player_, dice_);

      board_[fields.first] = 0;
      thrown_out_player = board_[fields.second] - 1;
      board_[fields.second] = cur_player_ + 1;
    }

    if (thrown_out_player >= 0) {
      out_[thrown_out_player]++;
    }
  }

  turn_history_info_.push_back(TurnHistoryInfo(
      cur_player_, prev_player_, dice_, prev_dice_, move, thrown_out_player));

  prev_player_ = cur_player_;
  prev_dice_ = dice_;

  cur_player_ = kChancePlayerId;
  dice_ = 0;
}

void MaednState::UndoAction(Player player, Action action) {
  {
    const TurnHistoryInfo& thi = turn_history_info_.back();
    SPIEL_CHECK_EQ(thi.player, player);
    SPIEL_CHECK_EQ(action, thi.action);
    cur_player_ = thi.player;
    prev_player_ = thi.prev_player;
    dice_ = thi.dice;
    prev_dice_ = thi.prev_dice;
    if (player != kChancePlayerId && action != kPassAction) {
      // Undo move.
      // Code basically is the inverse of DoApplyAction(Action move).
      if (action == kBringInAction) {
        // Un-bring in new piece.
        int players_first_field = GetPlayersFirstField(cur_player_);

        board_[players_first_field] = thi.thrown_out_player + 1;
        out_[cur_player_]++;
      } else {
        // Normal piece move.
        std::pair<int, int> fields =
            GetFieldsFromAction(action, cur_player_, dice_);

        board_[fields.first] = cur_player_ + 1;
        board_[fields.second] = thi.thrown_out_player + 1;
      }

      if (thi.thrown_out_player >= 0) {
        out_[thi.thrown_out_player]--;
      }
    }
  }
  turn_history_info_.pop_back();
  history_.pop_back();
  --move_number_;
}

std::pair<int, int> MaednState::GetFieldsFromAction(Action action,
                                                    Player player,
                                                    int dice) const {
  int position = PlayerToPosition(player);
  int relative_source_field = action - kFieldActionsOffset;
  int relative_target_field = relative_source_field + dice;

  return {RelPosToAbsPos(relative_source_field, position),
          RelPosToAbsPos(relative_target_field, position)};
}

int MaednState::RelPosToAbsPos(int relative_position, int position) const {
  if (relative_position < kNumCommonFields) {
    int players_first_field = (kNumCommonFields / kMaxNumPlayers) * position;
    return (relative_position + players_first_field) % kNumCommonFields;
  } else {
    return kNumGoalFieldsPerPlayer * position + relative_position;
  }
}

int MaednState::AbsPosToRelPos(int absolute_position, int position) const {
  if (absolute_position < kNumCommonFields) {
    int playersFirstField = (kNumCommonFields / kMaxNumPlayers) * position;
    return (kNumCommonFields + absolute_position - playersFirstField) %
           kNumCommonFields;
  } else {
    return absolute_position - kNumGoalFieldsPerPlayer * position;
  }
}

int MaednState::GetPlayersFirstField(Player player) const {
  int position = PlayerToPosition(player);
  return (kNumCommonFields / kMaxNumPlayers) * position;
}

std::vector<std::pair<Action, double>> MaednState::ChanceOutcomes() const {
  SPIEL_CHECK_TRUE(IsChanceNode());
  return kChanceOutcomes;
}

std::vector<Action> MaednState::LegalActions() const {
  if (IsChanceNode()) return LegalChanceOutcomes();
  if (IsTerminal()) return {};

  std::vector<Action> legal_actions;

  // Follows these rules in this exact order:
  // - If a player's own piece is standing on the start field
  //   and player has at least one piece off the board, player
  //   MUST move the piece on the start field away unless it is
  //   blocked by another own piece. If that is the case,
  //   player is free to move any own piece.
  // - If player rolls a 6 and has at least one piece off the
  //   board, player MUST bring in a new piece.
  // - If player has no (moveable) piece on the board, player
  //   must pass.
  // - In any other case, player is free to move any own piece
  //   on the board.
  int players_first_field = GetPlayersFirstField(cur_player_);
  if (out_[cur_player_] > 0) {
    if (board_[players_first_field] == cur_player_ + 1) {
      // Is piece on start field moveable by dice roll?
      // (playersFirstField + dice) cannot overflow, simple
      // addition is suitable.
      if (board_[players_first_field + dice_] != cur_player_ + 1) {
        legal_actions.push_back(kFieldActionsOffset);
        return legal_actions;
      }
    }

    if (dice_ == 6) {
      // Player MUST bring in a new piece if possible.
      // Check whether start field is bloked.
      if (board_[players_first_field] != cur_player_ + 1) {
        legal_actions.push_back(kBringInAction);
        return legal_actions;
      }
      // Start field is blocked and this piece itself is
      // blocked due (has already been checked).
    }
  }

  // Look for pieces of current player on board if there is at least one:
  if (out_[cur_player_] < 4) {
    int position = PlayerToPosition(cur_player_);
    const int max_field = kNumCommonFields + kNumGoalFieldsPerPlayer - dice_;
    for (int relative_source_field = 0; relative_source_field < max_field;
         relative_source_field++) {
      int relative_target_field = relative_source_field + dice_;

      int absolute_source_field =
          RelPosToAbsPos(relative_source_field, position);
      int absolute_target_field =
          RelPosToAbsPos(relative_target_field, position);

      if (board_[absolute_source_field] == cur_player_ + 1) {
        if (board_[absolute_target_field] != cur_player_ + 1) {
          legal_actions.push_back(relative_source_field + kFieldActionsOffset);
        }
      }
    }
  }

  // If nothing is possible, player must pass.
  if (legal_actions.empty()) {
    legal_actions.push_back(kPassAction);
  }

  return legal_actions;
}

std::string MaednState::ToString() const {
  std::vector<std::string> board_array = {
      ". .     o-o-S     . .", ". .     o . o     . .", "        o . o        ",
      "        o . o        ", "S-o-o-o-o . o-o-o-o-o", "o . . . .   . . . . o",
      "o-o-o-o-o . o-o-o-o-S", "        o . o        ", "        o . o        ",
      ". .     o . o     . .", ". .     S-o-o     . .",
  };

  // Fill the board.
  for (int pos = 0; pos < kNumFields; pos++) {
    if (board_[pos] > 0) {
      Coords coords = kFieldToBoardString[pos];
      board_array[coords.y][coords.x] = 48 + board_[pos];
    }
  }
  // Pieces off the board.
  for (int ply = 0; ply < kMaxNumPlayers; ply++) {
    int out = out_[ply];
    int position = PlayerToPosition(ply);
    int offset = kNumFields + kNumGoalFieldsPerPlayer * position;
    for (int i = 0; i < out; i++) {
      Coords coords = kFieldToBoardString[offset + i];
      board_array[coords.y][coords.x] = 49 + ply;
    }
  }

  std::string board_str = absl::StrJoin(board_array, "\n") + "\n";

  // Extra info like whose turn it is etc.
  absl::StrAppend(&board_str, "Turn: ");
  absl::StrAppend(&board_str, CurPlayerToString(cur_player_));
  absl::StrAppend(&board_str, "\n");
  absl::StrAppend(&board_str, "Dice: ");
  absl::StrAppend(&board_str, dice_ != 0 ? std::to_string(dice_) : "");
  absl::StrAppend(&board_str, "\n");

  return board_str;
}

bool MaednState::AllInGoal(Player player) const {
  int position = PlayerToPosition(player);
  int offset = kNumCommonFields + position * kNumGoalFieldsPerPlayer;
  return board_[offset] != 0 && board_[offset + 1] != 0 &&
         board_[offset + 2] != 0 && board_[offset + 3] != 0;
}

bool MaednState::IsTerminal() const {
  for (int ply = 0; ply < num_players_; ply++) {
    if (AllInGoal(ply)) {
      return true;
    }
  }
  return false;
}

std::vector<double> MaednState::Returns() const {
  std::vector<double> returns;

  if (IsTerminal()) {
    for (int ply = 0; ply < num_players_; ply++) {
      returns.push_back(AllInGoal(ply) ? num_players_ - 1.0 : -1.0);
    }
  } else {
    for (int ply = 0; ply < num_players_; ply++) {
      returns.push_back(0.0);
    }
  }

  return returns;
}

std::unique_ptr<State> MaednState::Clone() const {
  return std::unique_ptr<State>(new MaednState(*this));
}

void MaednState::SetState(int cur_player, int dice, int prev_player,
                          int prev_dice, const std::vector<int>& board,
                          const std::vector<int>& out) {
  cur_player_ = cur_player;
  prev_player_ = prev_player;
  dice_ = dice;
  prev_dice_ = prev_dice;
  board_ = board;
  out_ = out;
}

MaednGame::MaednGame(const GameParameters& params)
    : Game(kGameType, params),
      two_player_opposite_(ParameterValue<bool>("twoPlayersOpposite")),
      num_players_(ParameterValue<int>("players")) {
  SPIEL_CHECK_GE(num_players_, kGameType.min_num_players);
  SPIEL_CHECK_LE(num_players_, kGameType.max_num_players);
}

}  // namespace maedn
}  // namespace open_spiel
