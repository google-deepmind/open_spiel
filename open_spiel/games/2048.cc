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
namespace twenty_forty_eight {
namespace {

constexpr int kMoveUp = 0;
constexpr int kMoveRight = 1;
constexpr int kMoveDown = 2;
constexpr int kMoveLeft = 3;
inline const std::vector<Action> kPlayerActions() {
  return {kMoveUp, kMoveRight, kMoveDown, kMoveLeft};
}

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
                         {{"max_game_length", GameParameter(kMaxGameLength)},
                          {"max_score", GameParameter(kMaxScore)}}};

std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::shared_ptr<const Game>(new TwentyFortyEightGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);
}  // namespace

TwentyFortyEightState::TwentyFortyEightState(std::shared_ptr<const Game> game)
    : State(game),
      board_(std::vector<Tile>(kRows * kColumns, Tile(0, false)))
      {}

void TwentyFortyEightState::SetCustomBoard(const std::vector<int>& board_seq) {
  current_player_ = 0;
  for (int r = 0; r < kRows; r++) {
    for (int c = 0; c < kColumns; c++) {
      SetBoard(r, c, Tile(board_seq[r * kRows + c], false));
    }
  }
}

ChanceAction TwentyFortyEightState
    ::SpielActionToChanceAction(Action action) const {
  std::vector<int> values = UnrankActionMixedBase(
      action, {kRows, kColumns, kChanceTiles.size()});
  return ChanceAction(values[0], values[1], values[2]);
}

Action TwentyFortyEightState
    ::ChanceActionToSpielAction(ChanceAction move) const {
  std::vector<int> action_bases = {kRows, kColumns, 
      kChanceTiles.size()};
  return RankActionMixedBase(
      action_bases, {move.row, move.column, move.is_four});
}

std::vector<std::vector<int>> TwentyFortyEightState
    ::BuildTraversals(int direction) const {
  std::vector<int> x, y;
  for (int pos = 0; pos < kRows; pos++) {
    x.push_back(pos);    
  }
  for (int pos = 0; pos < kColumns; pos++) {
    y.push_back(pos);    
  }
  switch (direction) {
    case kMoveRight:
      reverse(x.begin(), x.end());
      reverse(y.begin(), y.end());
      break;
    case kMoveLeft:
    case kMoveDown:
      reverse(x.begin(), x.end());
      break;
  }
  return {x, y};
};

bool TwentyFortyEightState::WithinBounds(int r, int c) const {
  return r >= 0 && r < kRows && c >= 0 && c < kColumns;
};

bool TwentyFortyEightState::CellAvailable(int r, int c) const {
  return BoardAt(r, c).value == 0;
}

Coordinate GetVector(int direction) {
  switch (direction) {
      case kMoveUp:
        return Coordinate(-1, 0);        
      case kMoveRight:
        return Coordinate(0, 1);
      case kMoveDown:
        return Coordinate(1, 0);
      case kMoveLeft:
        return Coordinate(0, -1);
    }
}

std::vector<Coordinate> TwentyFortyEightState
    ::FindFarthestPosition(int r, int c, int direction) const {  
  // Progress towards the vector direction until an obstacle is found
  Coordinate prev = Coordinate(r, c);
  do {
    prev = Coordinate(r, c);
    Coordinate direction_diff = GetVector(direction);
    r += direction_diff.row;
    c += direction_diff.column;
  } while (WithinBounds(r, c) && CellAvailable(r, c));
  return std::vector<Coordinate> {prev, Coordinate(r, c)};
};

// Check for available matches between tiles (more expensive check)
bool TwentyFortyEightState::TileMatchesAvailable() const {
  for (int r = 0; r < kRows; r++) {
    for (int c = 0; c < kColumns; c++) {
      int tile = BoardAt(r, c).value;
      if (tile > 0) {
        for (int direction = 0; direction < 4; direction++) {
          Coordinate vector = GetVector(direction);
          int other = GetCellContent(r + vector.row, c + vector.column);
          if (other > 0 && other == tile) {
            return true; // These two tiles can be merged
          }
        }
      }
    }
  }
  return false;
};

void TwentyFortyEightState::PrepareTiles() {
  for (int r = 0; r < kRows; r++) {
    for (int c = 0; c < kColumns; c++) {
      Tile tile = BoardAt(r, c);
      if (tile.is_merged) {
        SetBoard(r, c, Tile(tile.value, false));
      }
    }
  }  
};

int TwentyFortyEightState::GetCellContent(int r, int c) const {
  if (!WithinBounds(r, c))
    return 0;
  return BoardAt(r, c).value;
}

void TwentyFortyEightState::DoApplyAction(Action action) {
  if (IsChanceNode()) {
    // The original 2048 game starts with two random tiles
    if (!extra_chance_turn_) {
      current_player_ = 0;
    }
    extra_chance_turn_ = false;
    
    if (action == kNoCellAvailableAction) {
      return;
    }
    ChanceAction chance_action = SpielActionToChanceAction(action);
    SetBoard(chance_action.row, chance_action.column,
        Tile(chance_action.is_four ? kChanceTiles[1] : kChanceTiles[0], false));
    return;
  }
  action_score_ = 0;
  std::vector<std::vector<int>> traversals = BuildTraversals(action);
  PrepareTiles();
  for (int r : traversals[0]) {
    for (int c : traversals[1]) {
      int tile = GetCellContent(r, c);
      if (tile > 0) {
        bool moved = false;
        std::vector<Coordinate> positions = FindFarthestPosition(r, c, action);
        Coordinate farthest_pos = positions[0];
        Coordinate next_pos = positions[1];
        int next_cell = GetCellContent(next_pos.row, next_pos.column);
        if (next_cell > 0 && next_cell == tile
            && !BoardAt(next_pos.row, next_pos.column).is_merged) {
          int merged = tile * 2;
          action_score_ += merged;
          SetBoard(next_pos.row, next_pos.column, Tile(merged, true));
          moved = true;
        } else if (farthest_pos.row != r || farthest_pos.column != c){
          SetBoard(farthest_pos.row, farthest_pos.column, Tile(tile, false));
          moved = true;
        }
        if (moved) {
          SetBoard(r, c, Tile(0, false));
          current_player_ = kChancePlayerId;
        }        
      }
    }
  }  
  total_score_ += action_score_;
}

std::string TwentyFortyEightState::ActionToString(Player player,
                                          Action action_id) const {
  if (IsChanceNode()) {
    if (action_id == kNoCellAvailableAction) {
      return "No Cell Available";
    }
    ChanceAction chance_action = SpielActionToChanceAction(action_id);
    return absl::StrCat(std::to_string(chance_action.is_four ? 4 : 2), 
        " added to row ", std::to_string(chance_action.row + 1),
        ", column ", std::to_string(chance_action.column + 1));
  }
  switch (action_id) {
    case kMoveUp:
      return "Up";
    case kMoveRight:
      return "Right";
    case kMoveDown:
      return "Down";
    case kMoveLeft:
      return "Left";
    default:
      return "Invalid action";
  }  
}

int TwentyFortyEightState::AvailableCellCount() const {
  int count = 0;
  for (int r = 0; r < kRows; r++) {
    for (int c = 0; c < kColumns; c++) {
      if (BoardAt(r, c).value == 0) {
        count++;
      }
    }
  }
  return count;
}

ActionsAndProbs TwentyFortyEightState::ChanceOutcomes() const {  
  int count = AvailableCellCount();
  if (count == 0) {
    return {{kNoCellAvailableAction, 1.0}};    
  }
  ActionsAndProbs action_and_probs;
  action_and_probs.reserve(count * 2);
  for (int r = 0; r < kRows; r++) {
    for (int c = 0; c < kColumns; c++) {
      if (BoardAt(r, c).value == 0) {
        // 2 appearing randomly on the board should be 9 times as likely as a 4
        action_and_probs.emplace_back(ChanceActionToSpielAction(
            ChanceAction(r, c, false)), .9 / count);
        action_and_probs.emplace_back(ChanceActionToSpielAction(
            ChanceAction(r, c, true)), .1 / count);
      }      
    }
  }  
  return action_and_probs;
}

std::vector<Action> TwentyFortyEightState::LegalActions() const {
  if (IsTerminal()) {
    return {};
  }
  if (IsChanceNode()) {
    return LegalChanceOutcomes();
  }
  return kPlayerActions();
}

bool TwentyFortyEightState::InBounds(int row, int column) const {
  return (row >= 0 && row < kRows && column >= 0
      && column < kColumns);
}

std::string TwentyFortyEightState::ToString() const {  
  std::string str;
  for (int r = 0; r < kRows; ++r) {
    for (int c = 0; c < kColumns; ++c) {
      std::string tile = std::to_string(BoardAt(r, c).value);
      absl::StrAppend(&str, std::string(5 - tile.length(), ' '));
      absl::StrAppend(&str, tile);
    }
    absl::StrAppend(&str, "\n");
  }
  return str;  
}

bool TwentyFortyEightState::IsTerminal() const {
  return Reached2048() 
      || (AvailableCellCount() == 0 && !TileMatchesAvailable());
}

bool TwentyFortyEightState::Reached2048() const {
  for (int r = 0; r < kRows; r++) {
    for (int c = 0; c < kColumns; c++) {
      if (BoardAt(r, c).value == 2048) {
        return true;
      }
    }
  }
  return false;
}

std::vector<double> TwentyFortyEightState::Rewards() const {
  return {action_score_};
}

std::vector<double> TwentyFortyEightState::Returns() const {
  return {total_score_};
}

std::string TwentyFortyEightState::InformationStateString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  return HistoryString();
}

std::string TwentyFortyEightState::ObservationString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  return ToString();
}

void TwentyFortyEightState::ObservationTensor(Player player,
                                      absl::Span<float> values) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  TensorView<2> view(values, {kRows, kColumns}, true);
  for (int row = 0; row < kRows; row++) {
    for (int column = 0; column < kColumns; column++) {
      view[{row, column}] = BoardAt(row, column).value;
    }
  }
}

void TwentyFortyEightState::UndoAction(Player player, Action action) {  
  history_.pop_back();
}

TwentyFortyEightGame::TwentyFortyEightGame(const GameParameters& params)
    : Game(kGameType, params),
      max_game_length_(ParameterValue<int>("max_game_length")),
      max_score_(ParameterValue<int>("max_score")) {}

int TwentyFortyEightGame::NumDistinctActions() const {
  return kPlayerActions().size();
}

}  // namespace twenty_forty_eight
}  // namespace open_spiel