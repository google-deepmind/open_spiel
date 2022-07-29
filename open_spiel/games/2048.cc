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

constexpr int kMoveUp = 0;
constexpr int kMoveRight = 1;
constexpr int kMoveDown = 2;
constexpr int kMoveLeft = 3;
const std::vector<Action> kPlayerActions = {kMoveUp, kMoveRight, kMoveDown, kMoveLeft};

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
                         /*provides_observation_tensor=*/true};

std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::shared_ptr<const Game>(new TwoZeroFourEightGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);
}  // namespace

TwoZeroFourEightState::TwoZeroFourEightState(std::shared_ptr<const Game> game)
    : State(game) {
  board_ = std::vector<Tile>(kDefaultRows * kDefaultColumns, Tile(0, false));
}

void TwoZeroFourEightState::SetCustomBoard(const std::vector<int> board_seq) {
  current_player_ = 0;
  for (int x = 0; x < kDefaultRows; x++) {
    for (int y = 0; y < kDefaultColumns; y++) {
      SetBoard(x, y, Tile(board_seq[x * kDefaultRows + y], false));
    }
  }
}

ChanceAction TwoZeroFourEightState::SpielActionToChanceAction(Action action) const {
  std::vector<int> values = UnrankActionMixedBase(
      action, {kDefaultRows, kDefaultColumns, kChanceTiles.size()});
  return ChanceAction(values[0], values[1], values[2]);
}

Action TwoZeroFourEightState::ChanceActionToSpielAction(ChanceAction move) const {
  std::vector<int> action_bases = {kDefaultRows, kDefaultColumns, kChanceTiles.size()};
  return RankActionMixedBase(
      action_bases, {move.row, move.column, move.is_four});
}

std::vector<std::vector<int>> TwoZeroFourEightState::BuildTraversals(int direction) const {
  std::vector<int> x, y;
  for (int pos = 0; pos < kDefaultRows; pos++) {
    x.push_back(pos);    
  }
  for (int pos = 0; pos < kDefaultColumns; pos++) {
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

bool TwoZeroFourEightState::WithinBounds(int x, int y) const {
  return x >= 0 && x < kDefaultRows && y >= 0 && y < kDefaultColumns;
};

bool TwoZeroFourEightState::CellAvailable(int x, int y) const {
  return BoardAt(x, y).value == 0;
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

std::vector<Coordinate> TwoZeroFourEightState::FindFarthestPosition(int x, int y, int direction) const {  
  // Progress towards the vector direction until an obstacle is found
  Coordinate prev = Coordinate(x, y);
  do {
    prev = Coordinate(x, y);
    Coordinate direction_diff = GetVector(direction);
    x += direction_diff.x;
    y += direction_diff.y;    
  } while (WithinBounds(x, y) && CellAvailable(x, y));
  return std::vector<Coordinate> {prev,
      Coordinate(x, y)};
};

// Check for available matches between tiles (more expensive check)
bool TwoZeroFourEightState::TileMatchesAvailable() const {
  for (int x = 0; x < kDefaultRows; x++) {
    for (int y = 0; y < kDefaultColumns; y++) {
      int tile = BoardAt(x, y).value;
      if (tile > 0) {
        for (int direction = 0; direction < 4; direction++) {
          Coordinate vector = GetVector(direction);
          int other = GetCellContent(x + vector.x, y + vector.y);
          if (other > 0 && other == tile) {
            return true; // These two tiles can be merged
          }
        }
      }
    }
  }
  return false;
};

void TwoZeroFourEightState::PrepareTiles() {
  for (int x = 0; x < kDefaultRows; x++) {
    for (int y = 0; y < kDefaultColumns; y++) {
      Tile tile = BoardAt(x, y);
      if (tile.is_merged) {
        SetBoard(x, y, Tile(tile.value, false));
      }
    }
  }  
};

int TwoZeroFourEightState::GetCellContent(int x, int y) const {
  if (!WithinBounds(x, y))
    return 0;
  return BoardAt(x, y).value;
}

void TwoZeroFourEightState::DoApplyAction(Action action) {
  if (IsChanceNode()) {
    current_player_ = 0;
    if (action == kNoCellAvailableAction) {
      return;
    }
    ChanceAction chance_action = SpielActionToChanceAction(action);
    SetBoard(chance_action.row, chance_action.column,
        Tile(chance_action.is_four ? kChanceTiles[1] : kChanceTiles[0], false));
    return;
  }
  std::vector<std::vector<int>> traversals = BuildTraversals(action);
  PrepareTiles();
  for (int x : traversals[0]) {
    for (int y : traversals[1]) {
      int tile = GetCellContent(x, y);
      if (tile > 0) {
        bool moved = false;
        std::vector<Coordinate> positions = FindFarthestPosition(x, y, action);
        Coordinate farthest_pos = positions[0];
        Coordinate next_pos = positions[1];
        int next_cell = GetCellContent(next_pos.x, next_pos.y);
        if (next_cell > 0 && next_cell == tile
            && !BoardAt(next_pos.x, next_pos.y).is_merged) {
          int merged = tile * 2;
          SetBoard(next_pos.x, next_pos.y, Tile(merged, true));
          moved = true;
        } else if (farthest_pos.x != x || farthest_pos.y != y){
          SetBoard(farthest_pos.x, farthest_pos.y, Tile(tile, false));
          moved = true;
        }
        if (moved) {
          SetBoard(x, y, Tile(0, false));
        }        
      }
    }
  }
  current_player_ = kChancePlayerId;
}

std::string TwoZeroFourEightState::ActionToString(Player player,
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
  for (int r = 0; r < kDefaultRows; r++) {
    for (int c = 0; c < kDefaultColumns; c++) {
      if (BoardAt(r, c).value == 0) {
        count++;
      }
    }
  }
  return count;
}

ActionsAndProbs TwoZeroFourEightState::ChanceOutcomes() const {
  ActionsAndProbs action_and_probs;
  int count = AvailableCellCount();
  if (count == 0) {
    action_and_probs.reserve(1);
    action_and_probs.emplace_back(kNoCellAvailableAction, 1);
    return action_and_probs;  
  }
  action_and_probs.reserve(count * 2);
  for (int r = 0; r < kDefaultRows; r++) {
    for (int c = 0; c < kDefaultColumns; c++) {
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

std::vector<Action> TwoZeroFourEightState::LegalActions() const {
  if (IsTerminal()) {
    return {};
  }
  if (IsChanceNode()) {
    return LegalChanceOutcomes();
  }
  return kPlayerActions;
}

bool TwoZeroFourEightState::InBounds(int row, int column) const {
  return (row >= 0 && row < kDefaultRows && column >= 0
      && column < kDefaultColumns);
}

std::string TwoZeroFourEightState::ToString() const {  
  std::string str;
  for (int r = 0; r < kDefaultRows; ++r) {
    for (int c = 0; c < kDefaultColumns; ++c) {
      std::string tile = std::to_string(BoardAt(r, c).value);
      absl::StrAppend(&str, std::string(5 - tile.length(), ' '));
      absl::StrAppend(&str, tile);
    }
    absl::StrAppend(&str, "\n");
  }
  return str;  
}

bool TwoZeroFourEightState::IsTerminal() const {
  return Reached2048() || (AvailableCellCount() == 0 && !TileMatchesAvailable());
}

bool TwoZeroFourEightState::Reached2048() const {
  for (int r = 0; r < kDefaultRows; r++) {
    for (int c = 0; c < kDefaultColumns; c++) {
      if (BoardAt(r, c).value == 2048) {
        return true;
      }
    }
  }
  return false;
}

std::vector<double> TwoZeroFourEightState::Returns() const {
  if (IsTerminal()) {
    if (Reached2048()) {
      return {1.0};
    }
    return {-1.0};
  }
  return {0.};  
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
  TensorView<2> view(values, {kDefaultRows, kDefaultColumns}, true);
  for (int row = 0; row < kDefaultRows; row++) {
    for (int column = 0; column < kDefaultColumns; column++) {
      view[{row, column}] = BoardAt(row, column).value;
    }
  }
}

void TwoZeroFourEightState::UndoAction(Player player, Action action) {  
  history_.pop_back();
}

TwoZeroFourEightGame::TwoZeroFourEightGame(const GameParameters& params)
    : Game(kGameType, params) {}

int TwoZeroFourEightGame::NumDistinctActions() const {
  return kPlayerActions.size();
}

}  // namespace two_zero_four_eight
}  // namespace open_spiel
