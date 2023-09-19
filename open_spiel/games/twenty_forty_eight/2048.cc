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

#include "open_spiel/games/twenty_forty_eight/2048.h"

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/spiel_globals.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/utils/tensor_view.h"

namespace open_spiel {
namespace twenty_forty_eight {
namespace {

constexpr std::array<Action, 4> kPlayerActions = {kMoveUp, kMoveRight,
                                                  kMoveDown, kMoveLeft};

// Facts about the game.
const GameType kGameType{
    /*short_name=*/"2048",
    /*long_name=*/"2048",
    GameType::Dynamics::kSequential,
    GameType::ChanceMode::kExplicitStochastic,
    GameType::Information::kPerfectInformation,
    GameType::Utility::kGeneralSum,
    GameType::RewardModel::kRewards,
    /*max_num_players=*/1,
    /*min_num_players=*/1,
    /*provides_information_state_string=*/false,
    /*provides_information_state_tensor=*/false,
    /*provides_observation_string=*/true,
    /*provides_observation_tensor=*/true,
    {{"max_tile", GameParameter(kDefaultMaxTile)}}};

std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::shared_ptr<const Game>(new TwentyFortyEightGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);

RegisterSingleTensorObserver single_tensor(kGameType.short_name);

constexpr bool InBounds(int r, int c) {
  return r >= 0 && r < kRows && c >= 0 && c < kColumns;
}

// An array that dictates the order of traveral of row and column coordinated
// by direction. E.g, kTraversals[direction][0] is an array of size four
// refering to the row order, and kTraversals[direction][1] is an array of
// size four refering to the column order.
constexpr std::array<std::array<std::array<int, 4>, 2>, 4> kTraversals = {{
    {{{0, 1, 2, 3}, {0, 1, 2, 3}}},  // Up
    {{{0, 1, 2, 3}, {3, 2, 1, 0}}},  // Right
    {{{3, 2, 1, 0}, {0, 1, 2, 3}}},  // Down
    {{{0, 1, 2, 3}, {0, 1, 2, 3}}}   // Left
}};
}  // namespace

TwentyFortyEightState::TwentyFortyEightState(std::shared_ptr<const Game> game)
    : State(game),
      parent_game_(open_spiel::down_cast<const TwentyFortyEightGame&>(*game)),
      board_(std::vector<Tile>(kRows * kColumns)) {}

void TwentyFortyEightState::SetCustomBoard(const std::vector<int>& board_seq) {
  current_player_ = 0;
  for (int r = 0; r < kRows; r++) {
    for (int c = 0; c < kColumns; c++) {
      SetBoard(r, c, Tile(board_seq[r * kColumns + c], false));
    }
  }
}

ChanceAction TwentyFortyEightState::SpielActionToChanceAction(
    Action action) const {
  std::vector<int> values =
      UnrankActionMixedBase(action, {kRows, kColumns, kChanceTiles.size()});
  return ChanceAction(values[0], values[1], values[2]);
}

Action TwentyFortyEightState::ChanceActionToSpielAction(
    ChanceAction move) const {
  std::vector<int> action_bases = {kRows, kColumns, kChanceTiles.size()};
  return RankActionMixedBase(action_bases,
                             {move.row, move.column, move.is_four});
}

bool TwentyFortyEightState::CellAvailable(int r, int c) const {
  return BoardAt(r, c).value == 0;
}

constexpr Coordinate GetVector(int direction) {
  switch (direction) {
    case kMoveUp:
      return Coordinate(-1, 0);
    case kMoveRight:
      return Coordinate(0, 1);
    case kMoveDown:
      return Coordinate(1, 0);
    case kMoveLeft:
      return Coordinate(0, -1);
    default:
      SpielFatalError("Unrecognized direction");
  }
}

std::array<Coordinate, 2> TwentyFortyEightState::FindFarthestPosition(
    int r, int c, int direction) const {
  // Progress towards the vector direction until an obstacle is found
  Coordinate prev = Coordinate(r, c);
  Coordinate direction_diff = GetVector(direction);
  do {
    prev = Coordinate(r, c);
    r += direction_diff.row;
    c += direction_diff.column;
  } while (InBounds(r, c) && CellAvailable(r, c));
  return std::array<Coordinate, 2>{prev, Coordinate(r, c)};
}

bool TwentyFortyEightState::TileMatchAvailable(int r, int c) const {
  int tile = BoardAt(r, c).value;
  if (tile > 0) {
    for (int direction : kPlayerActions) {
      Coordinate vector = GetVector(direction);
      int other = GetCellContent(r + vector.row, c + vector.column);
      if (other > 0 && other == tile) {
        return true;  // These two tiles can be merged
      }
    }
  }
  return false;
}

// Check for available matches between tiles (more expensive check)
bool TwentyFortyEightState::TileMatchesAvailable() const {
  for (int r = 0; r < kRows; r++) {
    for (int c = 0; c < kColumns; c++) {
      if (TileMatchAvailable(r, c)) {
        return true;
      }
    }
  }
  return false;
}

void TwentyFortyEightState::PrepareTiles() {
  for (int r = 0; r < kRows; r++) {
    for (int c = 0; c < kColumns; c++) {
      SetTileIsMerged(r, c, false);
    }
  }
}

int TwentyFortyEightState::GetCellContent(int r, int c) const {
  if (!InBounds(r, c)) return 0;
  return BoardAt(r, c).value;
}

void TwentyFortyEightState::DoApplyAction(Action action) {
  if (IsChanceNode()) {
    // The original 2048 game starts with two random tiles. To achieve this,
    // an extra move is given to the chance player during the beginning of the
    // game.
    if (!extra_chance_turn_) {
      current_player_ = 0;
    }
    extra_chance_turn_ = false;

    if (action == kNoCellAvailableAction) {
      return;
    }
    ChanceAction chance_action = SpielActionToChanceAction(action);
    SetBoard(
        chance_action.row, chance_action.column,
        Tile(chance_action.is_four ? kChanceTiles[1] : kChanceTiles[0], false));
    return;
  }
  action_score_ = 0;
  const std::array<std::array<int, 4>, 2>& traversals = kTraversals[action];
  PrepareTiles();
  for (int r : traversals[0]) {
    for (int c : traversals[1]) {
      int tile = GetCellContent(r, c);
      if (tile > 0) {
        bool moved = false;
        std::array<Coordinate, 2> positions =
            FindFarthestPosition(r, c, action);
        Coordinate farthest_pos = positions[0];
        Coordinate next_pos = positions[1];
        int next_cell = GetCellContent(next_pos.row, next_pos.column);
        if (next_cell > 0 && next_cell == tile &&
            !BoardAt(next_pos).is_merged) {
          int merged = tile * 2;
          action_score_ += merged;
          SetBoard(next_pos.row, next_pos.column, Tile(merged, true));
          moved = true;
        } else if (farthest_pos.row != r || farthest_pos.column != c) {
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
  total_actions_++;
}

bool TwentyFortyEightState::DoesActionChangeBoard(Action action) const {
  const std::array<std::array<int, 4>, 2>& traversals = kTraversals[action];
  for (int r : traversals[0]) {
    for (int c : traversals[1]) {
      int tile = GetCellContent(r, c);
      if (tile > 0) {
        std::array<Coordinate, 2> positions =
            FindFarthestPosition(r, c, action);
        Coordinate farthest_pos = positions[0];
        Coordinate next_pos = positions[1];
        int next_cell = GetCellContent(next_pos.row, next_pos.column);
        if (next_cell > 0 && next_cell == tile &&
            !BoardAt(next_pos).is_merged) {
          return true;
        } else if (farthest_pos.row != r || farthest_pos.column != c) {
          return true;
        }
      }
    }
  }
  return false;
}

std::string TwentyFortyEightState::ActionToString(Player player,
                                                  Action action_id) const {
  if (player == kChancePlayerId) {
    if (action_id == kNoCellAvailableAction) {
      return "No Cell Available";
    }
    ChanceAction chance_action = SpielActionToChanceAction(action_id);
    return absl::StrCat(chance_action.is_four ? 4 : 2, " added to row ",
                        chance_action.row + 1, ", column ",
                        chance_action.column + 1);
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
        // 2 appearing randomly on the board should be 9 times as likely as a 4.
        action_and_probs.emplace_back(
            ChanceActionToSpielAction(ChanceAction(r, c, false)), .9 / count);
        action_and_probs.emplace_back(
            ChanceActionToSpielAction(ChanceAction(r, c, true)), .1 / count);
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

  // Construct a vector from the array.
  std::vector<Action> actions =
      std::vector<Action>(kPlayerActions.begin(), kPlayerActions.end());

  std::vector<Action> actions_allowed = {};
  for (Action action : actions) {
    if (DoesActionChangeBoard(action)) actions_allowed.push_back(action);
  }
  return actions_allowed;
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
  if (move_number_ >= parent_game_.MaxGameLength()) {
    return true;
  }

  // Scan the board.
  int count = 0;
  int tile_matches_available = 0;
  for (int r = 0; r < kRows; r++) {
    for (int c = 0; c < kColumns; c++) {
      // Check for 2048, if necessary,
      if (BoardAt(r, c).value == parent_game_.max_tile()) {
        return true;
      }

      // Check for increase of available cell count.
      if (BoardAt(r, c).value == 0) {
        count++;
      }

      // Check for tile matches.
      if (TileMatchAvailable(r, c)) {
        tile_matches_available++;
      }
    }
  }

  if (count == 0 && tile_matches_available == 0) {
    return true;
  } else {
    return false;
  }
}

std::vector<double> TwentyFortyEightState::Rewards() const {
  return {static_cast<double>(action_score_)};
}

std::vector<double> TwentyFortyEightState::Returns() const {
  return {static_cast<double>(total_score_)};
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

TwentyFortyEightGame::TwentyFortyEightGame(const GameParameters& params)
    : Game(kGameType, params),
      max_tile_(ParameterValue<int>("max_tile", kDefaultMaxTile)) {}

int TwentyFortyEightGame::NumDistinctActions() const {
  return kPlayerActions.size();
}

}  // namespace twenty_forty_eight
}  // namespace open_spiel
