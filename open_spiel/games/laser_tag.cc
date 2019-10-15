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

#include "open_spiel/games/laser_tag.h"

#include <map>
#include <memory>
#include <utility>

#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace laser_tag {

namespace {

// Default parameters.
constexpr int kDefaultHorizon = 1000;
constexpr bool kDefaultZeroSum = false;

// Register with general sum, since the game is not guaranteed to be zero sum.
// If we create a zero sum instance, the type on the created game will show it.
const GameType kGameTypeGeneralSum{
    /*short_name=*/"laser_tag",
    /*long_name=*/"Laser Tag",
    GameType::Dynamics::kSimultaneous,
    GameType::ChanceMode::kDeterministic,
    GameType::Information::kPerfectInformation,
    GameType::Utility::kGeneralSum,
    GameType::RewardModel::kRewards,
    /*max_num_players=*/2,
    /*min_num_players=*/2,
    /*provides_information_state=*/false,
    /*provides_information_state_as_normalized_vector=*/false,
    /*provides_observation=*/true,
    /*provides_observation_as_normalized_vector=*/true,
    /*parameter_specification=*/
    {{"horizon", GameParameter(kDefaultHorizon)},
     {"zero_sum", GameParameter(kDefaultZeroSum)},
     {"grid", GameParameter(std::string(kDefaultGrid))}}};

GameType GameTypeForParams(const GameParameters& params) {
  auto game_type = kGameTypeGeneralSum;
  bool is_zero_sum = kDefaultZeroSum;
  auto it = params.find("zero_sum");
  if (it != params.end()) is_zero_sum = it->second.bool_value();
  if (is_zero_sum) game_type.utility = GameType::Utility::kZeroSum;
  return game_type;
}

std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::shared_ptr<const Game>(new LaserTagGame(params));
}

REGISTER_SPIEL_GAME(kGameTypeGeneralSum, Factory);

// Valid characters: AB*.
constexpr int kCellStates = 4;

// Movement.
enum MovementType {
  kLeftTurn = 0,
  kRightTurn = 1,
  kForwardMove = 2,
  kBackwardMove = 3,
  kStepLeft = 4,
  kStepRight = 5,
  kStand = 6,
  kForwardLeft = 7,
  kForwardRight = 8,
  kFire = 9
};

// Orientation
enum Orientation { kNorth = 0, kSouth = 1, kEast = 2, kWest = 3 };

// mapping of start and end orientations for left and right turn
std::map<int, int> leftMapping = {{0, 3}, {1, 2}, {2, 0}, {3, 1}};
std::map<int, int> rightMapping = {{0, 2}, {1, 3}, {2, 1}, {3, 0}};

// Chance outcomes.
enum ChanceOutcome {
  kChanceLoc1 = 0,
  kChanceLoc2 = 1,
  kChanceLoc3 = 2,
  kChanceLoc4 = 3,
  kChanceInit1 = 4,
  kChanceInit2 = 5
};

// four directions: N,S,E,W
constexpr std::array<std::array<int, 10>, 4> row_offsets = {
    {{0, 0, -1, 1, 0, 0, 0, -1, -1, 0},
     {0, 0, 1, -1, 0, 0, 0, 1, 1, 0},
     {0, 0, 0, 0, -1, 1, 0, 0, 0, 0},
     {0, 0, 0, 0, 1, -1, 0, 0, 0, 0}}};
constexpr std::array<std::array<int, 10>, 4> col_offsets = {
    {{0, 0, 0, 0, -1, 1, 0, 0, 0, 0},
     {0, 0, 0, 0, 1, -1, 0, 0, 0, 0},
     {0, 0, 1, -1, 0, 0, 0, 1, 1, 0},
     {0, 0, -1, 1, 0, 0, 0, -1, -1, 0}}};
}  // namespace

LaserTagState::LaserTagState(std::shared_ptr<const Game> game, const Grid& grid)
    : SimMoveState(game), grid_(grid) {}

std::string LaserTagState::ActionToString(int player, Action action_id) const {
  if (player == kSimultaneousPlayerId)
    return FlatJointActionToString(action_id);
  SPIEL_CHECK_GE(action_id, 0);
  SPIEL_CHECK_LT(action_id, 10);

  std::string result = "";
  if (player == kChancePlayerId) {
    // Chance moves.
    if (action_id == kChanceLoc1) {
      result = "(spawned at spawn location #1)";
    } else if (action_id == kChanceLoc2) {
      result = "(spawned at spawn location #2)";
    } else if (action_id == kChanceLoc3) {
      result = "(spawned at spawn location #3)";
    } else if (action_id == kChanceLoc4) {
      result = "(spawned at spawn location #4)";
    } else if (action_id == kChanceInit1) {
      result = "(A's action first)";
    } else if (action_id == kChanceInit2) {
      result = "(B's action first)";
    }
  } else {
    if (action_id == kLeftTurn) {
      result = "left turn";
    } else if (action_id == kRightTurn) {
      result = "right turn";
    } else if (action_id == kForwardMove) {
      result = "move forward";
    } else if (action_id == kBackwardMove) {
      result = "move backward";
    } else if (action_id == kStepLeft) {
      result = "step left";
    } else if (action_id == kStepRight) {
      result = "step right";
    } else if (action_id == kStand) {
      result = "stand";
    } else if (action_id == kForwardLeft) {
      result = "step forward and left turn";
    } else if (action_id == kForwardRight) {
      result = "step forward and right turn";
    } else if (action_id == kFire) {
      result = "fire";
    }
  }
  return result;
}

void LaserTagState::SetField(int r, int c, char v) {
  field_[r * grid_.num_cols + c] = v;

  if (v == 'A') {
    player_row_[0] = r;
    player_col_[0] = c;
  } else if (v == 'B') {
    player_row_[1] = r;
    player_col_[1] = c;
  }
}

char LaserTagState::field(int r, int c) const {
  return field_[r * grid_.num_cols + c];
}

void LaserTagState::Reset(int horizon, bool zero_sum) {
  num_tags_ = 0;
  horizon_ = horizon;
  zero_sum_rewards_ = zero_sum;
  field_.resize(grid_.num_rows * grid_.num_cols, '.');

  for (auto i : grid_.obstacles) {
    SetField(i.first, i.second, '*');
  }

  cur_player_ = kChancePlayerId;
  total_moves_ = 0;
  needs_respawn_ = {0, 1};
  rewards_ = {0, 0};
  returns_ = {0, 0};
  player_facing_ = {{kSouth, kSouth}};
}

void LaserTagState::DoApplyActions(const std::vector<Action>& moves) {
  SPIEL_CHECK_EQ(moves.size(), 2);
  SPIEL_CHECK_EQ(cur_player_, kSimultaneousPlayerId);
  moves_[0] = moves[0];
  moves_[1] = moves[1];
  cur_player_ = kChancePlayerId;
}

bool LaserTagState::InBounds(int r, int c) const {
  return (r >= 0 && c >= 0 && r < grid_.num_rows && c < grid_.num_cols);
}

bool LaserTagState::ResolveMove(int player, int move) {
  int old_row = player_row_[player];
  int old_col = player_col_[player];

  int current_orientation = player_facing_[player];

  // move depends on player's current orientation
  int new_row = old_row + row_offsets[current_orientation][move];
  int new_col = old_col + col_offsets[current_orientation][move];

  if (!InBounds(new_row, new_col)) {  // move is out of bounds so do nothing
    return false;
  }

  char from_piece = field(old_row, old_col);

  if (move == kLeftTurn) {  // turn left
    player_facing_[player] = leftMapping.find(current_orientation)->second;
    return false;
  } else if (move == kRightTurn) {  // turn right
    player_facing_[player] = rightMapping.find(current_orientation)->second;
    return false;
  } else if (move == kForwardMove || move == kBackwardMove ||
             move == kStepLeft || move == kStepLeft || move == kForwardLeft ||
             move == kForwardRight) {  // move left or right or forward or
                                       // backward if able

    if (field(new_row, new_col) == '.') {
      SetField(old_row, old_col, '.');
      SetField(new_row, new_col, from_piece);

      // move and also turn
      if (move == kForwardLeft) {
        player_facing_[player] = leftMapping.find(current_orientation)->second;
      } else if (move == kForwardRight) {
        player_facing_[player] = rightMapping.find(current_orientation)->second;
      }
    }

    return false;
  } else if (move == kFire) {  // fire!
    int cur_row = old_row;
    int cur_col = new_col;
    int tagger = kInvalidPlayer;
    int got_tagged = kInvalidPlayer;

    // laser goes in direction agent is facing
    if (current_orientation == kNorth) {
      cur_row--;
    } else if (current_orientation == kSouth) {
      cur_row++;
    } else if (current_orientation == kEast) {
      cur_col++;
    } else if (current_orientation == kWest) {
      cur_col--;
    }

    while (InBounds(cur_row,
                    cur_col)) {  // shoot and track laser while it is in bounds
      char fired_upon = field(cur_row, cur_col);

      if (fired_upon == 'A') {  // A was hit!
        tagger = 1;
        got_tagged = 0;
        break;
      } else if (fired_upon == 'B') {  // B was hit!
        tagger = 0;
        got_tagged = 1;
        break;
      } else if (fired_upon == '*') {  // obstacle was hit so do nothing
        return false;
      }

      // laser goes in direction agent is facing
      if (current_orientation == kNorth) {
        cur_row--;
      } else if (current_orientation == kSouth) {
        cur_row++;
      } else if (current_orientation == kEast) {
        cur_col++;
      } else if (current_orientation == kWest) {
        cur_col--;
      }
    }

    // If there was a tag, set the rewards appropriately.
    if (tagger != kInvalidPlayer) {
      num_tags_++;
      needs_respawn_ = {got_tagged};
      SetField(player_row_[got_tagged], player_col_[got_tagged], '.');
      player_row_[got_tagged] = -1;
      player_col_[got_tagged] = -1;
    }

    if (tagger == 0 && zero_sum_rewards_) {
      rewards_[0] += 1;
      rewards_[1] -= 1;
    } else if (tagger == 0 && !zero_sum_rewards_) {
      rewards_[0] += 1;
    } else if (tagger == 1 && zero_sum_rewards_) {
      rewards_[0] -= 1;
      rewards_[1] += 1;
    } else if (tagger == 1 && !zero_sum_rewards_) {
      rewards_[1] += 1;
    }

    return tagger != kInvalidPlayer;
  }

  return false;
}

void LaserTagState::DoApplyAction(Action action_id) {
  if (IsSimultaneousNode()) {
    ApplyFlatJointAction(action_id);
    return;
  }
  SPIEL_CHECK_TRUE(IsChanceNode());
  SPIEL_CHECK_GE(action_id, 0);
  SPIEL_CHECK_LT(action_id, 6);

  char spawning_player_char = ' ';

  // spawn locations and move resolve order
  if (!needs_respawn_.empty()) {
    int spawning_player = needs_respawn_.back();
    spawning_player_char = spawning_player == 0 ? 'A' : 'B';
  }

  if (action_id == kChanceLoc1) {
    SPIEL_CHECK_NE(spawning_player_char, ' ');
    if (field(grid_.spawn_points[0].first, grid_.spawn_points[0].second) !=
        '.') {
      return;
    }
    SetField(grid_.spawn_points[0].first, grid_.spawn_points[0].second,
             spawning_player_char);
    needs_respawn_.pop_back();
  } else if (action_id == kChanceLoc2) {
    SPIEL_CHECK_NE(spawning_player_char, ' ');
    if (field(grid_.spawn_points[1].first, grid_.spawn_points[1].second) !=
        '.') {
      return;
    }
    SetField(grid_.spawn_points[1].first, grid_.spawn_points[1].second,
             spawning_player_char);
    needs_respawn_.pop_back();
  } else if (action_id == kChanceLoc3) {
    SPIEL_CHECK_NE(spawning_player_char, ' ');
    if (field(grid_.spawn_points[2].first, grid_.spawn_points[2].second) !=
        '.') {
      return;
    }
    SetField(grid_.spawn_points[2].first, grid_.spawn_points[2].second,
             spawning_player_char);
    needs_respawn_.pop_back();
  } else if (action_id == kChanceLoc4) {
    SPIEL_CHECK_NE(spawning_player_char, ' ');
    if (field(grid_.spawn_points[3].first, grid_.spawn_points[3].second) !=
        '.') {
      return;
    }
    SetField(grid_.spawn_points[3].first, grid_.spawn_points[3].second,
             spawning_player_char);
    needs_respawn_.pop_back();
  } else if (action_id == kChanceInit1) {
    rewards_ = {0, 0};
    bool tagged = ResolveMove(0, moves_[0]);
    if (!tagged) {
      ResolveMove(1, moves_[1]);
    }
    returns_[0] += rewards_[0];
    returns_[1] += rewards_[1];
    total_moves_++;
  } else if (action_id == kChanceInit2) {
    rewards_ = {0, 0};
    bool tagged = ResolveMove(1, moves_[1]);
    if (!tagged) {
      ResolveMove(0, moves_[0]);
    }
    returns_[0] += rewards_[0];
    returns_[1] += rewards_[1];
    total_moves_++;
  }

  if (needs_respawn_.empty()) {
    cur_player_ = kSimultaneousPlayerId;
  } else {
    cur_player_ = kChancePlayerId;
  }
}

std::vector<Action> LaserTagState::LegalActions(int player) const {
  if (IsTerminal()) return {};
  if (IsChanceNode()) {
    if (!needs_respawn_.empty()) {
      return {kChanceLoc1, kChanceLoc2, kChanceLoc3, kChanceLoc4};
    } else {
      return {kChanceInit1, kChanceInit2};
    }
  } else {
    return {kLeftTurn,  kRightTurn, kForwardMove, kBackwardMove, kStepLeft,
            kStepRight, kStand,     kForwardLeft, kForwardRight, kFire};
  }
}

std::vector<std::pair<Action, double>> LaserTagState::ChanceOutcomes() const {
  SPIEL_CHECK_TRUE(IsChanceNode());
  if (!needs_respawn_.empty()) {
    return {std::pair<Action, double>(kChanceLoc1, 0.25),
            std::pair<Action, double>(kChanceLoc2, 0.25),
            std::pair<Action, double>(kChanceLoc3, 0.25),
            std::pair<Action, double>(kChanceLoc4, 0.25)};
  } else {
    return {std::pair<Action, double>(kChanceInit1, 0.5),
            std::pair<Action, double>(kChanceInit2, 0.5)};
  }
}

std::string LaserTagState::ToString() const {
  std::string result = "";

  for (int r = 0; r < grid_.num_rows; r++) {
    for (int c = 0; c < grid_.num_cols; c++) {
      result += field(r, c);
    }

    absl::StrAppend(&result, "\n");
  }

  absl::StrAppend(&result, "Orientations: ", player_facing_[0], " ",
                  player_facing_[1], "\n");
  if (IsChanceNode()) absl::StrAppend(&result, "Chance Node");
  return result;
}

bool LaserTagState::IsTerminal() const {
  return ((horizon_ >= 0 && total_moves_ >= horizon_) ||
          (horizon_ < 0 && num_tags_ > 0));
}

std::vector<double> LaserTagState::Rewards() const { return rewards_; }

std::vector<double> LaserTagState::Returns() const { return returns_; }

int LaserTagState::observation_plane(int r, int c) const {
  int plane = -1;
  switch (field(r, c)) {
    case 'A':
      plane = 0;
      break;
    case 'B':
      plane = 1;
      break;
    case '.':
      plane = 2;
      break;
    case '*':
      plane = 3;
      break;
    default:
      std::cerr << "Invalid character on field: " << field(r, c) << std::endl;
      plane = -1;
      break;
  }

  return plane;
}

void LaserTagState::ObservationAsNormalizedVector(
    int player, std::vector<double>* values) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);

  values->resize(game_->ObservationNormalizedVectorSize());
  std::fill(values->begin(), values->end(), 0.);
  int plane_size = grid_.num_rows * grid_.num_cols;

  for (int r = 0; r < grid_.num_rows; r++) {
    for (int c = 0; c < grid_.num_cols; c++) {
      int plane = observation_plane(r, c);
      SPIEL_CHECK_TRUE(plane >= 0 && plane < kCellStates);
      (*values)[plane * plane_size + r * grid_.num_cols + c] = 1.0;
    }
  }
}

std::unique_ptr<State> LaserTagState::Clone() const {
  return std::unique_ptr<State>(new LaserTagState(*this));
}

std::unique_ptr<State> LaserTagGame::NewInitialState() const {
  std::unique_ptr<LaserTagState> state(
      new LaserTagState(shared_from_this(), grid_));
  state->Reset(horizon_, zero_sum_);
  return state;
}

double LaserTagGame::MinUtility() const {
  if (horizon_ < 0) {
    return -1;
  } else {
    return -horizon_;
  }
}

double LaserTagGame::MaxUtility() const {
  if (horizon_ < 0) {
    return 1;
  } else {
    return horizon_;
  }
}

std::vector<int> LaserTagGame::ObservationNormalizedVectorShape() const {
  return {kCellStates, grid_.num_rows, grid_.num_cols};
}

namespace {
Grid ParseGrid(const std::string& grid_string) {
  Grid grid{/*num_rows=*/0, /*num_cols=*/0};
  int row = 0;
  int col = 0;
  int count_empty_cells = 0;
  for (auto c : grid_string) {
    if (c == '\n') {
      row += 1;
      col = 0;
    } else {
      if (row >= grid.num_rows) grid.num_rows = row + 1;
      if (col >= grid.num_cols) grid.num_cols = col + 1;
      if (c == '*') {
        grid.obstacles.emplace_back(row, col);
      } else if (c == 'S') {
        grid.spawn_points.emplace_back(row, col);
      } else if (c == '.') {
        ++count_empty_cells;
      } else {
        SpielFatalError(absl::StrCat("Invalid char '", std::string(1, c),
                                     "' at grid (", row, ",", col, ")"));
      }
      col += 1;
    }
  }
  SPIEL_CHECK_EQ(4, grid.spawn_points.size());
  SPIEL_CHECK_EQ(
      grid.num_rows * grid.num_cols,
      count_empty_cells + grid.spawn_points.size() + grid.obstacles.size());
  return grid;
}
}  // namespace

LaserTagGame::LaserTagGame(const GameParameters& params)
    : SimMoveGame(GameTypeForParams(params), params),
      grid_(ParseGrid(ParameterValue<std::string>("grid"))),
      horizon_(ParameterValue<int>("horizon")),
      zero_sum_(ParameterValue<bool>("zero_sum")) {}

}  // namespace laser_tag
}  // namespace open_spiel
