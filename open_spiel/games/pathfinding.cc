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

#include "open_spiel/games/pathfinding.h"

#include <algorithm>
#include <cctype>
#include <map>
#include <memory>
#include <utility>

#include "open_spiel/abseil-cpp/absl/container/flat_hash_map.h"
#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/game_parameters.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_globals.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/utils/combinatorics.h"
#include "open_spiel/utils/tensor_view.h"

namespace open_spiel {
namespace pathfinding {
namespace {

// Offsets for the actions: stay, left, up, right, down.
constexpr std::array<int, kNumActions> kRowOffsets = {0, 0, -1, 0, 1};
constexpr std::array<int, kNumActions> kColOffsets = {0, -1, 0, 1, 0};

// Register with general sum, since the game is not guaranteed to be zero sum.
// If we create a zero sum instance, the type on the created game will show it.
const GameType kGameType{
    /*short_name=*/"pathfinding",
    /*long_name=*/"Pathfinding",
    GameType::Dynamics::kSimultaneous,
    GameType::ChanceMode::kExplicitStochastic,
    GameType::Information::kPerfectInformation,
    GameType::Utility::kGeneralSum,
    GameType::RewardModel::kRewards,
    /*max_num_players=*/10,
    /*min_num_players=*/1,
    /*provides_information_state_string=*/false,
    /*provides_information_state_tensor=*/false,
    /*provides_observation_string=*/true,
    /*provides_observation_tensor=*/true,
    /*parameter_specification=*/
    {{"horizon", GameParameter(kDefaultHorizon)},
     {"grid", GameParameter(std::string(kDefaultSingleAgentGrid))},
     {"group_reward", GameParameter(kDefaultGroupReward)},
     {"players", GameParameter(kDefaultNumPlayers)},
     {"solve_reward", GameParameter(kDefaultSolveReward)},
     {"step_reward", GameParameter(kDefaultStepReward)}}};

std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::shared_ptr<const Game>(new PathfindingGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);

GridSpec ParseGrid(const std::string& grid_string, int max_num_players) {
  GridSpec grid{/*num_rows=*/0, /*num_cols=*/0};
  int row = 0;
  int col = 0;
  int count_empty_cells = 0;
  absl::flat_hash_map<Player, std::pair<int, int>> starting_positions_map;
  absl::flat_hash_map<Player, std::pair<int, int>> destinations_map;

  for (auto c : grid_string) {
    if (c == '\n') {
      row += 1;
      col = 0;
    } else {
      if (row >= grid.num_rows) grid.num_rows = row + 1;
      if (col >= grid.num_cols) grid.num_cols = col + 1;
      if (c == '*') {
        grid.obstacles.emplace_back(row, col);
      } else if (islower(c)) {
        // 97 is the ASCII code for 'a'.
        Player player = static_cast<int>(c) - 97;
        SPIEL_CHECK_GE(player, 0);
        SPIEL_CHECK_LT(player, max_num_players);
        starting_positions_map[player] = {row, col};
      } else if (isupper(c)) {
        // 65 is the ASCII code for 'A'.
        Player player = static_cast<int>(c) - 65;
        SPIEL_CHECK_GE(player, 0);
        SPIEL_CHECK_LT(player, max_num_players);
        destinations_map[player] = {row, col};
      } else if (c == '.') {
        ++count_empty_cells;
      } else {
        SpielFatalError(absl::StrCat("Invalid char '", std::string(1, c),
                                     "' at grid (", row, ",", col, ")"));
      }
      col += 1;
    }
  }

  grid.num_players = starting_positions_map.size();
  SPIEL_CHECK_EQ(starting_positions_map.size(), destinations_map.size());
  SPIEL_CHECK_GE(grid.num_players, 1);
  SPIEL_CHECK_LE(grid.num_players, max_num_players);

  // Move map entries to vectors.
  grid.starting_positions.resize(grid.num_players);
  grid.destinations.resize(grid.num_players);
  for (Player p = 0; p < grid.num_players; ++p) {
    // Check that we found a starting position, and move it to the vector.
    const auto iter1 = starting_positions_map.find(p);
    SPIEL_CHECK_TRUE(iter1 != starting_positions_map.end());
    grid.starting_positions[p] = iter1->second;
    // Check that we found a destination, and move it to the vector.
    const auto iter2 = destinations_map.find(p);
    SPIEL_CHECK_TRUE(iter2 != destinations_map.end());
    grid.destinations[p] = iter2->second;
  }
  return grid;
}

}  // namespace

PathfindingState::PathfindingState(std::shared_ptr<const Game> game,
                                   const GridSpec& grid_spec, int horizon)
    : SimMoveState(game),
      parent_game_(down_cast<const PathfindingGame&>(*game)),
      grid_spec_(grid_spec),
      cur_player_(kSimultaneousPlayerId),
      total_moves_(0),
      horizon_(horizon),
      player_positions_(num_players_),
      actions_(num_players_, kInvalidAction),
      rewards_(num_players_, 0.0),
      returns_(num_players_, 0.0),
      contested_players_(num_players_, 0),
      reached_destinations_(num_players_, 0) {
  grid_.reserve(grid_spec_.num_rows);
  for (int r = 0; r < grid_spec_.num_rows; ++r) {
    grid_.push_back(std::vector<int>(grid_spec_.num_cols, kEmpty));
  }

  for (const std::pair<int, int>& c : grid_spec_.obstacles) {
    grid_[c.first][c.second] = kWall;
  }

  SPIEL_CHECK_EQ(grid_spec_.starting_positions.size(), num_players_);
  for (Player p = 0; p < num_players_; ++p) {
    const std::pair<int, int>& c = grid_spec_.starting_positions[p];
    SPIEL_CHECK_EQ(grid_[c.first][c.second], kEmpty);
    grid_[c.first][c.second] = p;
    player_positions_[p] = c;
  }
}

std::string PathfindingState::ActionToString(int player,
                                             Action action_id) const {
  return parent_game_.ActionToString(player, action_id);
}

void PathfindingState::DoApplyActions(const std::vector<Action>& moves) {
  SPIEL_CHECK_EQ(moves.size(), num_players_);
  SPIEL_CHECK_EQ(cur_player_, kSimultaneousPlayerId);

  std::fill(rewards_.begin(), rewards_.end(), 0.0);
  std::fill(contested_players_.begin(), contested_players_.end(), 0);

  actions_ = moves;
  if (num_players_ == 1) {
    ResolvePlayerAction(0);
  } else {
    ResolveActions();
  }

  if (cur_player_ == kSimultaneousPlayerId) {
    // Only increment total moves if actions fully resolved.
    total_moves_++;
  }

  // If all players are at their destinations.
  if (AllPlayersOnDestinations()) {
    // Terminal state reached, all players get a bonus.
    for (Player p = 0; p < num_players_; ++p) {
      rewards_[p] += parent_game_.group_reward();
      returns_[p] += parent_game_.group_reward();
    }
  }
}

bool PathfindingState::InBounds(int r, int c) const {
  return (r >= 0 && c >= 0 && r < grid_spec_.num_rows &&
          c < grid_spec_.num_cols);
}

std::pair<int, int> PathfindingState::GetNextCoord(Player p) const {
  int row = player_positions_[p].first + kRowOffsets[actions_[p]];
  int col = player_positions_[p].second + kColOffsets[actions_[p]];
  if (!InBounds(row, col) || grid_[row][col] == kWall) {
    // Can't run out of bounds or into a wall.
    return player_positions_[p];
  }
  return {row, col};
}

void PathfindingState::ResolvePlayerAction(Player p) {
  const std::pair<int, int>& cur_coord = player_positions_[p];
  std::pair<int, int> next_coord = GetNextCoord(p);

  // Check if there is a player there. If so, change next_coord to cur_coord.
  Player other_player = PlayerAt(next_coord);
  if (other_player != kInvalidPlayer && other_player != p) {
    next_coord = cur_coord;
  }

  // Distribute rewards.
  if (next_coord != cur_coord && reached_destinations_[p] == 0 &&
      next_coord == grid_spec_.destinations[p]) {
    // Player is just getting to the destination for the first time!
    rewards_[p] += parent_game_.solve_reward();
    returns_[p] += parent_game_.solve_reward();
    reached_destinations_[p] = 1;
  } else if (next_coord == grid_spec_.destinations[p]) {
    // Player getting to destination again, or staying there: no penalty.
  } else {
    rewards_[p] += parent_game_.step_reward();
    returns_[p] += parent_game_.step_reward();
  }

  grid_[cur_coord.first][cur_coord.second] = kEmpty;
  grid_[next_coord.first][next_coord.second] = p;
  player_positions_[p] = next_coord;
}

Player PathfindingState::PlayerAt(const std::pair<int, int>& coord) const {
  int cell_state = grid_[coord.first][coord.second];
  if (cell_state >= 0 && cell_state < num_players_) {
    return cell_state;
  } else {
    return kInvalidPlayer;
  }
}

int PathfindingState::TryResolveContested() {
  int num_resolutions = 0;
  for (Player p = 0; p < num_players_; ++p) {
    if (contested_players_[p] == 1) {
      std::pair<int, int> next_coord = GetNextCoord(p);
      // A contested player can be resolved iff:
      //   - There is no other player on the next coord, and
      //   - No other (contested) player is planning to go there.
      Player other_player = PlayerAt(next_coord);
      if (other_player == kInvalidPlayer) {
        bool conflict = false;
        for (Player op = 0; op < num_players_; ++op) {
          if (p == op) {
            continue;
          }
          if (contested_players_[op] == 1) {
            std::pair<int, int> op_next_coord = GetNextCoord(op);
            if (next_coord == op_next_coord) {
              conflict = true;
              break;
            }
          }
        }

        if (!conflict) {
          contested_players_[p] = 0;
          num_resolutions++;
          ResolvePlayerAction(p);
        }
      }
    }
  }

  return num_resolutions;
}

void PathfindingState::ResolveActions() {
  // Get the next coords, and check for potentially conflicting actions.
  std::vector<std::pair<int, int>> next_coords;
  next_coords.reserve(num_players_);
  for (Player p = 0; p < num_players_; ++p) {
    std::pair<int, int> next_coord = GetNextCoord(p);
    // If there is a different player there, mark as potentially contested.
    // If another player is going there, mark both players as contested.
    Player other_player = PlayerAt(next_coord);
    if (other_player != kInvalidPlayer && other_player != p) {
      // Different player already there. Potentially contested (other player
      // may move out).
      contested_players_[p] = 1;
    } else if (actions_[p] == kStay) {
      // Stay action is never contested.
    } else {
      // Check if another player planning to go there.
      auto iter = std::find(next_coords.begin(), next_coords.end(), next_coord);
      if (iter != next_coords.end()) {
        Player other_player = iter - next_coords.begin();
        contested_players_[p] = 1;
        contested_players_[other_player] = 1;
      }
    }

    next_coords.push_back(next_coord);
  }

  // Check for head-on collisions. These should not be marked as contested,
  // because they result in a no-op.
  for (Player p = 0; p < num_players_; ++p) {
    if (contested_players_[p] == 1) {
      int op = PlayerAt(next_coords[p]);
      if (op != kInvalidPlayer && p != op) {
        Player opp = PlayerAt(next_coords[op]);
        if (opp != kInvalidPlayer && opp == p) {
          contested_players_[p] = 0;
          contested_players_[op] = 0;
          continue;
        }
      }
    }
  }

  // Move the uncontested, and repeatedly check the contested players to see if
  // moving resolves the contestations. If so, move them and mark as
  // uncontested. Stop when there is a pass with no moves.
  int num_contested = 0;
  for (Player p = 0; p < num_players_; ++p) {
    if (contested_players_[p] == 1) {
      num_contested++;
    } else {
      ResolvePlayerAction(p);
    }
  }

  int num_resolved = 0;
  do {
    num_resolved = TryResolveContested();
    num_contested -= num_resolved;
  } while (num_resolved > 0);

  // If there remain contestations, must resolve them via a chance node, which
  // will determine order of resolution.
  if (num_contested > 0) {
    cur_player_ = kChancePlayerId;
  }
}

void PathfindingState::DoApplyAction(Action action_id) {
  if (IsSimultaneousNode()) {
    ApplyFlatJointAction(action_id);
    return;
  } else {
    SPIEL_CHECK_TRUE(IsChanceNode());
    int num_contested_players =
        std::count_if(contested_players_.begin(), contested_players_.end(),
                      [](int i) { return i == 1; });
    std::vector<Player> contested_player_ids;
    contested_player_ids.reserve(num_contested_players);
    for (Player p = 0; p < contested_players_.size(); ++p) {
      if (contested_players_[p] == 1) {
        contested_player_ids.push_back(p);
      }
    }
    SPIEL_CHECK_EQ(contested_player_ids.size(), num_contested_players);
    std::vector<int> indices(num_contested_players);
    std::iota(indices.begin(), indices.end(), 0);
    std::vector<int> resolution_order = UnrankPermutation(indices, action_id);
    for (int idx : resolution_order) {
      ResolvePlayerAction(contested_player_ids[idx]);
    }
    std::fill(contested_players_.begin(), contested_players_.end(), 0);
    cur_player_ = kSimultaneousPlayerId;
    total_moves_++;
  }
}

std::vector<Action> PathfindingState::LegalActions(int player) const {
  if (IsTerminal()) return {};
  if (IsChanceNode()) {
    return LegalChanceOutcomes();
  } else {
    return parent_game_.legal_actions();
  }
}

std::vector<std::pair<Action, double>> PathfindingState::ChanceOutcomes()
    const {
  SPIEL_CHECK_TRUE(IsChanceNode());
  int num_contested_players =
      std::count_if(contested_players_.begin(), contested_players_.end(),
                    [](int i) { return i == 1; });
  int num_permutations = Factorial(num_contested_players);
  double prob = 1.0 / num_permutations;
  ActionsAndProbs outcomes;
  outcomes.reserve(num_permutations);
  for (int i = 0; i < num_permutations; ++i) {
    outcomes.push_back({i, prob});
  }
  return outcomes;
}

Player PathfindingState::PlayerAtPos(const std::pair<int, int>& coord) const {
  if (grid_[coord.first][coord.second] >= 0 &&
      grid_[coord.first][coord.second] < num_players_) {
    return grid_[coord.first][coord.second];
  } else {
    return kInvalidPlayer;
  }
}

std::string PathfindingState::ToString() const {
  std::string str;
  for (int r = 0; r < grid_spec_.num_rows; ++r) {
    for (int c = 0; c < grid_spec_.num_cols; ++c) {
      if (grid_[r][c] >= 0 && grid_[r][c] < num_players_) {
        absl::StrAppend(&str, grid_[r][c]);
      } else if (grid_[r][c] == kWall) {
        absl::StrAppend(&str, "*");
      } else {
        absl::StrAppend(&str, ".");
      }
    }
    absl::StrAppend(&str, "\n");
  }
  return str;
}

int PathfindingState::PlayerPlaneIndex(int observing_player,
                                       int actual_player) const {
  // Need to add a num_players_ inside the brackets here because of how C++
  // handles mod of negative values.
  return (actual_player - observing_player + num_players_) % num_players_;
}

// Note: currently, the observations are current non-Markovian because the time
// step is not included and the horizon is finite.
std::string PathfindingState::ObservationString(int player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  return ToString();
}

// Note: currently, the observations are current non-Markovian because the time
// step is not included and the horizon is finite.
void PathfindingState::ObservationTensor(int player,
                                         absl::Span<float> values) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  std::fill(values.begin(), values.end(), 0.0);

  TensorView<3> view(values,
                     {parent_game_.NumObservationPlanes(), grid_spec_.num_rows,
                      grid_spec_.num_cols},
                     true);

  // Let n be the number of players.
  //   - First n planes refer to player
  //   - Second n planes refer to player's starting position
  //   - Third n planes refer to player's destination position
  //   - 1 plane for wall
  //   - 1 plane for empty
  //
  // The first three sets of n planes corresponding to the players are each
  // ordered ego-centrically:
  //   - the first plane is the observing player's plane, followed by the next
  //     player, followed by the next etc. so in a 4-player game, if player 2
  //     is the observing player, the planes would be ordered by player 2, 3, 0,
  //     1.
  for (int r = 0; r < grid_spec_.num_rows; ++r) {
    for (int c = 0; c < grid_spec_.num_cols; ++c) {
      // Player on the position.
      if (grid_[r][c] >= 0 && grid_[r][c] < num_players_) {
        view[{PlayerPlaneIndex(player, grid_[r][c]), r, c}] = 1.0;
      }

      // Wall
      if (grid_[r][c] == kWall) {
        view[{3 * num_players_, r, c}] = 1.0;
      }

      // Empty
      if (grid_[r][c] == kEmpty) {
        view[{3 * num_players_ + 1, r, c}] = 1.0;
      }
    }
  }

  for (Player p = 0; p < num_players_; ++p) {
    const std::pair<int, int>& start_pos = grid_spec_.starting_positions[p];
    const std::pair<int, int>& dest_pos = grid_spec_.destinations[p];
    int pidx = PlayerPlaneIndex(player, p);
    view[{num_players_ + pidx, start_pos.first, start_pos.second}] = 1.0;
    view[{2 * num_players_ + pidx, dest_pos.first, dest_pos.second}] = 1.0;
  }
}

bool PathfindingState::AllPlayersOnDestinations() const {
  for (Player p = 0; p < num_players_; ++p) {
    const std::pair<int, int>& c = grid_spec_.destinations[p];
    if (grid_[c.first][c.second] != p) {
      return false;
    }
  }
  return true;
}

bool PathfindingState::IsTerminal() const {
  if (total_moves_ >= horizon_) {
    return true;
  }

  // Check if all players at their destinations.
  return AllPlayersOnDestinations();
}

std::vector<double> PathfindingState::Rewards() const { return rewards_; }

std::vector<double> PathfindingState::Returns() const { return returns_; }

std::unique_ptr<State> PathfindingState::Clone() const {
  return std::unique_ptr<State>(new PathfindingState(*this));
}

std::unique_ptr<State> PathfindingGame::NewInitialState() const {
  return std::unique_ptr<PathfindingState>(
      new PathfindingState(shared_from_this(), grid_spec_, horizon_));
}

int PathfindingGame::MaxChanceOutcomes() const {
  return Factorial(NumPlayers());
}

double PathfindingGame::MinUtility() const {
  // Add a small constant here due to numeral issues.
  return horizon_ * step_reward_ - FloatingPointDefaultThresholdRatio();
}

double PathfindingGame::MaxUtility() const {
  return solve_reward_ + group_reward_;
}

int PathfindingGame::NumObservationPlanes() const {
  // Number of position planes:
  // - one per player present on the pos
  // - one per player (starting position)
  // - one per player (destination)
  // - one for empty positions
  // - one for wall positions
  return 3 * grid_spec_.num_players + 2;
}

std::vector<int> PathfindingGame::ObservationTensorShape() const {
  return {NumObservationPlanes(), grid_spec_.num_rows, grid_spec_.num_cols};
}

std::string PathfindingGame::ActionToString(int player,
                                            Action action_id) const {
  if (player == kChancePlayerId) {
    return absl::StrCat("Chance outcome ", action_id);
  }

  switch (action_id) {
    case kStay:
      return "Stay";
    case kLeft:
      return "Left";
    case kUp:
      return "Up";
    case kRight:
      return "Right";
    case kDown:
      return "Down";
    default:
      SpielFatalError(absl::StrCat("Unknown action: ", action_id));
  }
}

int PathfindingGame::NumPlayers() const { return num_players_; }

PathfindingGame::PathfindingGame(const GameParameters& params)
    : SimMoveGame(kGameType, params),
      grid_spec_(ParseGrid(ParameterValue<std::string>(
          "grid", std::string(kDefaultSingleAgentGrid)),
                           kGameType.max_num_players)),
      num_players_(ParameterValue<int>("players", kDefaultNumPlayers)),
      horizon_(ParameterValue<int>("horizon", kDefaultHorizon)),
      group_reward_(ParameterValue<double>("group_reward",
                                           kDefaultGroupReward)),
      solve_reward_(
          ParameterValue<double>("solve_reward", kDefaultSolveReward)),
      step_reward_(ParameterValue<double>("step_reward", kDefaultStepReward)),
      legal_actions_({kStay, kLeft, kUp, kRight, kDown}) {
  // Override the number of players from the grid specification.
  //
  // Currently, the game only supports specific grids, so this will always be
  // overridden. This will change in a future version with random grids.
  if (grid_spec_.num_players >= 1) {
    num_players_ = grid_spec_.num_players;
  }
}

}  // namespace pathfinding
}  // namespace open_spiel
