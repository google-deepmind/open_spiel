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

#ifndef OPEN_SPIEL_GAMES_PATHFINDING_H_
#define OPEN_SPIEL_GAMES_PATHFINDING_H_

#include <array>
#include <memory>
#include <string>
#include <vector>

#include "open_spiel/simultaneous_move_game.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_globals.h"

namespace open_spiel {
namespace pathfinding {

// A simple simultaneous-move (single- and multi-agent) grid world pathfinding
// game.
//
// Grids can be expressed as ASCII strings, where lower case characters refer
// to starting positions, upper case characters refer to destinations,
// '.' refers to an empty cell, '*' refers to an wall.
//
// Parameters:
//   "grid"         int     The grid world the agents play in (default below).
//   "group_reward" double  Extra reward (to each agent) if all agents reach
//                          their desitnation (default: 100.0).
//   "horizon"      int     Maximum number of steps in an episode (def: 1000).
//   "players"      int     Number of players (default: 1, and overridden by
//                          the grid).
//   "solve_reward" double  Reward obtained when reaching the destination
//                          (default: 100.0).
//   "step_reward"  double  The reward given to every agent on each per step
//                          (default: -0.01).
//
// Note: currently, the observations are current non-Markovian because the time
// step is not included and the horizon is finite. This can be easily added as
// an option if desired.

inline constexpr char kDefaultSingleAgentGrid[] =
    "A.*..**\n"
    "..*....\n"
    "....*a.\n";

inline constexpr char kExampleMultiAgentGrid[] =
    "A.*Db**\n"
    "..*....\n"
    "..*.*a.\n"
    ".B*.**.\n"
    ".*..*..\n"
    "......c\n"
    "C..*..d";

// Default parameters.
constexpr int kDefaultHorizon = 1000;
constexpr int kDefaultNumPlayers = 1;
constexpr double kDefaultStepReward = -0.01;
constexpr double kDefaultSolveReward = 100.0;
constexpr double kDefaultGroupReward = 100.0;

struct GridSpec {
  int num_rows;
  int num_cols;
  int num_players = -1;
  std::vector<std::pair<int, int>> obstacles;
  std::vector<std::pair<int, int>> starting_positions;
  std::vector<std::pair<int, int>> destinations;
};

// Movement.
enum MovementType {
  kStay = 0,
  kLeft = 1,
  kUp = 2,
  kRight = 3,
  kDown = 4,
};

enum CellState { kEmpty = -1, kWall = -2 };

constexpr int kNumActions = 5;

class PathfindingGame : public SimMoveGame {
 public:
  explicit PathfindingGame(const GameParameters& params);
  int NumDistinctActions() const { return kNumActions; }
  std::string ActionToString(int player, Action action_id) const override;
  std::unique_ptr<State> NewInitialState() const override;
  int MaxChanceOutcomes() const override;
  int NumPlayers() const override;
  double MinUtility() const override;
  double MaxUtility() const override;
  double UtilitySum() const override { return 0; }
  std::vector<int> ObservationTensorShape() const override;
  int MaxGameLength() const override { return horizon_; }
  int MaxChanceNodesInHistory() const override { return MaxGameLength(); }

  int NumObservationPlanes() const;
  const std::vector<Action>& legal_actions() const { return legal_actions_; }
  double group_reward() const { return group_reward_; }
  double solve_reward() const { return solve_reward_; }
  double step_reward() const { return step_reward_; }

 private:
  GridSpec grid_spec_;
  int num_players_;
  int horizon_;
  double group_reward_;
  double solve_reward_;
  double step_reward_;
  std::vector<Action> legal_actions_;
};

class PathfindingState : public SimMoveState {
 public:
  explicit PathfindingState(std::shared_ptr<const Game> game,
                            const GridSpec& grid_spec, int horizon);
  PathfindingState(const PathfindingState&) = default;

  std::string ActionToString(int player, Action action_id) const override;
  std::string ToString() const override;
  bool IsTerminal() const override;
  std::vector<double> Rewards() const override;
  std::vector<double> Returns() const override;
  std::string ObservationString(int player) const override;
  void ObservationTensor(int player, absl::Span<float> values) const override;
  int CurrentPlayer() const override {
    return IsTerminal() ? kTerminalPlayerId : cur_player_;
  }
  std::unique_ptr<State> Clone() const override;

  ActionsAndProbs ChanceOutcomes() const override;

  std::vector<Action> LegalActions(int player) const override;

  std::pair<int, int> PlayerPos(int player) const {
    return player_positions_[player];
  }

  Player PlayerAtPos(const std::pair<int, int>& coord) const;

 protected:
  void DoApplyAction(Action action_id) override;
  void DoApplyActions(const std::vector<Action>& moves) override;

 private:
  std::pair<int, int> GetNextCoord(Player p) const;
  void ResolvePlayerAction(Player p);
  void ResolveActions();
  bool InBounds(int r, int c) const;
  Player PlayerAt(const std::pair<int, int>& coord) const;
  int TryResolveContested();
  bool AllPlayersOnDestinations() const;
  int PlayerPlaneIndex(int observing_player, int actual_player) const;

  const PathfindingGame& parent_game_;
  const GridSpec& grid_spec_;

  int cur_player_;
  int total_moves_;
  int horizon_;
  std::vector<std::pair<int, int>> player_positions_;

  // The state of the board. Coordinates indices are in row-major order.
  // - Values from 0 to num_players - 1 refer to the player.
  // - Otherwise the value is above (kEmpty or kWall).
  std::vector<std::vector<int>> grid_;

  // The player's chosen actions.
  std::vector<Action> actions_;

  // Rewards this turn and cumulative rewards.
  std::vector<double> rewards_;
  std::vector<double> returns_;

  // Used when conflicting actions need to be resolved.
  // 0 = uncontested, 1 = contested.
  std::vector<int> contested_players_;

  // Has the player reached the destination? (1 if yes, 0 if no).
  std::vector<int> reached_destinations_;
};

}  // namespace pathfinding
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_PATHFINDING_H_
