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

#ifndef OPEN_SPIEL_GAMES_COOP_BOX_PUSHING_SOCCER_H_
#define OPEN_SPIEL_GAMES_COOP_BOX_PUSHING_SOCCER_H_

#include <array>
#include <memory>
#include <string>
#include <vector>

#include "open_spiel/simultaneous_move_game.h"
#include "open_spiel/spiel.h"

// This is the cooperative box-pushing domain presented by Seuken & Zilberstein
// in their paper "Improved Memory-Bounded Dynamic Programming for Dec-POMDPs"
// http://rbr.cs.umass.edu/papers/SZuai07.pdf
//
// Parameters:
//     "fully_observable" bool   agents see everything, or only partial view as
//                               described in the original paper (def: false)
//     "horizon"          int    length of horizon (default = 100)

namespace open_spiel {
namespace coop_box_pushing {

// To indicate the status of each agent's action.
enum class ActionStatusType {
  kUnresolved,
  kSuccess,
  kFail,
};

// Direction each agent can be facing.
enum OrientationType {
  kNorth = 0,
  kEast = 1,
  kSouth = 2,
  kWest = 3,
  kInvalid = 4
};

// When not fully-observable, the number of observations (taken from Seuken &
// Zilberstein '12): empty field, wall, other agent, small box, large box.
enum ObservationType {
  kEmptyFieldObs,
  kWallObs,
  kOtherAgentObs,
  kSmallBoxObs,
  kBigBoxObs
};
constexpr int kNumObservations = 5;

// Different actions used by the agent.
enum class ActionType { kTurnLeft, kTurnRight, kMoveForward, kStay };

class CoopBoxPushingState : public SimMoveState {
 public:
  CoopBoxPushingState(std::shared_ptr<const Game> game, int horizon,
                      bool fully_observable);

  std::string ActionToString(Player player, Action action) const override;
  std::string ToString() const override;
  bool IsTerminal() const override;
  std::vector<double> Returns() const override;
  std::vector<double> Rewards() const override;
  void ObservationTensor(Player player,
                         absl::Span<float> values) const override;
  std::string ObservationString(Player player) const override;

  Player CurrentPlayer() const override {
    return IsTerminal() ? kTerminalPlayerId : cur_player_;
  }
  std::unique_ptr<State> Clone() const override;

  ActionsAndProbs ChanceOutcomes() const override;

  void Reset(const GameParameters& params);
  std::vector<Action> LegalActions(Player player) const override;

 protected:
  void DoApplyAction(Action action) override;
  void DoApplyActions(const std::vector<Action>& actions) override;

 private:
  void SetField(std::pair<int, int> coord, char v);
  void SetPlayer(std::pair<int, int> coord, Player player,
                 OrientationType orientation);
  void SetPlayer(std::pair<int, int> coord, Player player);
  void AddReward(double reward);
  char field(std::pair<int, int> coord) const;
  void ResolveMoves();
  void MoveForward(Player player);
  bool InBounds(std::pair<int, int> coord) const;
  bool SameAsPlayer(std::pair<int, int> coord, Player player) const;

  // Partial observation of the specific agent.
  ObservationType PartialObservation(Player player) const;

  // Observation planes for the fully-observable case.
  int ObservationPlane(std::pair<int, int> coord, Player player) const;

  // Fields sets to bad/invalid values. Use Game::NewInitialState().
  double total_rewards_ = -1;
  int horizon_ = -1;  // Limit on the total number of moves.
  Player cur_player_ = kSimultaneousPlayerId;
  int total_moves_ = 0;
  int initiative_;  // player id of player to resolve actions first.
  bool win_;        // True if agents push the big box to the goal.
  bool fully_observable_;

  // Most recent rewards.
  double reward_;
  // All coordinates below are (row, col).
  std::array<std::pair<int, int>, 2> player_coords_;  // Players' coordinates.
  // Players' orientations.
  std::array<OrientationType, 2> player_orient_;
  // Moves chosen by agents.
  std::array<ActionType, 2> moves_;
  // The status of each of the players' moves.
  std::array<ActionStatusType, 2> action_status_;
  // Actual field used by the players.
  std::vector<char> field_;
};

class CoopBoxPushingGame : public SimMoveGame {
 public:
  explicit CoopBoxPushingGame(const GameParameters& params);
  int NumDistinctActions() const override;
  std::unique_ptr<State> NewInitialState() const override;
  int MaxChanceOutcomes() const override { return 4; }
  int NumPlayers() const override;
  double MinUtility() const override;
  double MaxUtility() const override;
  std::vector<int> ObservationTensorShape() const override;
  int MaxGameLength() const override { return horizon_; }
  // TODO: verify whether this bound is tight and/or tighten it.
  int MaxChanceNodesInHistory() const override { return MaxGameLength(); }

 private:
  int horizon_;
  bool fully_observable_;
};

}  // namespace coop_box_pushing
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_COOP_BOX_PUSHING
