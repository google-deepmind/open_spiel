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

#ifndef OPEN_SPIEL_GAMES_CLIFF_WALKING_H_
#define OPEN_SPIEL_GAMES_CLIFF_WALKING_H_

#include <memory>
#include <string>
#include <vector>

#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

// A cliff walking reinforcement learning environment.
//
// This is a deterministic environment that can be used to test RL algorithms.
// Note there are *no illegal moves* in this environment--if the agent is on the
// gridworld boundary and takes an action which would yield an invalid position,
// the action is ignored (as if there were walls surrounding the grid world).
//
// The player spawns at the bottom left and must reach the goal located on the
// bottom right. A game is terminal when the player reaches the goal, the
// maximum episode length has been reached (horizon) or when the player steps
// off the cliff edge (see figure below). The player receives a reward of -1 for
// all transitions except when stepping off the cliff, where a reward of -100 is
// received.
//
// Cliff example for height=3 and width=5:
//
//               |   |   |   |   |   |
//               |   |   |   |   |   |
//               | S | x | x | x | G |
//
// where `S` is always the starting position, `G` is always the goal and `x`
// represents the zone of high negative reward to be avoided. For this instance,
// the optimum policy is depicted as follows:
//
//               |   |   |   |   |   |
//               |-->|-->|-->|-->|\|/|
//               |/|\| x | x | x | G |
//
// yielding a reward of -6 (minus 1 per time step).
//
// See pages 132 of Rich Sutton's book for details:
// http://www.incompleteideas.net/book/bookdraft2018mar21.pdf
//
// Parameters:
//  "height"     int     rows of the board                       (default = 4)
//  "width"      int     columns of the board                    (default = 8)
//  "horizon"    int     maximum episode length                  (default = 100)

namespace open_spiel {
namespace cliff_walking {

// Constants.
inline constexpr int kNumPlayers = 1;
inline constexpr int kNumActions = 4;  // Right, Up, Left, Down.

inline constexpr int kDefaultHeight = 4;
inline constexpr int kDefaultWidth = 8;
inline constexpr int kDefaultHorizon = 100;

class CliffWalkingGame;

// State of an in-play game.
class CliffWalkingState : public State {
 public:
  CliffWalkingState(std::shared_ptr<const Game> game);
  CliffWalkingState(const CliffWalkingState&) = default;

  Player CurrentPlayer() const override;
  std::string ActionToString(Player player, Action action_id) const override;
  std::string ToString() const override;
  bool IsTerminal() const override;
  std::vector<double> Rewards() const override;
  std::vector<double> Returns() const override;
  std::string InformationStateString(Player player) const override;
  std::string ObservationString(Player player) const override;
  void ObservationTensor(Player player,
                         absl::Span<float> values) const override;
  void InformationStateTensor(Player player,
                              absl::Span<float> values) const override;
  std::unique_ptr<State> Clone() const override;
  std::vector<Action> LegalActions() const override;

 protected:
  void DoApplyAction(Action move) override;

 private:
  // Check if player position is in bottom row between start and goal.
  bool IsCliff(int row, int col) const;

  bool IsGoal(int row, int col) const;

  // Copied from CliffWalkingGame.
  int height_;
  int width_;
  int horizon_;

  int player_row_;
  int player_col_ = 0;
  int time_counter_ = 0;
};

// Game object.
class CliffWalkingGame : public Game {
 public:
  explicit CliffWalkingGame(const GameParameters& params);
  std::unique_ptr<State> NewInitialState() const override {
    return std::unique_ptr<State>(new CliffWalkingState(shared_from_this()));
  }
  std::vector<int> ObservationTensorShape() const override {
    return {height_, width_};
  }
  std::vector<int> InformationStateTensorShape() const override {
    return {kNumActions * horizon_};
  }

  int NumDistinctActions() const override { return kNumActions; }
  int NumPlayers() const override { return kNumPlayers; }
  double MaxUtility() const override { return -width_ - 1; }
  double MinUtility() const override { return -horizon_ + 1 - 100; }
  int MaxGameLength() const override { return horizon_; }
  int Height() const { return height_; }
  int Width() const { return width_; }

 private:
  const int height_;
  const int width_;
  const int horizon_;
};

}  // namespace cliff_walking
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_CLIFF_WALKING_H_
