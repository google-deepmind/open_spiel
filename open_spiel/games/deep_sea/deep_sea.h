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

#ifndef OPEN_SPIEL_GAMES_DEEP_SEA_H_
#define OPEN_SPIEL_GAMES_DEEP_SEA_H_

#include <memory>
#include <string>
#include <vector>

#include "open_spiel/spiel.h"

// Implementation of 'Deep Sea' exploration environment.
//
// This environment is designed as a stylized version of the 'exploration
// chain', such as the classical river swim environment (Strehl & Littman '04,
// https://ieeexplore.ieee.org/document/1374179). The observation is an N x N
// grid, with a falling block starting in top left. Each timestep the agent can
// move 'left' or 'right', which are mapped to discrete actions 0 and 1 on a
// state-dependent level. There is a large reward of +1 in the bottom right
// state, but this can be hard for many exploration algorithms to find.
//
// For more information, see papers:
// [1] https://arxiv.org/abs/1703.07608
// [2] https://arxiv.org/abs/1806.03335
//
// Parameters:
//  "size"                int      rows and columns             (default = 5)
//  "seed"                int      seed for randomizing actions (default = 42)
//  "unscaled_move_cost"  double   move cost                    (default = 0.01)
//  "randomize_actions"   bool     state dependent actions      (default = true)

namespace open_spiel {
namespace deep_sea {

// Constants.
constexpr int kNumPlayers = 1;
constexpr int kNumActions = 2;

constexpr int kDefaultSize = 5;
constexpr int kDefaultSeed = 42;
constexpr double kDefaultUnscaledMoveCost = 0.01;
constexpr bool kDefaultRandomizeActions = true;

// State of an in-play game.
class DeepSeaState : public State {
 public:
  DeepSeaState(const std::shared_ptr<const Game> game);
  DeepSeaState(const DeepSeaState&) = default;

  Player CurrentPlayer() const override;
  std::string ActionToString(Player player, Action action_id) const override;
  std::string ToString() const override;
  bool IsTerminal() const override;
  std::vector<double> Rewards() const override;
  std::vector<double> Returns() const override;
  std::string ObservationString(Player player) const override;
  void ObservationTensor(Player player,
                         absl::Span<float> values) const override;
  std::unique_ptr<State> Clone() const override;
  void UndoAction(Player player, Action move) override;
  std::vector<Action> LegalActions() const override;

 protected:
  void DoApplyAction(Action move) override;

 private:
  // Copied from game.
  int size_;
  double move_cost_;  // Cost of moving right.
  std::vector<bool> action_mapping_;

  // Position of the player.
  int player_row_ = 0;
  int player_col_ = 0;

  // History of actual moves. `true` means RIGHT, otherwise LEFT.
  std::vector<bool> direction_history_;
};

// Game object.
class DeepSeaGame : public Game {
 public:
  explicit DeepSeaGame(const GameParameters& params);
  std::unique_ptr<State> NewInitialState() const override {
    return std::unique_ptr<State>(new DeepSeaState(shared_from_this()));
  }
  std::vector<int> ObservationTensorShape() const override {
    return {size_, size_};
  }

  int NumDistinctActions() const override { return kNumActions; }
  int MaxChanceOutcomes() const override { return kNumActions; }
  int NumPlayers() const override { return kNumPlayers; }
  double MaxUtility() const override { return 1 - unscaled_move_cost_; }
  double MinUtility() const override { return -unscaled_move_cost_; }
  int MaxGameLength() const override { return size_; }

  double UnscaledMoveCost() const { return unscaled_move_cost_; }

  // Wether the action will be reversed (false) or upheld (true).
  std::vector<bool> ActionMapping() const { return action_mapping_; }

 private:
  const int size_;
  const double unscaled_move_cost_;
  std::vector<bool> action_mapping_;
};

}  // namespace deep_sea
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_DEEP_SEA_H_
