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

// Mean Field Crowd Modelling Game.
//
// This game corresponds to the "Beach Bar Process" defined in section 4.2 of
// "Fictitious play for mean field games: Continuous time analysis and
// applications", Perrin & al. 2019 (https://arxiv.org/abs/2007.03458).
//
// In a nutshell, each representative agent evolves on a circle, with {left,
// neutral, right} actions. The reward includes the proximity to an imagined bar
// placed at a fixed location in the circle, and penalties for moving and for
// being in a crowded place.

#ifndef OPEN_SPIEL_GAMES_MFG_CROWD_MODELLING_H_
#define OPEN_SPIEL_GAMES_MFG_CROWD_MODELLING_H_

#include <array>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "open_spiel/abseil-cpp/absl/memory/memory.h"
#include "open_spiel/spiel.h"

namespace open_spiel {
namespace crowd_modelling {

inline constexpr int kNumPlayers = 1;
inline constexpr int kDefaultHorizon = 10;
inline constexpr int kDefaultSize = 10;
inline constexpr int kNumActions = 3;
inline constexpr int kNumChanceActions = 3;
// Action that leads to no displacement on the circle of the game.
inline constexpr int kNeutralAction = 1;

// Game state.
// The high-level state transitions are as follows:
// - First game state is a chance node where the initial position on the
//   circle is selected.
// Then we cycle over:
// 1. Decision node with actions {left, neutral, right}, represented by integers
//    0, 1, 2. This moves the position on the circle.
// 2. Mean field node, where we expect that external logic will call
//    DistributionSupport() and UpdateDistribution().
// 3. Chance node, where one of {left, neutral, right} actions is externally
//    selected.
// The game stops after a non-initial chance node when the horizon is reached.
class CrowdModellingState : public State {
 public:
  CrowdModellingState(std::shared_ptr<const Game> game, int size, int horizon);
  CrowdModellingState(std::shared_ptr<const Game> game, int size, int horizon,
                      Player current_player, bool is_chance_init, int x, int t,
                      int last_action, double return_value,
                      const std::vector<double>& distribution);

  CrowdModellingState(const CrowdModellingState&) = default;
  CrowdModellingState& operator=(const CrowdModellingState&) = default;

  Player CurrentPlayer() const override {
    return IsTerminal() ? kTerminalPlayerId : current_player_;
  }
  std::string ActionToString(Player player, Action action) const override;
  std::string ToString() const override;
  bool IsTerminal() const override;
  std::vector<double> Rewards() const override;
  std::vector<double> Returns() const override;
  std::string InformationStateString(Player player) const override;
  std::string ObservationString(Player player) const override;
  void ObservationTensor(Player player,
                         absl::Span<float> values) const override;
  std::unique_ptr<State> Clone() const override;
  std::vector<Action> LegalActions() const override;
  ActionsAndProbs ChanceOutcomes() const override;

  std::vector<std::string> DistributionSupport() override;
  void UpdateDistribution(const std::vector<double>& distribution) override;
  std::vector<double> Distribution() const { return distribution_; }

  std::string Serialize() const override;

 protected:
  void DoApplyAction(Action action) override;

 private:
  // Size of the circle.
  const int size_ = -1;
  const int horizon_ = -1;
  Player current_player_ = kChancePlayerId;
  bool is_chance_init_ = true;
  // Position on the circle [0, size_) when valid.
  int x_ = -1;
  // Current time, in [0, horizon_].
  int t_ = 0;
  int last_action_ = kNeutralAction;
  double return_value_ = 0.;

  // kActionToMove[action] is the displacement on the circle of the game for
  // 'action'.
  static constexpr std::array<int, 3> kActionToMove = {-1, 0, 1};
  // Represents the current probability distribution over game states.
  std::vector<double> distribution_;
};

class CrowdModellingGame : public Game {
 public:
  explicit CrowdModellingGame(const GameParameters& params);
  int NumDistinctActions() const override { return kNumActions; }
  std::unique_ptr<State> NewInitialState() const override {
    return absl::make_unique<CrowdModellingState>(shared_from_this(), size_,
                                                  horizon_);
  }
  int NumPlayers() const override { return kNumPlayers; }
  double MinUtility() const override {
    return -std::numeric_limits<double>::infinity();
  }
  double UtilitySum() const override { return 0; }
  double MaxUtility() const override {
    return std::numeric_limits<double>::infinity();
  }
  int MaxGameLength() const override { return horizon_; }
  int MaxChanceNodesInHistory() const override {
    // + 1 to account for the initial extra chance node.
    return horizon_ + 1;
  }
  std::vector<int> ObservationTensorShape() const override;
  int MaxChanceOutcomes() const override {
    return std::max(size_, kNumChanceActions);
  }
  std::unique_ptr<State> DeserializeState(
      const std::string& str) const override;

 private:
  const int size_;
  const int horizon_;
};

}  // namespace crowd_modelling
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_MFG_CROWD_MODELLING_H_
