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

// Mean Field Crowd Modelling Game in 2d.
//
// This game corresponds to a 2D "Beach Bar Process" defined in section 4.2 of
// "Fictitious play for mean field games: Continuous time analysis and
// applications", Perrin & al. 2019 (https://arxiv.org/abs/2007.03458).
//
// In a nutshell, each representative agent evolves on a 2d torus, with {down,
// left, neutral, right, up} actions. The reward includes the proximity to an
// imagined bar placed at a fixed location in the torus, and penalties for
// moving and for being in a crowded place.

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
namespace crowd_modelling_2d {

inline constexpr int kNumPlayers = 1;
inline constexpr int kDefaultHorizon = 10;
inline constexpr int kDefaultSize = 10;
inline constexpr int kNumActions = 5;
inline constexpr int kNumChanceActions = 5;
inline constexpr bool kDefaultOnlyDistributionReward = false;
inline constexpr bool kDefaultWithCongestion = false;
// Noise intensity is distributed uniformly among the legal actions in the
// chance node. Neutral action will get an additional probability of 1 -
// noise_intensity.
inline constexpr double kDefaultNoiseIntensity = 1.0;
inline constexpr double kDefaultCrowdAversionCoef = 1.0;

// Example string format: "[0|0;0|1]"
inline constexpr const char* kDefaultForbiddenStates = "[]";
// Example string format: "[0|2;0|3]"
inline constexpr const char* kDefaultInitialDistribution = "[]";
// Example string format: "[0.5;0.5]"
inline constexpr const char* kDefaultInitialDistributionValue = "[]";
// Example string format: "[0|2;0|3]"
inline constexpr const char* kDefaultPositionalReward = "[]";
// Example string format: "[1.5;2.5]"
inline constexpr const char* kDefaultPositionalRewardValue = "[]";

// Action that leads to no displacement on the torus of the game.
inline constexpr int kNeutralAction = 2;

std::vector<absl::string_view> ProcessStringParam(
    const std::string& string_param_str, int max_size);

// Game state.
// The high-level state transitions are as follows:
// - First game state is a chance node where the initial position on the
//   torus is selected.
// Then we cycle over:
// 1. Decision node with actions {down, left, neutral, right, up}, represented
// by integers 0, 1, 2, 3, 4. This moves the position on the torus.
// 2. Mean field node, where we expect that external logic will call
//    DistributionSupport() and UpdateDistribution().
// 3. Chance node, where one of {down, left, neutral, right, up} actions is
// externally selected.
// The game stops after a non-initial chance node when the horizon is reached.
class CrowdModelling2dState : public State {
 public:
  // forbidden_states, initial_distribution and positional_reward are formated
  // like '[int|int;...;int|int]'. Example : "[]" or "[0|0;0|1]".
  // initial_distribution_value and positional_reward_value are formated like
  // '[float;...;float]'. Example : "[]" or "[0.5;0.5]"
  CrowdModelling2dState(std::shared_ptr<const Game> game, int size, int horizon,
                        bool only_distribution_reward,
                        const std::string& forbidden_states,
                        const std::string& initial_distribution,
                        const std::string& initial_distribution_value,
                        const std::string& positional_reward,
                        const std::string& positional_reward_value,
                        bool with_congestion, double noise_intensity,
                        double crowd_aversion_coef);
  CrowdModelling2dState(std::shared_ptr<const Game> game, int size, int horizon,
                        bool only_distribution_reward,
                        const std::string& forbidden_states,
                        const std::string& initial_distribution,
                        const std::string& initial_distribution_value,
                        const std::string& positional_reward,
                        const std::string& positional_reward_value,
                        Player current_player, bool is_chance_init_, int x,
                        int y, int t, int last_action, double return_value,
                        const std::vector<double>& distribution,
                        bool with_congestion, double noise_intensity,
                        double crowd_aversion_coef);

  CrowdModelling2dState(const CrowdModelling2dState&) = default;
  CrowdModelling2dState& operator=(const CrowdModelling2dState&) = default;

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

  std::string Serialize() const override;

 protected:
  void DoApplyAction(Action action) override;
  // Returns true if the specified action leads to a forbidden position.
  bool IsForbidden(Action action) const;
  // Returns true if the specified position is forbidden.
  bool IsForbiddenPosition(int x, int y) const;
  // Returns the list if legal actions for the player.
  std::vector<Action> LegalPlayerActions() const;

 private:
  Player current_player_ = kChancePlayerId;
  bool is_chance_init_ = true;
  // 2D positions on the torus [0, size_) x [0, size_).
  int x_ = -1;
  int y_ = -1;
  // Current time, in [0, horizon_].
  int t_ = 0;
  // Size of the torus.
  const int size_ = -1;
  const int horizon_ = -1;
  const bool only_distribution_reward_ = false;
  ActionsAndProbs initial_distribution_action_prob_;
  std::vector<std::pair<int, int>> forbidden_states_xy_;
  std::vector<std::pair<int, int>> positional_reward_xy_;
  std::vector<float> positional_reward_value_;
  int last_action_ = kNeutralAction;
  double return_value_ = 0.;
  bool with_congestion_;
  double noise_intensity_;
  double crowd_aversion_coef_;

  // kActionToMoveX[action] and kActionToMoveY[action] is the displacement on
  // the torus of the game for 'action'.
  static constexpr std::array<int, 5> kActionToMoveX = {0, -1, 0, 1, 0};
  static constexpr std::array<int, 5> kActionToMoveY = {-1, 0, 0, 0, 1};
  // Represents the current probability distribution over game states.
  std::vector<double> distribution_;
};

class CrowdModelling2dGame : public Game {
 public:
  explicit CrowdModelling2dGame(const GameParameters& params);
  int NumDistinctActions() const override { return kNumActions; }
  std::unique_ptr<State> NewInitialState() const override {
    return absl::make_unique<CrowdModelling2dState>(
        shared_from_this(), size_, horizon_, only_distribution_reward_,
        forbidden_states_, initial_distribution_, initial_distribution_value_,
        positional_reward_, positional_reward_value_, with_congestion_,
        noise_intensity_, crowd_aversion_coef_);
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
    return std::max(size_ * size_, kNumChanceActions);
  }
  std::unique_ptr<State> DeserializeState(
      const std::string& str) const override;

 private:
  const int size_;
  const int horizon_;
  const bool only_distribution_reward_;
  std::string forbidden_states_;            // Default "", example "[0|0;0|1]"
  std::string initial_distribution_;        // Default "", example  "[0|2;0|3]"
  std::string initial_distribution_value_;  // Default "", example "[0.5;0.5]"
  std::string positional_reward_;           // Default "", example  "[0|2;0|3]"
  std::string positional_reward_value_;     // Default "", example "[1.5;2.5]"
  const bool with_congestion_;
  const double noise_intensity_;
  const double crowd_aversion_coef_;
};

}  // namespace crowd_modelling_2d
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_MFG_CROWD_MODELLING_H_
