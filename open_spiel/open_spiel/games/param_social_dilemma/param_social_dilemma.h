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

#ifndef OPEN_SPIEL_GAMES_PARAM_SOCIAL_DILEMMA_H_
#define OPEN_SPIEL_GAMES_PARAM_SOCIAL_DILEMMA_H_

#include <memory>
#include <string>
#include <vector>

#include "open_spiel/simultaneous_move_game.h"
#include "open_spiel/spiel.h"

namespace open_spiel {
namespace param_social_dilemma {

// Forward declaration
class ParamSocialDilemmaState;

// Game parameters and constants
constexpr int kMinPlayers = 2;
constexpr int kMaxPlayers = 10;
constexpr int kMinActions = 2;
constexpr double kDefaultTerminationProbability = 0.125;
constexpr int kDefaultMaxGameLength = 9999;

// Game type information
extern const GameType kGameType;

// Factory function
std::unique_ptr<Game> Factory(const GameParameters& params);

// Game class
class ParamSocialDilemmaGame : public SimultaneousMoveGame {
 public:
  explicit ParamSocialDilemmaGame(const GameParameters& params);
  ~ParamSocialDilemmaGame() override = default;

  int NumDistinctActions() const override;
  std::unique_ptr<State> NewInitialState() const override;
  int MaxGameLength() const override;
  int NumPlayers() const override;
  double MinUtility() const override;
  double MaxUtility() const override;
  std::vector<double> RewardBound() const override;
  std::shared_ptr<const Game> Clone() const override;

 private:
  int num_players_;
  int num_actions_;
  double termination_probability_;
  std::vector<std::vector<double>> payoff_matrix_;
  double reward_noise_std_;
  std::string reward_noise_type_;
  int max_game_length_;
  
  double CalculateMinUtility() const;
  double CalculateMaxUtility() const;
};

// State class
class ParamSocialDilemmaState : public SimultaneousMoveState {
 public:
  ParamSocialDilemmaState(std::shared_ptr<const Game> game, int num_players,
                          int num_actions, double termination_probability,
                          const std::vector<std::vector<double>>& payoff_matrix,
                          double reward_noise_std,
                          const std::string& reward_noise_type,
                          int seed);
  
  ParamSocialDilemmaState(const ParamSocialDilemmaState& other);
  ParamSocialDilemmaState& operator=(const ParamSocialDilemmaState& other);
  ~ParamSocialDilemmaState() override = default;

  std::string ActionToString(Player player, Action action_id) const override;
  std::string ToString() const override;
  bool IsTerminal() const override;
  std::vector<double> Rewards() const override;
  std::vector<double> Returns() const override;
  std::string InformationStateString(Player player) const override;
  std::string ObservationString(Player player) const override;
  void ObservationTensor(Player player,
                     absl::Span<float> values) const override;

 protected:
  std::unique_ptr<State> Clone() const override;
  void DoApplyActions(const std::vector<Action>& actions) override;
  void DoApplyAction(Action action) override;

 private:
  // Game parameters
  int num_players_;
  int num_actions_;
  double termination_probability_;
  std::vector<std::vector<double>> payoff_matrix_;
  double reward_noise_std_;
  std::string reward_noise_type_;
  
  // Random number generation
  std::mt19937 rng_;
  std::normal_distribution<double> normal_dist_;
  std::uniform_real_distribution<double> uniform_dist_;
  
  // State tracking
  int current_iteration_;
  bool is_chance_;
  std::vector<std::vector<Action>> action_history_;
  std::vector<double> rewards_;
  std::vector<double> returns_;
  
  // Helper methods
  int GetPayoffIndex(Player player, const std::vector<Action>& actions) const;
  double AddNoise(double base_reward);
  std::vector<double> CreateDefaultPayoffMatrix(int num_players, int num_actions) const;
};

}  // namespace param_social_dilemma
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_PARAM_SOCIAL_DILEMMA_H_
