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

// Mean Field Garnet.
//
// This game corresponds to a garnet defined in section 5.1 of
// "Scaling up Mean Field Games with Online Mirror Descent", Perolat & al. 2021
// (https://arxiv.org/pdf/2103.00623.pdf)
//
// A garnet is a parametrized family of randomly generated Mean Field Game. One
// can control the number of action, the number of chance actions and the
// sparsity of the reward.
// - The transition is randomly generated as an unnormalized uniform(0,1) over
// the chance actions and the next state is selected uniformly over the state
// space.
// - The reward is parametrized by eta as r(x,a) - eta * log(mu(x)) where r(x,a)
// is sampled uniformly over [0,1] only with probability the sparsity and 0.0
// otherwise.

#ifndef OPEN_SPIEL_GAMES_MFG_GARNET_H_
#define OPEN_SPIEL_GAMES_MFG_GARNET_H_

#include <array>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "open_spiel/abseil-cpp/absl/memory/memory.h"
#include "open_spiel/spiel.h"

namespace open_spiel {
namespace garnet {

inline constexpr int kNumPlayers = 1;
inline constexpr int kDefaultHorizon = 10;
inline constexpr int kDefaultSize = 10;
inline constexpr int kDefaultSeed = 0;
inline constexpr int kDefaultNumActions = 3;
inline constexpr int kDefaultNumChanceActions = 3;
inline constexpr double kDefaultSparsityFactor = 1.0;
inline constexpr double kDefaultEta = 1.0;
// Action that leads to no displacement on the circle of the game.
inline constexpr int kNeutralAction = 0;

// Game state.
class GarnetState : public State {
 public:
  GarnetState(std::shared_ptr<const Game> game, int size, int horizon, int seed,
              int num_action, int num_chance_action, double sparsity_factor,
              double eta);
  GarnetState(std::shared_ptr<const Game> game, int size, int horizon, int seed,
              int num_action, int num_chance_action, double sparsity_factor,
              double eta, Player current_player, bool is_chance_init, int x,
              int t, int last_action, double return_value,
              const std::vector<double>& distribution);

  double GetTransitionProba(int x, int action, int chance_action) const;
  int GetTransition(int x, int action, int chance_action) const;
  double GetReward(int x, int action) const;

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
  // Size of the garnet.
  const int size_ = -1;
  const int horizon_ = -1;
  const int seed_ = 0;
  const int num_action_ = 0;
  const int num_chance_action_ = 0;
  double sparsity_factor_ = kDefaultSparsityFactor;
  double eta_ = kDefaultEta;
  Player current_player_ = kChancePlayerId;
  bool is_chance_init_ = true;
  // Position on the garnet [0, size_) when valid.
  int x_ = -1;
  // Current time, in [0, horizon_].
  int t_ = 0;
  int last_action_ = kNeutralAction;
  double return_value_ = 0.;

  // Represents the current probability distribution over game states.
  std::vector<double> distribution_;
  std::vector<int> garnet_transition_;
  std::vector<double> garnet_transition_proba_unnormalized_;
  std::vector<double> garnet_transition_proba_normalization_;
  std::vector<double> garnet_reward_;
};

class GarnetGame : public Game {
 public:
  explicit GarnetGame(const GameParameters& params);
  int NumDistinctActions() const override { return num_action_; }
  std::unique_ptr<State> NewInitialState() const override {
    return absl::make_unique<GarnetState>(
        shared_from_this(), size_, horizon_, seed_, num_action_,
        num_chance_action_, sparsity_factor_, eta_);
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
    return std::max(size_, num_chance_action_);
  }
  std::unique_ptr<State> DeserializeState(
      const std::string& str) const override;

 private:
  const int size_;
  const int horizon_;
  const int seed_;
  const int num_action_ = 0;
  const int num_chance_action_ = 0;
  const double sparsity_factor_;
  const double eta_;
};

}  // namespace garnet
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_MFG_GARNET_H_
