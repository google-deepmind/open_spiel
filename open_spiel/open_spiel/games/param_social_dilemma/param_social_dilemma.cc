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

#include "open_spiel/abseil-cpp/absl/random/bit_gen_ref.h"
#include "open_spiel/simultaneous_move_game.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

#include <algorithm>
#include <cmath>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace open_spiel {
namespace param_social_dilemma {

namespace {

// Default payoff matrix for 2-player prisoner's dilemma
constexpr std::array<double, 4> kDefault2PlayerPayoffs = {
    3, 0,  // Player 0: C-C, C-D
    5, 1   // Player 0: D-C, D-D
};

// Game parameters
constexpr int kMinPlayers = 2;
constexpr int kMaxPlayers = 10;
constexpr int kMinActions = 2;
constexpr double kDefaultTerminationProbability = 0.125;
constexpr int kDefaultMaxGameLength = 9999;

// Facts about the game
const GameType kGameType{
    /*short_name=*/"param_social_dilemma",
    /*long_name=*/"Parameterized Social Dilemma",
    GameType::Dynamics::kSimultaneous,
    GameType::ChanceMode::kExplicitStochastic,
    GameType::Information::kPerfectInformation,
    GameType::Utility::kGeneralSum,
    GameType::RewardModel::kRewards,
    /*max_num_players=*/kMaxPlayers,
    /*min_num_players=*/kMinPlayers,
    /*provides_information_state_string=*/true,
    /*provides_information_state_tensor=*/true,
    /*provides_observation_string=*/true,
    /*provides_observation_tensor=*/true,
    /*parameter_specification=*/
    {
        {"num_players", GameParameter::Type::kInt, false, 2},
        {"num_actions", GameParameter::Type::kInt, false, 2},
        {"payoff_matrix", GameParameter::Type::kDoubleList, true},
        {"termination_probability", GameParameter::Type::kDouble, false, kDefaultTerminationProbability},
        {"max_game_length", GameParameter::Type::kInt, false, kDefaultMaxGameLength},
        {"reward_noise_std", GameParameter::Type::kDouble, false, 0.0},
        {"reward_noise_type", GameParameter::Type::kString, false, "none"},
        {"seed", GameParameter::Type::kInt, false, -1}
    }
};

std::unique_ptr<Game> Factory(const GameParameters& params) {
  return std::unique_ptr<Game>(new ParamSocialDilemmaGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);

}  // namespace

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

ParamSocialDilemmaState::ParamSocialDilemmaState(
    std::shared_ptr<const Game> game, int num_players, int num_actions,
    double termination_probability,
    const std::vector<std::vector<double>>& payoff_matrix,
    double reward_noise_std, const std::string& reward_noise_type, int seed)
    : SimultaneousMoveState(game),
      num_players_(num_players),
      num_actions_(num_actions),
      termination_probability_(termination_probability),
      payoff_matrix_(payoff_matrix),
      reward_noise_std_(reward_noise_std),
      reward_noise_type_(reward_noise_type),
      current_iteration_(1),
      is_chance_(false),
      rewards_(num_players, 0.0),
      returns_(num_players, 0.0) {
  
  // Initialize random number generator
  if (seed != -1) {
    rng_.seed(seed);
  } else {
    std::random_device rd;
    rng_.seed(rd());
  }
  
  normal_dist_ = std::normal_distribution<double>(0.0, reward_noise_std);
  uniform_dist_ = std::uniform_real_distribution<double>(-reward_noise_std, reward_noise_std);
  
  // Initialize action history
  action_history_.resize(num_players);
  
  // Validate payoff matrix
  if (payoff_matrix_.empty()) {
    payoff_matrix_ = CreateDefaultPayoffMatrix(num_players, num_actions);
  }
}

ParamSocialDilemmaState::ParamSocialDilemmaState(const ParamSocialDilemmaState& other)
    : SimultaneousMoveState(other),
      num_players_(other.num_players_),
      num_actions_(other.num_actions_),
      termination_probability_(other.termination_probability_),
      payoff_matrix_(other.payoff_matrix_),
      reward_noise_std_(other.reward_noise_std_),
      reward_noise_type_(other.reward_noise_type_),
      current_iteration_(other.current_iteration_),
      is_chance_(other.is_chance_),
      action_history_(other.action_history_),
      rewards_(other.rewards_),
      returns_(other.returns_),
      rng_(other.rng_),
      normal_dist_(other.normal_dist_),
      uniform_dist_(other.uniform_dist_) {}

ParamSocialDilemmaState& ParamSocialDilemmaState::operator=(
    const ParamSocialDilemmaState& other) {
  if (this == &other) return *this;
  
  SimultaneousMoveState::operator=(other);
  
  num_players_ = other.num_players_;
  num_actions_ = other.num_actions_;
  termination_probability_ = other.termination_probability_;
  payoff_matrix_ = other.payoff_matrix_;
  reward_noise_std_ = other.reward_noise_std_;
  reward_noise_type_ = other.reward_noise_type_;
  current_iteration_ = other.current_iteration_;
  is_chance_ = other.is_chance_;
  action_history_ = other.action_history_;
  rewards_ = other.rewards_;
  returns_ = other.returns_;
  rng_ = other.rng_;
  normal_dist_ = other.normal_dist_;
  uniform_dist_ = other.uniform_dist_;
  
  return *this;
}

std::string ParamSocialDilemmaState::ActionToString(Player player,
                                               Action action_id) const {
  if (player == kChancePlayerId) {
    return action_id == 0 ? "Continue" : "Stop";
  }
  
  if (num_actions_ == 2) {
    return action_id == 0 ? "Cooperate" : "Defect";
  } else {
    return "Action" + std::to_string(action_id);
  }
}

std::string ParamSocialDilemmaState::ToString() const {
  std::string result;
  for (Player p = 0; p < num_players_; ++p) {
    if (p > 0) result += " ";
    result += "p" + std::to_string(p) + ":";
    
    for (Action action : action_history_[p]) {
      if (num_actions_ == 2) {
        result += action == 0 ? "C" : "D";
      } else {
        result += std::to_string(action);
      }
    }
  }
  return result;
}

bool ParamSocialDilemmaState::IsTerminal() const {
  return current_player_ == kTerminalPlayerId;
}

std::vector<double> ParamSocialDilemmaState::Rewards() const {
  return rewards_;
}

std::vector<double> ParamSocialDilemmaState::Returns() const {
  return returns_;
}

std::string ParamSocialDilemmaState::InformationStateString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  
  std::string result = "History: ";
  for (Action action : action_history_[player]) {
    if (num_actions_ == 2) {
      result += action == 0 ? "C" : "D";
    } else {
      result += std::to_string(action);
    }
  }
  return result;
}

std::string ParamSocialDilemmaState::ObservationString(Player player) const {
  return InformationStateString(player);
}

void ParamSocialDilemmaState::ObservationTensor(Player player,
                                            absl::Span<float> values) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  
  // Create observation with action history and iteration
  const Game* game = game_.get();
  int max_length = game->max_game_length();
  
  // Fill with action history for all players
  for (int p = 0; p < num_players_; ++p) {
    for (int i = 0; i < max_length; ++i) {
      int idx = p * max_length + i;
      if (i < static_cast<int>(action_history_[p].size())) {
        values[idx] = static_cast<float>(action_history_[p][i]);
      } else {
        values[idx] = 0.0f;
      }
    }
  }
  
  // Add current iteration (normalized)
  values[num_players_ * max_length] = static_cast<float>(current_iteration_) / max_length;
}

std::unique_ptr<State> ParamSocialDilemmaState::Clone() const {
  return std::unique_ptr<State>(new ParamSocialDilemmaState(*this));
}

void ParamSocialDilemmaState::DoApplyActions(const std::vector<Action>& actions) {
  SPIEL_CHECK_EQ(actions.size(), num_players_);
  
  // Store actions in history
  for (Player p = 0; p < num_players_; ++p) {
    action_history_[p].push_back(actions[p]);
  }
  
  // Calculate rewards for each player
  for (Player p = 0; p < num_players_; ++p) {
    int payoff_index = GetPayoffIndex(p, actions);
    double base_reward = payoff_matrix_[p * num_actions_ + actions[p] * num_players_ + payoff_index];
    rewards_[p] = AddNoise(base_reward);
    returns_[p] += rewards_[p];
  }
  
  // Move to chance node for termination decision
  current_iteration_++;
  is_chance_ = true;
  current_player_ = kChancePlayerId;
}

void ParamSocialDilemmaState::DoApplyAction(Action action) {
  SPIEL_CHECK_EQ(current_player_, kChancePlayerId);
  
  is_chance_ = false;
  
  if (action == 1) {  // Stop
    current_player_ = kTerminalPlayerId;
  } else {  // Continue
    current_player_ = kSimultaneousPlayerId;
    
    // Check max game length
    if (current_iteration_ > game_->max_game_length()) {
      current_player_ = kTerminalPlayerId;
    }
  }
}

int ParamSocialDilemmaState::GetPayoffIndex(
    Player player, const std::vector<Action>& actions) const {
  int index = 0;
  int multiplier = 1;
  
  for (Player p = 0; p < num_players_; ++p) {
    if (p != player) {
      index += actions[p] * multiplier;
      multiplier *= num_actions_;
    }
  }
  
  return index;
}

double ParamSocialDilemmaState::AddNoise(double base_reward) {
  if (reward_noise_std_ <= 0.0) {
    return base_reward;
  }
  
  if (reward_noise_type_ == "gaussian") {
    return base_reward + normal_dist_(rng_);
  } else if (reward_noise_type_ == "uniform") {
    return base_reward + uniform_dist_(rng_);
  } else if (reward_noise_type_ == "discrete") {
    // Discrete noise: -std, 0, +std with equal probability
    std::uniform_int_distribution<int> discrete_dist(-1, 1);
    return base_reward + discrete_dist(rng_) * reward_noise_std_;
  }
  
  return base_reward;
}

std::vector<double> ParamSocialDilemmaState::CreateDefaultPayoffMatrix(
    int num_players, int num_actions) const {
  std::vector<double> payoff_matrix;
  
  if (num_players == 2 && num_actions == 2) {
    // Classic prisoner's dilemma
    payoff_matrix = {
        3, 0,  // Player 0: C-C, C-D
        5, 1,  // Player 0: D-C, D-D
        3, 5,  // Player 1: C-C, D-C
        0, 1   // Player 1: C-D, D-D
    };
  } else {
    // Generate N-player social dilemma payoffs
    int total_entries = num_players * num_actions * std::pow(num_actions, num_players - 1);
    payoff_matrix.resize(total_entries);
    
    for (Player p = 0; p < num_players; ++p) {
      for (int a = 0; a < num_actions; ++a) {
        for (int combo = 0; combo < std::pow(num_actions, num_players - 1); ++combo) {
          int idx = p * num_actions * std::pow(num_actions, num_players - 1) + 
                    a * std::pow(num_actions, num_players - 1) + combo;
          
          // Count cooperators among others
          int num_cooperators = 0;
          int temp = combo;
          for (Player other = 0; other < num_players; ++other) {
            if (other != p) {
              int other_action = temp % num_actions;
              if (other_action == 0) num_cooperators++;
              temp /= num_actions;
            }
          }
          
          // Calculate payoff based on cooperation/defection pattern
          double payoff;
          if (a == 0) {  // Cooperate
            payoff = 2.0 + num_cooperators * 0.5;
          } else {  // Defect
            payoff = 4.0 + num_cooperators * 0.2;
          }
          
          payoff_matrix[idx] = payoff;
        }
      }
    }
  }
  
  return payoff_matrix;
}

class ParamSocialDilemmaGame : public SimultaneousMoveGame {
 public:
  explicit ParamSocialDilemmaGame(const GameParameters& params);
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

ParamSocialDilemmaGame::ParamSocialDilemmaGame(const GameParameters& params)
    : SimultaneousMoveGame(kGameType, params),
      num_players_(ParameterValue<int>(params, "num_players")),
      num_actions_(ParameterValue<int>(params, "num_actions")),
      termination_probability_(
          ParameterValue<double>(params, "termination_probability")),
      reward_noise_std_(ParameterValue<double>(params, "reward_noise_std", 0.0)),
      reward_noise_type_(
          ParameterValue<std::string>(params, "reward_noise_type", "none")),
      max_game_length_(ParameterValue<int>(params, "max_game_length")) {
  
  SPIEL_CHECK_GE(num_players_, kMinPlayers);
  SPIEL_CHECK_LE(num_players_, kMaxPlayers);
  SPIEL_CHECK_GE(num_actions_, kMinActions);
  
  // Load or create payoff matrix
  if (params.count("payoff_matrix") > 0) {
    payoff_matrix_ = ParameterValue<std::vector<double>>(params, "payoff_matrix");
  } else {
    // Create default payoff matrix
    ParamSocialDilemmaState dummy_state(
        std::shared_ptr<const Game>(), num_players_, num_actions_,
        termination_probability_, {}, reward_noise_std_, reward_noise_type_, -1);
    payoff_matrix_ = dummy_state.CreateDefaultPayoffMatrix(num_players_, num_actions_);
  }
}

int ParamSocialDilemmaGame::NumDistinctActions() const {
  return num_actions_;
}

std::unique_ptr<State> ParamSocialDilemmaGame::NewInitialState() const {
  return std::make_unique<ParamSocialDilemmaState>(
      shared_from_this(), num_players_, num_actions_, termination_probability_,
      payoff_matrix_, reward_noise_std_, reward_noise_type_, -1);
}

int ParamSocialDilemmaGame::MaxGameLength() const {
  return max_game_length_;
}

int ParamSocialDilemmaGame::NumPlayers() const {
  return num_players_;
}

double ParamSocialDilemmaGame::MinUtility() const {
  return CalculateMinUtility();
}

double ParamSocialDilemmaGame::MaxUtility() const {
  return CalculateMaxUtility();
}

std::vector<double> ParamSocialDilemmaGame::RewardBound() const {
  double min_val = CalculateMinUtility();
  double max_val = CalculateMaxUtility();
  return {min_val, max_val};
}

std::shared_ptr<const Game> ParamSocialDilemmaGame::Clone() const {
  return std::shared_ptr<const Game>(new ParamSocialDilemmaGame(*this));
}

double ParamSocialDilemmaGame::CalculateMinUtility() const {
  if (payoff_matrix_.empty()) return 0.0;
  return *std::min_element(payoff_matrix_.begin(), payoff_matrix_.end()) * max_game_length_;
}

double ParamSocialDilemmaGame::CalculateMaxUtility() const {
  if (payoff_matrix_.empty()) return 0.0;
  return *std::max_element(payoff_matrix_.begin(), payoff_matrix_.end()) * max_game_length_;
}

}  // namespace param_social_dilemma
}  // namespace open_spiel
