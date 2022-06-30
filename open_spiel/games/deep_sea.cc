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

#include "open_spiel/games/deep_sea.h"

#include <algorithm>
#include <cstdlib>
#include <utility>

#include "open_spiel/abseil-cpp/absl/random/distributions.h"
#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/game_parameters.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace deep_sea {
namespace {

// Facts about the game.
const GameType kGameType{
    /*short_name=*/"deep_sea",
    /*long_name=*/"DeepSea",
    GameType::Dynamics::kSequential,
    GameType::ChanceMode::kDeterministic,
    GameType::Information::kImperfectInformation,
    GameType::Utility::kGeneralSum,
    GameType::RewardModel::kRewards,
    /*max_num_players=*/1,
    /*min_num_players=*/1,
    /*provides_information_state_string=*/false,
    /*provides_information_state_tensor=*/false,
    /*provides_observation_string=*/true,
    /*provides_observation_tensor=*/true,
    /*parameter_specification=*/
    {
        {"size", GameParameter(kDefaultSize)},
        {"seed", GameParameter(kDefaultSeed)},
        {"unscaled_move_cost", GameParameter(kDefaultUnscaledMoveCost)},
        {"randomize_actions", GameParameter(kDefaultRandomizeActions)},
    }};

std::shared_ptr<Game> Factory(const GameParameters& params) {
  return std::shared_ptr<Game>(new DeepSeaGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);

}  // namespace

DeepSeaState::DeepSeaState(std::shared_ptr<const Game> game) : State(game) {
  SPIEL_CHECK_TRUE(game);
  const DeepSeaGame& parent_game = static_cast<const DeepSeaGame&>(*game);
  size_ = parent_game.MaxGameLength();
  move_cost_ = -parent_game.UnscaledMoveCost() / size_;
  action_mapping_ = parent_game.ActionMapping();
}

int DeepSeaState::CurrentPlayer() const {
  if (IsTerminal()) return kTerminalPlayerId;
  return 0;
}

std::vector<Action> DeepSeaState::LegalActions() const {
  if (IsTerminal()) return {};
  return {0, 1};
}

std::string DeepSeaState::ActionToString(Player player,
                                         Action action_id) const {
  SPIEL_CHECK_EQ(player, 0);
  return action_id ? "RIGHT" : "LEFT";
}

std::string DeepSeaState::ToString() const {
  std::string str;
  str.reserve((size_ + 1) * (size_ + 2));
  for (int r = 0; r < size_ + 1; ++r) {
    for (int c = 0; c < size_ + 1; ++c) {
      if (player_row_ == r && player_col_ == c) {
        str += "x";
      } else if (r < size_ && c <= r) {
        str += action_mapping_[r * size_ + c] ? 'R' : 'L';
      } else {
        str += ".";
      }
    }
    str += "\n";
  }
  return str;
}

bool DeepSeaState::IsTerminal() const { return player_row_ == size_; }

std::vector<double> DeepSeaState::Rewards() const {
  double reward = 0;
  if (!direction_history_.empty() && direction_history_.back()) {
    reward += move_cost_;
  }
  if (IsTerminal() && player_col_ == size_) {
    reward += 1;
  }
  return {reward};
}

std::vector<double> DeepSeaState::Returns() const {
  double reward_sum = 0;
  for (bool direction : direction_history_)
    if (direction) reward_sum += move_cost_;
  if (IsTerminal() && player_col_ == size_) {
    reward_sum += 1;
  }
  return {reward_sum};
}

std::string DeepSeaState::ObservationString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);

  // We need to account for the possibility that `player_row == size_` at
  // terminal states, so that's why we add the +1.
  std::string str((size_ + 1) * size_, '.');
  str[player_row_ * size_ + player_col_] = 'x';
  return str;
}

void DeepSeaState::ObservationTensor(Player player,
                                     absl::Span<float> values) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);

  std::fill(values.begin(), values.end(), 0.);
  SPIEL_CHECK_EQ(values.size(), size_ * size_);
  if (player_row_ < size_ && player_col_ < size_)
    values[player_row_ * size_ + player_col_] = 1.0;
}

std::unique_ptr<State> DeepSeaState::Clone() const {
  return std::unique_ptr<State>(new DeepSeaState(*this));
}

void DeepSeaState::UndoAction(Player player, Action move) {
  // Can only reliably undo by replaying the actions. This is because moving
  // left from column 0 is a no-op, so we can't deduce the previous column if
  // we're now at column 0.
  direction_history_.pop_back();
  history_.pop_back();
  --move_number_;
  player_row_ = 0;
  player_col_ = 0;
  for (auto action_right : direction_history_) {
    if (action_right) {
      ++player_col_;
    } else if (player_col_ > 0) {
      --player_col_;
    }
    ++player_row_;
  }
}

void DeepSeaState::DoApplyAction(Action move) {
  const int i = player_row_ * size_ + player_col_;
  const bool action_right = move == action_mapping_[i];
  if (action_right) {
    ++player_col_;
  } else if (player_col_ > 0) {
    --player_col_;
  }
  ++player_row_;
  direction_history_.push_back(action_right);
}

DeepSeaGame::DeepSeaGame(const GameParameters& params)
    : Game(kGameType, params),
      size_(ParameterValue<int>("size")),
      unscaled_move_cost_(ParameterValue<double>("unscaled_move_cost")) {
  action_mapping_.resize(size_ * size_);
  if (ParameterValue<bool>("randomize_actions")) {
    std::seed_seq seq{ParameterValue<int>("seed")};
    std::mt19937 rng(seq);
    for (int i = 0; i < action_mapping_.size(); ++i) {
      action_mapping_[i] = absl::Uniform<int>(rng, 0, 2);
    }
  } else {
    std::fill(action_mapping_.begin(), action_mapping_.end(), true);
  }
}

}  // namespace deep_sea
}  // namespace open_spiel
