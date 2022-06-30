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

#include "open_spiel/games/blotto.h"

#include <set>

#include "open_spiel/game_parameters.h"
#include "open_spiel/spiel.h"

namespace open_spiel {
namespace blotto {

constexpr const int kDefaultNumCoins = 10;
constexpr const int kDefaultNumFields = 3;
constexpr const int kDefaultNumPlayers = 2;

namespace {

const GameType kGameType{/*short_name=*/"blotto",
                         /*long_name=*/"Blotto",
                         GameType::Dynamics::kSimultaneous,
                         GameType::ChanceMode::kDeterministic,
                         GameType::Information::kOneShot,
                         GameType::Utility::kZeroSum,
                         GameType::RewardModel::kTerminal,
                         /*max_num_players=*/10,
                         /*min_num_players=*/2,
                         /*provides_information_state_string=*/true,
                         /*provides_information_state_tensor=*/true,
                         /*provides_observation_string=*/true,
                         /*provides_observation_tensor=*/true,
                         /*parameter_specification=*/
                         {{"coins", GameParameter(kDefaultNumCoins)},
                          {"fields", GameParameter(kDefaultNumFields)},
                          {"players", GameParameter(kDefaultNumPlayers)}}};

std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::shared_ptr<const Game>(new BlottoGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);
}  // namespace

BlottoState::BlottoState(std::shared_ptr<const Game> game, int coins,
                         int fields, const ActionMap* action_map,
                         const std::vector<Action>* legal_actions)
    : NFGState(game),
      coins_(coins),
      fields_(fields),
      joint_action_({}),
      action_map_(action_map),
      legal_actions_(legal_actions),
      returns_({}) {}

void BlottoState::DoApplyActions(const std::vector<Action>& actions) {
  joint_action_ = actions;

  // Now determine returns.
  returns_.resize(num_players_, 0);
  std::vector<int> scores(num_players_, 0);
  std::vector<std::vector<int>> player_actions;

  for (int f = 0; f < fields_; ++f) {
    int num_winners = 0;
    int winner = 0;
    int max_value = -1;

    for (auto p = Player{0}; p < num_players_; ++p) {
      // Get the expanded action if necessary.
      if (p >= player_actions.size()) {
        player_actions.push_back(action_map_->at(joint_action_[p]));
      }

      if (player_actions[p][f] > max_value) {
        num_winners = 1;
        winner = p;
        max_value = player_actions[p][f];
      } else if (player_actions[p][f] == max_value) {
        num_winners++;
      }
    }

    // Give the winner of this field one point. Draw if tied.
    if (num_winners == 1) {
      scores[winner]++;
    }
  }

  // Find the global winner(s).
  std::set<int> winners;
  int max_points = 0;
  for (auto p = Player{0}; p < num_players_; ++p) {
    if (scores[p] > max_points) {
      max_points = scores[p];
      winners = {p};
    } else if (scores[p] == max_points) {
      winners.insert(p);
    }
  }

  // Finally, assign returns. Each winner gets 1/num_winners, each loser gets
  // -1 / num_losers.
  for (auto p = Player{0}; p < num_players_; ++p) {
    if (winners.size() == num_players_) {
      // All players won same number of fields. Draw.
      returns_[p] = 0;
    } else if (winners.find(p) != winners.end()) {
      SPIEL_CHECK_GE(winners.size(), 1);
      returns_[p] = 1.0 / winners.size();
    } else {
      SPIEL_CHECK_GE(num_players_ - winners.size(), 1);
      returns_[p] = -1.0 / (num_players_ - winners.size());
    }
  }
}

std::vector<Action> BlottoState::LegalActions(Player player) const {
  if (IsTerminal()) return {};
  return (*legal_actions_);
}

std::string BlottoState::ActionToString(Player player, Action move_id) const {
  return "[" + absl::StrJoin(action_map_->at(move_id), ",") + "]";
}

std::string BlottoState::ToString() const {
  std::string str = "";
  absl::StrAppend(&str, "Terminal? ", IsTerminal(), "\n");
  for (int p = 0; p < joint_action_.size(); ++p) {
    absl::StrAppend(&str, "P", p,
                    " action: ", ActionToString(p, joint_action_[p]), "\n");
  }
  return str;
}

bool BlottoState::IsTerminal() const { return !joint_action_.empty(); }

std::vector<double> BlottoState::Returns() const {
  return IsTerminal() ? returns_ : std::vector<double>(num_players_, 0.);
}

std::unique_ptr<State> BlottoState::Clone() const {
  return std::unique_ptr<State>(new BlottoState(*this));
}

int BlottoGame::NumDistinctActions() const { return num_distinct_actions_; }

void BlottoGame::CreateActionMapRec(int* count, int coins_left,
                                    const std::vector<int>& action) {
  if (action.size() == fields_) {
    if (coins_left == 0) {
      // All coins used, valid move.
      (*action_map_)[*count] = action;
      (*count)++;
      return;
    } else {
      // Not all coins used, invalid move.
      return;
    }
  } else {
    for (int num_coins = 0; num_coins <= coins_left; ++num_coins) {
      std::vector<int> new_action = action;
      new_action.push_back(num_coins);
      CreateActionMapRec(count, coins_left - num_coins, new_action);
    }
  }
}

BlottoGame::BlottoGame(const GameParameters& params)
    : NormalFormGame(kGameType, params),
      num_distinct_actions_(0),  // Set properly after CreateActionMap.
      coins_(ParameterValue<int>("coins")),
      fields_(ParameterValue<int>("fields")),
      players_(ParameterValue<int>("players")) {
  action_map_.reset(new ActionMap());
  CreateActionMapRec(&num_distinct_actions_, coins_, {});

  // The action set is static for all states, so create it only once.
  legal_actions_.reset(new std::vector<Action>(num_distinct_actions_));
  for (Action action = 0; action < num_distinct_actions_; ++action) {
    (*legal_actions_)[action] = action;
  }
}

}  // namespace blotto
}  // namespace open_spiel
