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

#include "open_spiel/games/oshi_zumo.h"

#include <algorithm>
#include <memory>
#include <utility>

#include "open_spiel/game_parameters.h"
#include "open_spiel/spiel.h"

namespace open_spiel {
namespace oshi_zumo {
namespace {

constexpr char kBoundaryPos = '#';
constexpr char kOpenPos = '.';
constexpr char kWrestler = 'W';

// Default parameters.
constexpr int kNoWinner = -1;
constexpr int kDefaultHorizon = 1000;
constexpr int kDefaultCoins = 50;
constexpr int kDefaultSize = 3;
constexpr bool kDefaultAlesia = false;
constexpr int kDefaultMinBid = 0;

const GameType kGameType{/*short_name=*/"oshi_zumo",
                         /*long_name=*/"Oshi Zumo",
                         GameType::Dynamics::kSimultaneous,
                         GameType::ChanceMode::kDeterministic,
                         GameType::Information::kPerfectInformation,
                         GameType::Utility::kZeroSum,
                         GameType::RewardModel::kTerminal,
                         /*max_num_players=*/2,
                         /*min_num_players=*/2,
                         /*provides_information_state_string=*/true,
                         /*provides_information_state_tensor=*/false,
                         /*provides_observation_string=*/true,
                         /*provides_observation_tensor=*/true,
                         /*parameter_specification=*/
                         {{"alesia", GameParameter(kDefaultAlesia)},
                          {"coins", GameParameter(kDefaultCoins)},
                          {"size", GameParameter(kDefaultSize)},
                          {"horizon", GameParameter(kDefaultHorizon)},
                          {"min_bid", GameParameter(kDefaultMinBid)}}};

std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::shared_ptr<const Game>(new OshiZumoGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);
}  // namespace

OshiZumoState::OshiZumoState(std::shared_ptr<const Game> game)
    : SimMoveState(game),
      parent_game_(static_cast<const OshiZumoGame&>(*game)),
      // Fields set to bad values. Use Game::NewInitialState().
      winner_(kNoWinner),
      total_moves_(0),
      horizon_(parent_game_.horizon()),
      starting_coins_(parent_game_.starting_coins()),
      size_(parent_game_.size()),
      alesia_(parent_game_.alesia()),
      min_bid_(parent_game_.min_bid()),
      // pos 0 and pos 2*size_+2 are "off the edge".
      wrestler_pos_(size_ + 1),
      coins_({{starting_coins_, starting_coins_}})

{}

int OshiZumoState::CurrentPlayer() const {
  return IsTerminal() ? kTerminalPlayerId : kSimultaneousPlayerId;
}

void OshiZumoState::DoApplyActions(const std::vector<Action>& actions) {
  SPIEL_CHECK_EQ(actions.size(), 2);
  SPIEL_CHECK_TRUE(actions[0] >= 0);
  SPIEL_CHECK_TRUE(actions[1] >= 0);
  SPIEL_CHECK_TRUE(actions[0] <= coins_[0]);
  SPIEL_CHECK_TRUE(actions[1] <= coins_[1]);

  // Move the wrestler.
  if (actions[0] > actions[1]) {
    wrestler_pos_++;
  } else if (actions[0] < actions[1]) {
    wrestler_pos_--;
  }

  // Remove coins.
  coins_[0] -= actions[0];
  coins_[1] -= actions[1];

  // Check winner.
  if (wrestler_pos_ == 0) {
    winner_ = 1;
  } else if (wrestler_pos_ == (2 * size_ + 2)) {
    winner_ = 0;
  }

  total_moves_++;
}

std::vector<Action> OshiZumoState::LegalActions(Player player) const {
  if (IsTerminal()) return {};
  if (player == kSimultaneousPlayerId) return LegalFlatJointActions();
  SPIEL_CHECK_FALSE(IsChanceNode());
  SPIEL_CHECK_TRUE(player == Player{0} || player == Player{1});

  std::vector<Action> movelist;
  for (int bet = min_bid_; bet <= coins_[player]; bet++) {
    movelist.push_back(bet);
  }

  if (movelist.empty()) {
    // Player does not have the minimum bid: force them to play what they have
    // left.
    movelist.push_back(coins_[player]);
  }

  return movelist;
}

std::string OshiZumoState::ActionToString(Player player,
                                          Action action_id) const {
  if (player == kSimultaneousPlayerId)
    return FlatJointActionToString(action_id);
  SPIEL_CHECK_GE(action_id, 0);
  std::string result = "";
  absl::StrAppend(&result, "[P", player, "]Bid: ", action_id);
  return result;
}

std::string OshiZumoState::ToString() const {
  std::string result = "Coins: ";

  absl::StrAppend(&result, coins_[0]);
  absl::StrAppend(&result, " ");
  absl::StrAppend(&result, coins_[1]);
  absl::StrAppend(&result, ", Field: ");

  for (int p = 0; p <= 2 * size_ + 2; p++) {
    if (p == wrestler_pos_) {
      result += kWrestler;
    } else if (p == 0 || p == (2 * size_ + 2)) {
      result += kBoundaryPos;
    } else {
      result += kOpenPos;
    }
  }

  absl::StrAppend(&result, "\n");
  return result;
}

bool OshiZumoState::IsTerminal() const {
  return (total_moves_ >= horizon_ || winner_ != kNoWinner ||
          (coins_[0] == 0 && coins_[1] == 0));
}

std::vector<double> OshiZumoState::Returns() const {
  if (!IsTerminal()) {
    return {0.0, 0.0};
  }

  if (winner_ == 0) {
    return {1.0, -1.0};
  } else if (winner_ == 1) {
    return {-1.0, 1.0};
  } else {
    // Wrestler not off the edge.
    if (alesia_) {
      return {0.0, 0.0};
    } else if (wrestler_pos_ > (size_ + 1)) {
      return {1.0, -1.0};
    } else if (wrestler_pos_ < (size_ + 1)) {
      return {-1.0, 1.0};
    } else {
      return {0.0, 0.0};
    }
  }
}

std::string OshiZumoState::InformationStateString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  return HistoryString();  // All the information is public.
}

std::string OshiZumoState::ObservationString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  return ToString();  // All the information is public.
}

void OshiZumoState::ObservationTensor(Player player,
                                      absl::Span<float> values) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);

  SPIEL_CHECK_EQ(values.size(), parent_game_.ObservationTensorShape()[0]);
  std::fill(values.begin(), values.end(), 0.);

  // 1 bit per coin value of player 1. { 0, 1, ... , starting_coins_ }
  // 1 bit per coin value of player 2. { 0, 1, ... , starting_coins_ }
  // 1 bit per position of the field. { 0, 1, ... , 2*size_+2 }

  int offset = 0;
  values[offset + coins_[0]] = 1;

  offset += (starting_coins_ + 1);
  values[offset + coins_[1]] = 1;

  offset += (starting_coins_ + 1);
  values[offset + wrestler_pos_] = 1;
}

std::unique_ptr<State> OshiZumoState::Clone() const {
  return std::unique_ptr<State>(new OshiZumoState(*this));
}

OshiZumoGame::OshiZumoGame(const GameParameters& params)
    : Game(kGameType, params),
      horizon_(ParameterValue<int>("horizon")),
      starting_coins_(ParameterValue<int>("coins")),
      size_(ParameterValue<int>("size")),
      alesia_(ParameterValue<bool>("alesia")),
      min_bid_(ParameterValue<int>("min_bid")) {
  SPIEL_CHECK_GE(min_bid_, 0);
  SPIEL_CHECK_LE(min_bid_, starting_coins_);
}

std::unique_ptr<State> OshiZumoGame::NewInitialState() const {
  return std::unique_ptr<State>(new OshiZumoState(shared_from_this()));
}

std::vector<int> OshiZumoGame::ObservationTensorShape() const {
  // 1 bit per coin value of player 1. { 0, 1, ..., starting_coins_ }
  // 1 bit per coin value of player 2. { 0, 1, ..., starting_coins_ }
  // 1 bit per position of the field. { 0, 1, ... , 2*size_+2 }
  return {(2 * (starting_coins_ + 1)) + (2 * size_ + 3)};
}

}  // namespace oshi_zumo
}  // namespace open_spiel
