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

#include "open_spiel/games/first_sealed_auction.h"

#include <algorithm>
#include <memory>
#include <utility>

namespace open_spiel {
namespace first_sealed_auction {
namespace {

// Facts about the game
const GameType kGameType{/*short_name=*/"first_sealed_auction",
                         /*long_name=*/"First-Price Sealed-Bid Auction",
                         GameType::Dynamics::kSequential,
                         GameType::ChanceMode::kExplicitStochastic,
                         GameType::Information::kImperfectInformation,
                         GameType::Utility::kGeneralSum,
                         GameType::RewardModel::kTerminal,
                         /*max_num_players=*/10,
                         /*min_num_players=*/2,
                         /*provides_information_state_string=*/true,
                         /*provides_information_state_tensor=*/true,
                         /*provides_observation_string=*/true,
                         /*provides_observation_tensor=*/true,
                         /*parameter_specification=*/
                         {
                             {"players", GameParameter(kDefaultPlayers)},
                             {"max_value", GameParameter(kDefaultMaxValue)},
                         }};

std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::shared_ptr<const Game>(new FPSBAGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);
}  // namespace

FPSBAGame::FPSBAGame(const GameParameters& params)
    : Game(kGameType, params),
      num_players_(ParameterValue<int>("players")),
      max_value_(ParameterValue<int>("max_value")) {}

FPSBAState::FPSBAState(std::shared_ptr<const Game> game)
    : State(game), max_value_(game->NumDistinctActions()) {}

int FPSBAState::CurrentPlayer() const {
  if (valuations_.size() < num_players_) return kChancePlayerId;
  if (bids_.size() < num_players_) return bids_.size();
  if (winner_ == kInvalidPlayer) return kChancePlayerId;
  return kTerminalPlayerId;
}

std::vector<Action> FPSBAState::EligibleWinners() const {
  int max_bid = *std::max_element(bids_.begin(), bids_.end());
  std::vector<Action> eligibles;
  for (auto player = Player{0}; player < num_players_; player++) {
    if (bids_[player] == max_bid) {
      eligibles.push_back(player);
    }
  }
  return eligibles;
}

std::vector<Action> FPSBAState::LegalActions() const {
  if (valuations_.size() < num_players_) {
    std::vector<Action> values(max_value_);
    std::iota(values.begin(), values.end(), 1);
    return values;
  }
  if (bids_.size() < num_players_) {
    std::vector<Action> bids(valuations_[bids_.size()]);
    std::iota(bids.begin(), bids.end(), 0);
    return bids;
  }
  if (winner_ == kInvalidPlayer) {
    return EligibleWinners();
  }
  return {};
}

std::string FPSBAState::ActionToString(Player player, Action action_id) const {
  if (player != kChancePlayerId) {
    return absl::StrCat("Player ", player, " bid: ", action_id);
  } else if (valuations_.size() < num_players_) {
    return absl::StrCat("Player ", valuations_.size(), " value: ", action_id);
  } else {
    return absl::StrCat("Chose winner ", action_id);
  }
}

std::string FPSBAState::ToString() const {
  return absl::StrCat(
      absl::StrJoin(valuations_, ","), ";", absl::StrJoin(bids_, ","),
      winner_ == kInvalidPlayer ? "" : absl::StrCat(";", winner_));
}

bool FPSBAState::IsTerminal() const { return winner_ != kInvalidPlayer; }

std::vector<double> FPSBAState::Returns() const {
  std::vector<double> returns(num_players_);
  if (winner_ != kInvalidPlayer) {
    returns[winner_] = valuations_[winner_] - bids_[winner_];
  }
  return returns;
}

std::unique_ptr<State> FPSBAState::Clone() const {
  return std::unique_ptr<State>(new FPSBAState(*this));
}

void FPSBAState::DoApplyAction(Action action_id) {
  if (valuations_.size() < num_players_) {
    valuations_.push_back(action_id);
  } else if (bids_.size() < num_players_) {
    bids_.push_back(action_id);
  } else if (winner_ == kInvalidPlayer) {
    winner_ = action_id;
  } else {
    SpielFatalError(
        absl::StrCat("Can't apply action in terminal state: ", action_id));
  }
}

std::string FPSBAState::InformationStateString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  if (valuations_.size() <= player) return absl::StrCat("p", player);
  if (bids_.size() <= player)
    return absl::StrCat("p", player, " val ", valuations_[player]);
  return absl::StrCat("p", player, " val ", valuations_[player], " bid ",
                      bids_[player]);
}

void FPSBAState::InformationStateTensor(Player player,
                                        absl::Span<float> values) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  SPIEL_CHECK_EQ(values.size(), 2 * max_value_ + num_players_);
  std::fill(values.begin(), values.end(), 0);
  auto cursor = values.begin();
  cursor[player] = 1;
  cursor += num_players_;
  if (valuations_.size() > player) {
    cursor[valuations_[player] - 1] = 1;
  }
  cursor += max_value_;
  if (bids_.size() > player) {
    cursor[bids_[player]] = 1;
  }
  cursor += max_value_;
  SPIEL_CHECK_EQ(cursor - values.begin(), values.size());
}

std::string FPSBAState::ObservationString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  if (valuations_.size() <= player) return "";
  return absl::StrCat(valuations_[player]);
}

void FPSBAState::ObservationTensor(Player player,
                                   absl::Span<float> values) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  SPIEL_CHECK_EQ(values.size(), max_value_);
  std::fill(values.begin(), values.end(), 0);
  if (valuations_.size() > player) {
    values[valuations_[player] - 1] = 1;
  }
}

ActionsAndProbs FPSBAState::ChanceOutcomes() const {
  ActionsAndProbs valuesAndProbs;
  if (valuations_.size() < num_players_) {
    for (int i = 1; i <= max_value_; i++) {
      valuesAndProbs.push_back(std::make_pair(i, 1. / max_value_));
    }
  } else if (bids_.size() == num_players_ && winner_ == kInvalidPlayer) {
    int max_bid = *std::max_element(bids_.begin(), bids_.end());
    int num_tie = std::count(bids_.begin(), bids_.end(), max_bid);
    for (auto player = Player{0}; player < num_players_; player++) {
      if (bids_[player] == max_bid) {
        valuesAndProbs.push_back(std::make_pair(player, 1. / num_tie));
      }
    }
  } else {
    SpielFatalError("This isn't a chance node");
  }
  return valuesAndProbs;
}

}  // namespace first_sealed_auction
}  // namespace open_spiel
