// Copyright 2022 DeepMind Technologies Limited
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
#include "open_spiel/games/bargaining.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <numeric>
#include <random>
#include <utility>

#include "open_spiel/abseil-cpp/absl/algorithm/container.h"
#include "open_spiel/abseil-cpp/absl/strings/numbers.h"
#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/abseil-cpp/absl/strings/str_join.h"
#include "open_spiel/abseil-cpp/absl/strings/str_split.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_globals.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/utils/file.h"

namespace open_spiel {
namespace bargaining {

namespace {

// Facts about the game
const GameType kGameType{/*short_name=*/"bargaining",
                         /*long_name=*/"Bargaining",
                         GameType::Dynamics::kSequential,
                         GameType::ChanceMode::kExplicitStochastic,
                         GameType::Information::kImperfectInformation,
                         GameType::Utility::kGeneralSum,
                         GameType::RewardModel::kTerminal,
                         /*max_num_players=*/kNumPlayers,
                         /*min_num_players=*/kNumPlayers,
                         /*provides_information_state_string=*/true,
                         /*provides_information_state_tensor=*/true,
                         /*provides_observation_string=*/true,
                         /*provides_observation_tensor=*/true,
                         /*parameter_specification=*/
                         {{"instances_file", GameParameter("")}}};

static std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::shared_ptr<const Game>(new BargainingGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);
}  // namespace

std::string Instance::ToString() const {
  return absl::StrCat(absl::StrJoin(pool, ","), " ",
                      absl::StrJoin(values[0], ","), " ",
                      absl::StrJoin(values[1], ","));
}

std::string Instance::ToPrettyString() const {
  return absl::StrCat("Pool:    ", absl::StrJoin(pool, " "), "\n",
                      "P0 vals: ", absl::StrJoin(values[0], " "), "\n",
                      "P1 vals: ", absl::StrJoin(values[1], " "), "\n");
}

std::string Offer::ToString() const {
  return absl::StrCat("Offer: ", absl::StrJoin(quantities, " "));
}

std::string BargainingState::ActionToString(Player player,
                                            Action move_id) const {
  return parent_game_->ActionToString(player, move_id);
}

bool BargainingState::IsTerminal() const {
  return agreement_reached_ || offers_.size() >= kMaxTurns;
}

std::vector<double> BargainingState::Returns() const {
  if (agreement_reached_) {
    int proposing_player = (offers_.size() + 1) % kNumPlayers;
    int other_player = 1 - proposing_player;
    std::vector<double> returns(kNumPlayers, 0);
    for (int i = 0; i < kNumItemTypes; ++i) {
      returns[proposing_player] +=
          instance_.values[proposing_player][i] * offers_.back().quantities[i];
      returns[other_player] +=
          instance_.values[other_player][i] *
          (instance_.pool[i] - offers_.back().quantities[i]);
    }
    return returns;
  } else {
    return std::vector<double>(kNumPlayers, 0);
  }
}

std::string BargainingState::ObservationString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);

  if (IsChanceNode()) {
    return "Initial chance node";
  }

  std::string str = absl::StrCat("Pool: ", absl::StrJoin(instance_.pool, " "));
  absl::StrAppend(&str,
                  "\nMy values: ", absl::StrJoin(instance_.values[player], " "),
                  "\n");
  absl::StrAppend(&str, "Agreement reached? ", agreement_reached_, "\n");
  absl::StrAppend(&str, "Number of offers: ", offers_.size(), "\n");
  if (!offers_.empty()) {
    // Only the most recent offer.
    absl::StrAppend(&str, "P", (offers_.size() + 1) % 2,
                    " offers: ", offers_.back().ToString(), "\n");
  }
  return str;
}

std::string BargainingState::InformationStateString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);

  if (IsChanceNode()) {
    return "Initial chance node";
  }

  std::string str = absl::StrCat("Pool: ", absl::StrJoin(instance_.pool, " "));
  absl::StrAppend(&str,
                  "\nMy values: ", absl::StrJoin(instance_.values[player], " "),
                  "\n");
  absl::StrAppend(&str, "Agreement reached? ", agreement_reached_, "\n");
  for (int i = 0; i < offers_.size(); ++i) {
    int proposer = i % 2;
    absl::StrAppend(&str, "P", proposer, " offers: ", offers_[i].ToString(),
                    "\n");
  }
  return str;
}

void BargainingState::ObservationTensor(Player player,
                                        absl::Span<float> values) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  SPIEL_CHECK_EQ(values.size(), game_->ObservationTensorSize());
  std::fill(values.begin(), values.end(), 0);

  if (IsChanceNode()) {
    // No observations at chance nodes.
    return;
  }

  int offset = 0;

  // Agreement reached?
  if (agreement_reached_) {
    values[offset] = 1;
  }
  offset += 1;

  // How many trade offers have happened?
  values[offers_.size()] = 1;
  offset += kMaxTurns + 1;

  // Pool
  for (int i = 0; i < kNumItemTypes; ++i) {
    for (int j = 0; j <= instance_.pool[i]; ++j) {
      values[offset + j] = 1;
    }
    offset += kPoolMaxNumItems + 1;
  }

  // My values
  for (int i = 0; i < kNumItemTypes; ++i) {
    for (int j = 0; j <= instance_.values[player][i]; ++j) {
      values[offset + j] = 1;
    }
    offset += kTotalValueAllItems + 1;
  }

  // Just the last offer
  if (!offers_.empty()) {
    for (int i = 0; i < kNumItemTypes; ++i) {
      for (int j = 0; j <= offers_.back().quantities[i]; ++j) {
        values[offset + j] = 1;
      }
      offset += kPoolMaxNumItems + 1;
    }
  } else {
    offset += (kPoolMaxNumItems + 1) * kNumItemTypes;
  }

  SPIEL_CHECK_EQ(offset, values.size());
}

void BargainingState::InformationStateTensor(Player player,
                                             absl::Span<float> values) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  SPIEL_CHECK_EQ(values.size(), game_->InformationStateTensorSize());
  std::fill(values.begin(), values.end(), 0);

  if (IsChanceNode()) {
    // No observations at chance nodes.
    return;
  }

  int offset = 0;

  // Agreement reached?
  if (agreement_reached_) {
    values[offset] = 1;
  }
  offset += 1;

  // How many trade offers have happened?
  values[offers_.size()] = 1;
  offset += kMaxTurns + 1;

  // Pool
  for (int i = 0; i < kNumItemTypes; ++i) {
    for (int j = 0; j <= instance_.pool[i]; ++j) {
      values[offset + j] = 1;
    }
    offset += kPoolMaxNumItems + 1;
  }

  // My values
  for (int i = 0; i < kNumItemTypes; ++i) {
    for (int j = 0; j <= instance_.values[player][i]; ++j) {
      values[offset + j] = 1;
    }
    offset += kTotalValueAllItems + 1;
  }

  // Offers
  for (int k = 0; k < kMaxTurns; ++k) {
    if (k < offers_.size()) {
      for (int i = 0; i < kNumItemTypes; ++i) {
        for (int j = 0; j <= offers_[k].quantities[i]; ++j) {
          values[offset + j] = 1;
        }
        offset += kPoolMaxNumItems + 1;
      }
    } else {
      offset += (kPoolMaxNumItems + 1) * kNumItemTypes;
    }
  }

  SPIEL_CHECK_EQ(offset, values.size());
}

BargainingState::BargainingState(std::shared_ptr<const Game> game)
    : State(game),
      cur_player_(kChancePlayerId),
      agreement_reached_(false),
      parent_game_(down_cast<const BargainingGame*>(game.get())) {}

int BargainingState::CurrentPlayer() const {
  return IsTerminal() ? kTerminalPlayerId : cur_player_;
}

void BargainingState::DoApplyAction(Action action) {
  if (IsChanceNode()) {
    instance_ = parent_game_->GetInstance(action);
    cur_player_ = 0;
  } else {
    const std::vector<Offer>& all_offers = parent_game_->AllOffers();
    if (action < all_offers.size()) {
      offers_.push_back(all_offers.at(action));
      cur_player_ = 1 - cur_player_;
    } else if (action == all_offers.size()) {
      // Agree action.
      agreement_reached_ = true;
    }
  }
}

bool BargainingState::IsLegalOffer(const Offer& offer) const {
  // An offer is legal if it's a proper subset of the current pool.
  for (int i = 0; i < kNumItemTypes; ++i) {
    if (offer.quantities[i] > instance_.pool[i]) {
      return false;
    }
  }
  return true;
}

std::vector<Action> BargainingState::LegalActions() const {
  if (IsChanceNode()) {
    return LegalChanceOutcomes();
  } else if (IsTerminal()) {
    return {};
  } else {
    const std::vector<Offer>& all_offers = parent_game_->AllOffers();
    std::vector<Action> legal_actions;
    for (int i = 0; i < all_offers.size(); ++i) {
      if (IsLegalOffer(all_offers.at(i))) {
        legal_actions.push_back(i);
      }
    }
    // Add the agree action if there's at least one offer on the table.
    if (!offers_.empty()) {
      legal_actions.push_back(all_offers.size());
    }
    return legal_actions;
  }
}

std::vector<std::pair<Action, double>> BargainingState::ChanceOutcomes() const {
  SPIEL_CHECK_TRUE(IsChanceNode());
  std::vector<std::pair<Action, double>> outcomes;
  const int num_boards = parent_game_->AllInstances().size();
  outcomes.reserve(num_boards);
  double uniform_prob = 1.0 / num_boards;
  for (int i = 0; i < num_boards; ++i) {
    outcomes.push_back({i, uniform_prob});
  }
  return outcomes;
}

std::string BargainingState::ToString() const {
  if (IsChanceNode()) {
    return "Initial chance node";
  }

  std::string str = instance_.ToPrettyString();
  absl::StrAppend(&str, "Agreement reached? ", agreement_reached_, "\n");
  for (int i = 0; i < offers_.size(); ++i) {
    int proposer = i % 2;
    absl::StrAppend(&str, "P", proposer, " offers: ", offers_[i].ToString(),
                    "\n");
  }
  return str;
}

std::unique_ptr<State> BargainingState::Clone() const {
  return std::unique_ptr<State>(new BargainingState(*this));
}

void BargainingGame::ParseInstancesFile(const std::string& filename) {
  open_spiel::file::File infile(filename, "r");
  std::string contents = infile.ReadContents();
  ParseInstancesString(contents);
}

void BargainingGame::ParseInstancesString(const std::string& instances_string) {
  std::vector<std::string> lines = absl::StrSplit(instances_string, '\n');
  SPIEL_CHECK_GT(lines.size(), 1);
  for (const std::string& line : lines) {
    if (!line.empty()) {
      std::vector<std::string> parts = absl::StrSplit(line, ' ');
      SPIEL_CHECK_EQ(parts.size(), kNumItemTypes);
      Instance instance;
      // pool
      std::vector<std::string> pool_parts = absl::StrSplit(parts[0], ',');
      for (int i = 0; i < kNumItemTypes; ++i) {
        SPIEL_CHECK_TRUE(absl::SimpleAtoi(pool_parts[i], &instance.pool[i]));
      }
      // p1 values
      std::vector<std::string> p1values_parts = absl::StrSplit(parts[1], ',');
      for (int i = 0; i < kNumItemTypes; ++i) {
        SPIEL_CHECK_TRUE(
            absl::SimpleAtoi(p1values_parts[i], &instance.values[0][i]));
      }
      // p2 values
      std::vector<std::string> p2values_parts = absl::StrSplit(parts[2], ',');
      for (int i = 0; i < kNumItemTypes; ++i) {
        SPIEL_CHECK_TRUE(
            absl::SimpleAtoi(p2values_parts[i], &instance.values[1][i]));
      }
      all_instances_.push_back(instance);
    }
  }
}

void BargainingGame::CreateOffers() {
  std::vector<int> cur_offer(kNumItemTypes, 0);
  bool done = false;
  do {
    if (std::accumulate(cur_offer.begin(), cur_offer.end(), 0) <=
        kPoolMaxNumItems) {
      all_offers_.push_back(Offer(cur_offer));
    }

    // Try adding a digit to the left-most, keep going until you can. Then
    // set everything to the left of it to 0.
    done = true;
    for (int i = 0; i < kNumItemTypes; ++i) {
      if (cur_offer[i] < kPoolMaxNumItems) {
        done = false;
        cur_offer[i]++;
        for (int j = i - 1; j >= 0; j--) {
          cur_offer[j] = 0;
        }
        break;
      }
    }
  } while (!done);
}

BargainingGame::BargainingGame(const GameParameters& params)
    : Game(kGameType, params) {
  std::string filename = ParameterValue<std::string>("instances_file", "");
  if (!filename.empty()) {
    ParseInstancesFile(filename);
  } else {
    ParseInstancesString(kDefaultInstancesString);
  }
  CreateOffers();
}

std::string BargainingGame::ActionToString(Player player,
                                           Action move_id) const {
  if (player == kChancePlayerId) {
    return absl::StrCat("Chance outcome ", move_id);
  } else if (move_id < all_offers_.size()) {
    return all_offers_[move_id].ToString();
  } else {
    SPIEL_CHECK_EQ(move_id, all_offers_.size());
    return "Agree";
  }
}

int BargainingGame::NumDistinctActions() const {
  // All offers + agree.
  return all_offers_.size() + 1;
}

std::vector<int> BargainingGame::ObservationTensorShape() const {
  return {
      1 +                                       // Agreement reached?
      kMaxTurns + 1 +                           // How many offers have happened
      (kPoolMaxNumItems + 1) * kNumItemTypes +  // Pool
      (kTotalValueAllItems + 1) * kNumItemTypes +  // My values
      (kPoolMaxNumItems + 1) * kNumItemTypes       // Most recent offer
  };
}

std::vector<int> BargainingGame::InformationStateTensorShape() const {
  return {
      1 +                                       // Agreement reached?
      kMaxTurns + 1 +                           // How many offers have happened
      (kPoolMaxNumItems + 1) * kNumItemTypes +  // Pool
      (kTotalValueAllItems + 1) * kNumItemTypes +         // My values
      kMaxTurns * (kPoolMaxNumItems + 1) * kNumItemTypes  // Offers
  };
}

}  // namespace bargaining
}  // namespace open_spiel
