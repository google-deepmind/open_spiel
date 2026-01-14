// Copyright 2019 DeepMind Technologies Ltd. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "open_spiel/games/mfg/dynamic_routing.h"

#include <algorithm>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "open_spiel/abseil-cpp/absl/container/btree_set.h"
#include "open_spiel/abseil-cpp/absl/container/flat_hash_map.h"
#include "open_spiel/abseil-cpp/absl/memory/memory.h"
#include "open_spiel/abseil-cpp/absl/strings/numbers.h"
#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/abseil-cpp/absl/strings/str_format.h"
#include "open_spiel/abseil-cpp/absl/strings/str_split.h"
#include "open_spiel/game_parameters.h"
#include "open_spiel/games/dynamic_routing/dynamic_routing_data.h"
#include "open_spiel/games/dynamic_routing/dynamic_routing_utils.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_globals.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel::dynamic_routing {

namespace {

inline constexpr double kEpsilon = 1e-4;

const GameType kGameType{
    /*short_name=*/"mfg_dynamic_routing",
    /*long_name=*/"Cpp Mean Field Dynamic Routing",
    GameType::Dynamics::kMeanField,
    GameType::ChanceMode::kExplicitStochastic,
    GameType::Information::kPerfectInformation,
    GameType::Utility::kGeneralSum,
    GameType::RewardModel::kRewards,
    /*max_num_players=*/kNumPlayers,
    /*min_num_players=*/kNumPlayers,
    /*provides_information_state_string=*/true,
    /*provides_information_state_tensor=*/false,
    /*provides_observation_string=*/true,
    /*provides_observation_tensor=*/false,
    /*parameter_specification=*/
    {{"max_num_time_step", GameParameter(10)},
     {"time_step_length", GameParameter(kDefaultTimeStepLength)},
     {"players", GameParameter(-1)},
     {"network", GameParameter(kDefaultNetworkName)},
     {"perform_sanity_checks", GameParameter(true)}},
    /*default_loadable*/ true,
    /*provides_factored_observation_string*/ true};

std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::shared_ptr<const Game>(new MeanFieldRoutingGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);

RegisterSingleTensorObserver single_tensor(kGameType.short_name);

}  // namespace

MeanFieldRoutingGame::MeanFieldRoutingGame(const GameParameters& params)
    : Game(kGameType, params) {
  // Maps data name from string to the enum.
  const absl::flat_hash_map<std::string, DynamicRoutingDataName>
      data_name_string_to_enum = {{"line", DynamicRoutingDataName::kLine},
                                  {"braess", DynamicRoutingDataName::kBraess}};

  int max_num_time_step =
      ParameterValue<int>("max_num_time_step", kDefaultMaxTimeStep);
  SPIEL_CHECK_NE(max_num_time_step, 0);
  time_step_length_ =
      ParameterValue<double>("time_step_length", kDefaultTimeStepLength);
  network_name_ = ParameterValue<std::string>("network", kDefaultNetworkName);
  SPIEL_CHECK_NE(network_name_, "");
  perform_sanity_checks_ = ParameterValue<bool>("perform_sanity_checks", true);
  std::unique_ptr<DynamicRoutingData> data =
      DynamicRoutingData::Create(data_name_string_to_enum.at(network_name_));
  network_ = std::move(data->network_);
  od_demand_ = std::move(data->od_demand_);
  network_->CheckListOfOdDemandIsCorrect(od_demand_.get());
  game_info_.num_distinct_actions = network_->num_actions();
  game_info_.max_chance_outcomes = static_cast<int>(od_demand_->size());
  game_info_.num_players = kNumPlayers;
  game_info_.min_utility = static_cast<double>(-max_num_time_step - 1);
  game_info_.max_utility = 0;
  game_info_.max_game_length = max_num_time_step;
}

std::unique_ptr<State> MeanFieldRoutingGame::DeserializeState(
    const std::string& str) const {
  std::vector<std::string> properties = absl::StrSplit(str, ',');
  if (properties.size() != 10) {
    SpielFatalError(
        absl::StrCat("Expected 10 properties for serialized state, got: ",
                     properties.size()));
  }
  int current_time_step;
  int player_id;
  bool is_chance_init, is_terminal, vehicle_at_destination,
      vehicle_without_legal_action;
  int waiting_time;
  double vehicle_final_travel_time;
  SPIEL_CHECK_TRUE(absl::SimpleAtoi(properties[0], &current_time_step));
  SPIEL_CHECK_TRUE(absl::SimpleAtoi(properties[1], &player_id));
  SPIEL_CHECK_TRUE(absl::SimpleAtob(properties[2], &is_chance_init));
  SPIEL_CHECK_TRUE(absl::SimpleAtob(properties[3], &is_terminal));
  SPIEL_CHECK_TRUE(absl::SimpleAtob(properties[4], &vehicle_at_destination));
  SPIEL_CHECK_TRUE(
      absl::SimpleAtob(properties[5], &vehicle_without_legal_action));
  SPIEL_CHECK_TRUE(absl::SimpleAtoi(properties[6], &waiting_time));
  SPIEL_CHECK_TRUE(absl::SimpleAtod(properties[7], &vehicle_final_travel_time));
  std::string vehicle_location(properties[8]),
      vehicle_destination(properties[9]);
  return MeanFieldRoutingGameState::Create(
      shared_from_this(), time_step_length_, od_demand_.get(), network_.get(),
      perform_sanity_checks_, current_time_step, player_id, is_chance_init,
      is_terminal, vehicle_at_destination, vehicle_without_legal_action,
      waiting_time, vehicle_final_travel_time, vehicle_location,
      vehicle_destination);
}

std::unique_ptr<MeanFieldRoutingGameState> MeanFieldRoutingGameState::Create(
    std::shared_ptr<const Game> game, double time_step_length,
    std::vector<OriginDestinationDemand>* od_demand, Network* network,
    bool perform_sanity_checks, int current_time_step,
    int player_id, bool is_chance_init, bool is_terminal,
    bool vehicle_at_destination, bool vehicle_without_legal_action,
    int waiting_time, double vehicle_final_travel_time,
    std::string vehicle_location, std::string vehicle_destination) {
  double total_num_vehicle = 0;
  for (const OriginDestinationDemand& od_demand_item : *od_demand) {
    total_num_vehicle += od_demand_item.counts;
  }
  int i = 0;
  ActionsAndProbs chance_outcomes;
  for (const auto& od_demand_item : *od_demand) {
    chance_outcomes.emplace_back(
        std::pair(i++, od_demand_item.counts / total_num_vehicle));
  }
  return absl::WrapUnique<MeanFieldRoutingGameState>(
      new MeanFieldRoutingGameState(
          game, time_step_length, od_demand, network, perform_sanity_checks,
          current_time_step, player_id, is_chance_init, is_terminal,
          vehicle_at_destination, vehicle_without_legal_action, waiting_time,
          vehicle_final_travel_time, vehicle_location, vehicle_destination,
          total_num_vehicle, chance_outcomes));
}

std::unique_ptr<MeanFieldRoutingGameState>
MeanFieldRoutingGameState::CreateNewInitialState(
    std::shared_ptr<const Game> game, double time_step_length,
    std::vector<OriginDestinationDemand>* od_demand, Network* network,
    bool perform_sanity_checks) {
  return MeanFieldRoutingGameState::Create(
      game, time_step_length, od_demand, network, perform_sanity_checks,
      /* current_time_step= */ 0,
      /* player_id = */ kChancePlayerId,
      /* is_chance_init = */ true,
      /* is_terminal = */ false,
      /* vehicle_at_destination = */ false,
      /* vehicle_without_legal_action = */ false,
      /* waiting_time = */ kWaitingTimeNotAssigned,
      /* vehicle_final_travel_time = */ 0.0,
      /* vehicle_location = */ "",
      /* vehicle_destination = */ "");
}

MeanFieldRoutingGameState::MeanFieldRoutingGameState(
    std::shared_ptr<const Game> game, double time_step_length,
    std::vector<OriginDestinationDemand>* od_demand, Network* network,
    bool perform_sanity_checks, int current_time_step,
    int player_id, bool is_chance_init, bool is_terminal,
    bool vehicle_at_destination, bool vehicle_without_legal_action,
    int waiting_time, double vehicle_final_travel_time,
    std::string vehicle_location, std::string vehicle_destination,
    double total_num_vehicle, const ActionsAndProbs& chance_outcomes)
    : State(game),
      current_time_step_(current_time_step),
      current_player_id_(player_id),
      is_chance_init_(is_chance_init),
      is_terminal_(is_terminal),
      vehicle_at_destination_(vehicle_at_destination),
      vehicle_without_legal_action_(vehicle_without_legal_action),
      waiting_time_(waiting_time),
      vehicle_final_travel_time_(vehicle_final_travel_time),
      vehicle_location_(vehicle_location),
      vehicle_destination_(vehicle_destination),
      time_step_length_(time_step_length),
      max_travel_time_(game->MaxGameLength()),
      perform_sanity_checks_(perform_sanity_checks),
      od_demand_(od_demand),
      network_(network),
      total_num_vehicle_(total_num_vehicle),
      chance_outcomes_(chance_outcomes) {}

std::string MeanFieldRoutingGameState::StateToString(
    std::string location, int time_step, int player_id, int waiting_time,
    std::string destination, double ret) const {
  std::string time;
  if (destination.empty()) {
    destination = vehicle_destination_;
  }
  if (is_chance_init_) {
    return "initial chance node";
  }
  if (player_id == kDefaultPlayerId ||
      player_id == kTerminalPlayerId) {
    time = absl::StrCat(time_step);
  } else if (player_id == kMeanFieldPlayerId) {
    time = absl::StrFormat("%d_mean_field", time_step);
  } else if (player_id == kChancePlayerId) {
    time = absl::StrFormat("%d_chance", time_step);
  } else {
    SpielFatalError(
        "Player id should be DEFAULT_PLAYER_ID, MEAN_FIELD or CHANCE");
  }
  if (vehicle_final_travel_time_ != 0.0) {
    return absl::StrFormat("Arrived at %s, with arrival time %.2f, t=%s",
                           location, vehicle_final_travel_time_, time);
  }
  return absl::StrFormat("Location=%s, waiting time=%d, t=%s, destination=%s",
                         location, waiting_time, time, destination);
}

std::vector<Action> MeanFieldRoutingGameState::LegalActions() const {
  if (is_terminal_) {
    return {};
  }
  SPIEL_CHECK_NE(CurrentPlayer(), kMeanFieldPlayerId);
  if (CurrentPlayer() == kChancePlayerId) {
    return LegalChanceOutcomes();
  }
  if (perform_sanity_checks_) {
    SPIEL_CHECK_EQ(CurrentPlayer(), kDefaultPlayerId);
  }
  if (waiting_time_ > 0) {
    return {kNoPossibleAction};
  }
  if (vehicle_without_legal_action_) {
    return {kNoPossibleAction};
  }
  std::string end_section_node = NodesFromRoadSection(vehicle_location_)[1];
  std::vector<std::string> successors =
      network_->GetSuccessors(end_section_node);
  if (perform_sanity_checks_) {
    SPIEL_CHECK_TRUE(!successors.empty());
  }
  std::vector<Action> actions;
  for (const auto& d : successors) {
    Action action = network_->GetActionIdFromMovement(end_section_node, d);
    network_->AssertValidAction(action);
    actions.push_back(action);
  }
  std::sort(actions.begin(), actions.end());
  return actions;
}

void MeanFieldRoutingGameState::DoApplyAction(Action action) {
  if (perform_sanity_checks_) {
    SPIEL_CHECK_TRUE(!IsTerminal());
    SPIEL_CHECK_NE(current_player_id_, kMeanFieldPlayerId);
  }
  switch (current_player_id_) {
    case kChancePlayerId: {
      current_player_id_ = kDefaultPlayerId;
      SPIEL_CHECK_EQ(is_chance_init_, true);
      auto od_demand = od_demand_->at(action);
      vehicle_destination_ = od_demand.vehicle.destination;
      vehicle_location_ = od_demand.vehicle.origin;
      waiting_time_ = static_cast<int>(od_demand.vehicle.departure_time /
                                       time_step_length_);
      is_chance_init_ = false;
      break;
    }
    case kDefaultPlayerId: {
      current_player_id_ = kMeanFieldPlayerId;
      if (!vehicle_without_legal_action_) {
        if (waiting_time_ > 0) {
          waiting_time_ -= 1;
        } else {
          if (perform_sanity_checks_) {
            network_->AssertValidAction(action, vehicle_location_);
          }
          vehicle_location_ = network_->GetRoadSectionFromActionId(action);
          if (vehicle_location_ == vehicle_destination_) {
            vehicle_final_travel_time_ = current_time_step_;
            vehicle_at_destination_ = true;
            vehicle_without_legal_action_ = true;
          } else if (network_->IsLocationASinkNode(vehicle_location_)) {
            vehicle_without_legal_action_ = true;
            vehicle_final_travel_time_ = -1 * GetGame()->MinUtility();
          } else {
            waiting_time_ = kWaitingTimeNotAssigned;
          }
        }
      }
      current_time_step_ += 1;
      break;
    }
    default:
      SpielFatalError(absl::StrCat("Unsupported Player ID in DoApplyAction(): ",
                                   current_player_id_));
  }

  if (current_time_step_ >= GetGame()->MaxGameLength()) {
    is_terminal_ = true;
    current_player_id_ = kTerminalPlayerId;
    if (!vehicle_at_destination_) {
      vehicle_final_travel_time_ = -1 * GetGame()->MinUtility();
    }
  }
}

std::string MeanFieldRoutingGameState::ActionToString(Player player,
                                                      Action action) const {
  SPIEL_CHECK_NE(player, kMeanFieldPlayerId);
  if (player == kChancePlayerId) {
    SPIEL_CHECK_TRUE(is_chance_init_);
    return absl::StrFormat("Vehicle is assigned to population %d", action);
  }
  if (perform_sanity_checks_) {
    SPIEL_CHECK_EQ(player, kDefaultPlayerId);
  }

  if (action == kNoPossibleAction) {
    return absl::StrFormat("Vehicle %d reach a sink node or its destination.",
                           player);
  }
  if (perform_sanity_checks_) {
    network_->AssertValidAction(action);
  }
  return absl::StrFormat("Vehicle %d would like to move to %s.", player,
                         network_->GetRoadSectionFromActionId(action));
}

Action MeanFieldRoutingGameState::GetLocationAsActionInt() const {
  return network_->GetRoadSectionAsInt(vehicle_location_);
}

Action MeanFieldRoutingGameState::GetDestinationAsActionInt() const {
  return network_->GetRoadSectionAsInt(vehicle_destination_);
}

int MeanFieldRoutingGameState::CurrentTimeStamp() const {
  return current_time_step_;
}

int MeanFieldRoutingGameState::CurrentPlayer() const {
  return current_player_id_;
}

bool MeanFieldRoutingGameState::IsTerminal() const { return is_terminal_; }

bool MeanFieldRoutingGameState::IsWaiting() const { return waiting_time_ > 0; }

const Network* MeanFieldRoutingGameState::network() const { return network_; }

std::vector<double> MeanFieldRoutingGameState::Returns() const {
  if (!IsTerminal()) {
    return std::vector<double>{0};
  }
  double ret = -vehicle_final_travel_time_ * time_step_length_;
  return std::vector<double>{ret};
}

std::vector<std::string> MeanFieldRoutingGameState::DistributionSupport() {
  if (vehicle_without_legal_action_) {
    return {};
  }
  std::vector<std::string> dist;
  for (int waiting_time = kWaitingTimeNotAssigned;
       waiting_time < max_travel_time_; waiting_time++) {
    for (const auto& od : *(od_demand_)) {
      std::string destination = od.vehicle.destination;
      std::string value =
          StateToString(vehicle_location_, current_time_step_,
                        kMeanFieldPlayerId, waiting_time, destination,
                        /*ret = */ 0.0);
      dist.push_back(value);
    }
  }
  absl::btree_set<std::string> dist_set(dist.begin(), dist.end());
  SPIEL_CHECK_EQ(dist_set.size(), dist.size());
  return dist;
}

void MeanFieldRoutingGameState::UpdateDistribution(
    const std::vector<double>& distribution) {
  if (current_player_id_ == kTerminalPlayerId) {
    return;
  }
  if (perform_sanity_checks_) {
    SPIEL_CHECK_EQ(current_player_id_, kMeanFieldPlayerId);
  }
  current_player_id_ = kDefaultPlayerId;

  if (!vehicle_without_legal_action_) {
    double normed_density_on_vehicle_link = 0;
    for (const double& d : distribution) {
      normed_density_on_vehicle_link += d;
    }
    if (perform_sanity_checks_) {
      SPIEL_CHECK_GE(normed_density_on_vehicle_link, 0);
      SPIEL_CHECK_LE(normed_density_on_vehicle_link, 1 + kEpsilon);
    }
    if (waiting_time_ == kWaitingTimeNotAssigned) {
      double volume = total_num_vehicle_ * normed_density_on_vehicle_link;
      waiting_time_ =
          static_cast<int>(network_->GetTravelTime(vehicle_location_, volume) /
                           time_step_length_) -
          1;
      waiting_time_ = std::max(0, waiting_time_);
    }
  }
}

ActionsAndProbs MeanFieldRoutingGameState::ChanceOutcomes() const {
  SPIEL_CHECK_NE(current_player_id_, kMeanFieldPlayerId);
  if (perform_sanity_checks_) {
    SPIEL_CHECK_EQ(current_player_id_, kChancePlayerId);
    SPIEL_CHECK_TRUE(is_chance_init_);
  }
  return chance_outcomes_;
}

std::unique_ptr<State> MeanFieldRoutingGameState::Clone() const {
  return absl::make_unique<MeanFieldRoutingGameState>(*this);
}

std::string MeanFieldRoutingGameState::Serialize() const {
  return absl::StrCat(current_time_step_, ",", current_player_id_, ",",
                      is_chance_init_, ",", is_terminal_, ",",
                      vehicle_at_destination_, ",",
                      vehicle_without_legal_action_, ",", waiting_time_, ",",
                      vehicle_final_travel_time_, ",", vehicle_location_, ",",
                      vehicle_destination_);
}

std::string MeanFieldRoutingGameState::ToString() const {
  if (!vehicle_location_.empty()) {
    return StateToString(vehicle_location_, current_time_step_,
                         current_player_id_, waiting_time_,
                         vehicle_destination_, Returns()[0]);
  }
  SPIEL_CHECK_EQ(current_time_step_, 0);
  return "Before initial chance node.";
}

}  // namespace open_spiel::dynamic_routing
