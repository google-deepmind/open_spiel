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

// Implementation of a mean field routing game.
//
// The game is derived from https://arxiv.org/abs/2110.11943.
// This game is also implemented in python, see
// open_spiel/python/mfg/games/dynamic_routing.py.
// The list of vehicles decribing the N player of the dynamic routing game is
// replaced by a list of OriginDestinationDemand. One OriginDestinationDemand
// corresponds to one population of vehicles (with the same origin, destination
// and departure time).
//
// This game is a variant of the mean field route choice game
// (https://ieeexplore.ieee.org/abstract/document/8619448) as the vehicle
// movement depends on the current network congestion. In the mean field route
// choice game, the number of time steps to reach the destination is constant
// and does not depend on the network congestion, neither of the vehicle cost
// function. In the dynamic driving and routing game
// (https://doi.org/10.1016/j.trc.2021.103189), the vehicle choose its
// speed to travel on each link in order to minimize its cost function.
// Therefore the congestion is encoded in the cost function.
//
// More context can be found on the docstring of the python_dynamic_routing
// class.

#ifndef OPEN_SPIEL_GAMES_MFG_DYNAMIC_ROUTING_H_
#define OPEN_SPIEL_GAMES_MFG_DYNAMIC_ROUTING_H_

#include <memory>
#include <string>
#include <vector>

#include "open_spiel/game_parameters.h"
#include "open_spiel/games/dynamic_routing/dynamic_routing_utils.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_globals.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel::dynamic_routing {

// This mean field game is a 1-population game, so it has only one
// (representative) player type.
inline constexpr int kNumPlayers = 1;
// A player moves to a new link during a decision node, then its waiting
// time is reassigned based on the number of players on the new link during the
// next chance node. Therefore the waiting time is assigned to
// `kWaitingTimeNotAssigned` between the decision node for a player that moves
// and the following chance node.
inline constexpr int kWaitingTimeNotAssigned = -1;
// kDefaultTimeStepLength is used to convert travel times into number of game
// time steps.
inline constexpr double kDefaultTimeStepLength = 1.0;
// Set the default values to pass auto tests with no args.
inline constexpr int kDefaultMaxTimeStep = 10;
inline constexpr const char* kDefaultNetworkName = "braess";

// State of the MeanFieldRoutingGame.
// One player is equal to one representative vehicle.
// See docstring of the MeanFieldRoutingGame class and of the file for more
// information.
class MeanFieldRoutingGameState : public State {
 public:
  static std::unique_ptr<MeanFieldRoutingGameState> CreateNewInitialState(
      std::shared_ptr<const Game> game, double time_step_length,
      std::vector<OriginDestinationDemand>* od_demand, Network* network,
      bool perform_sanity_checks = true);

  // Returns the vehicle location.
  // This will be 1-based action index of the location, or 0 when the location
  // is empty before the initial chance node.
  Action GetLocationAsActionInt() const;

  // Returns the vehicle destination.
  // This will be 1-based action index of the destination, or 0 when the
  // destination is emtpy before the initial chance node.
  Action GetDestinationAsActionInt() const;

  int CurrentTimeStamp() const;
  const Network* network() const;
  bool IsWaiting() const;

  Player CurrentPlayer() const override;
  bool IsTerminal() const override;
  std::vector<double> Returns() const override;
  std::string ActionToString(Player player, Action action) const override;
  std::unique_ptr<State> Clone() const override;
  std::string ToString() const override;
  std::string Serialize() const override;

  // Converts the representation player state to its unique string
  // representation. The string representation will be used in hashmaps for
  // various algorithms that computes the state value, expected return, best
  // response or find the mean field Nash equilibrium. The state of the
  // representative player is uniquely defined by the current time, the type of
  // node (decision, mean field or chance), the vehicle location, its
  // destination and its waiting time.
  // Args:
  //  `is_chance_init`: True if at chance initialization.
  //  `location`: the location of the representative player.
  //  `time_step`: the current time step.
  //  `player_id`: the current node type as a player id.
  //  `waiting_time`: the representative player waiting time.
  //  `destination`: the destination of the representative player.
  std::string StateToString(std::string location, int time_step,
                            int player_id = kDefaultPlayerId,
                            int waiting_time = 0, std::string destination = "",
                            double ret = 0) const;

  // Returns the list of states for which we need to know the distribution of
  // players over to update the current representative player state.
  // The distribution of the vehicle's states is used to determined the number
  // of cars on the new link location link of the representative vehicle in
  // order to define their waiting time of the representative vehicle when they
  // join this link. Therefore, If the representative vehicle does not move at
  // this time step, then no states are useful. If the representative vehicle
  // moves at this time step, then only the states corresponding to be on the
  // new link of the representative vehicle are needed to compute the
  // representative vehicle new waiting time.
  // Returns:
  //   An array of the string representation of all OD_DEMANDs.
  std::vector<std::string> DistributionSupport() override;

  // Updates the travel time from the distribution.
  // Using the distribution `distribution` of vehicles over the states in
  // `DistributionSupport()`, computes the number of cars on the same link as
  // the representative player if they has moved during the last time step and
  // store it internally to assign a new waiting time to the player. If the
  // player has not moved during the last time step, do nothing.
  // Args:
  //   `distribution`: the probability for a vehicle to be in the states in
  //     distribution_support. The distribution is a list of probabilities.
  void UpdateDistribution(const std::vector<double>& distribution) override;

  // On the initial node, returns the initial state probability distribution.
  // One chance outcome correspond to each possible origin, destination,
  // departure time tuple, the probability of each chance outcome is the
  // proportion of the corresponding tuple.
  ActionsAndProbs ChanceOutcomes() const override;

  // Returns an array of legal actions.
  // If the game is finished, if the vehicle is at its destination, has a
  // positive waiting time or if it is on a node without successors then an
  // empty list is returned. Otherwise the list of successors nodes of the
  // current vehicle location is returned.
  std::vector<Action> LegalActions() const override;

  std::string InformationStateString(Player player) const override {
    SPIEL_CHECK_GE(player, 0);
    SPIEL_CHECK_LT(player, num_players_);
    return HistoryString();
  }

  std::string ObservationString(Player player) const override {
    SPIEL_CHECK_GE(player, 0);
    SPIEL_CHECK_LT(player, num_players_);
    return ToString();
  }

 protected:
  // Can be either called on a chance node or on a decision node.
  // If called on the initial chance node, the action gives in which OD
  // demand the representative vehicle belongs too (it put the vehicle at
  // this location and define its destination).
  // If called on decision node, the action defines on which link the vehicle
  // will move (if it is not stuck in traffic) and assign a waiting time to the
  // vehicle.
  void DoApplyAction(Action action) override;

 private:
  static std::unique_ptr<MeanFieldRoutingGameState> Create(
      std::shared_ptr<const Game> game, double time_step_length,
      std::vector<OriginDestinationDemand>* od_demand, Network* network,
      bool perform_sanity_checks, int current_time_step,
      int player_id, bool is_chance_init, bool is_terminal,
      bool vehicle_at_destination, bool vehicle_without_legal_action,
      int waiting_time, double vehicle_final_travel_time,
      std::string vehicle_location, std::string vehicle_destination);

  explicit MeanFieldRoutingGameState(
      std::shared_ptr<const Game> game, double time_step_length,
      std::vector<OriginDestinationDemand>* od_demand, Network* network,
      bool perform_sanity_checks, int current_time_step,
      int player_id, bool is_chance_init, bool is_terminal,
      bool vehicle_at_destination, bool vehicle_without_legal_action,
      int waiting_time, double vehicle_final_travel_time,
      std::string vehicle_location, std::string vehicle_destination,
      double total_num_vehicle, const ActionsAndProbs& chance_outcomes);

  int current_time_step_;
  int current_player_id_;
  bool is_chance_init_;
  bool is_terminal_;
  // Boolean that encodes if the representative vehicle has reached its
  // destination.
  bool vehicle_at_destination_;
  // Boolean that encodes if the representative vehicle has reach a sink node,
  // meaning that it will not be able to move anymore.
  bool vehicle_without_legal_action_;
  // Time that the vehicle has to wait before moving to the next link (equal to
  // the link travel time when the vehicle just reached the link).
  int waiting_time_;
  // The arrival time of the representative vehicle, the travel is either 0 if
  // the vehicle is still in the network or its arrival time if the vehicle has
  // reached its destination.
  double vehicle_final_travel_time_;
  // Current location of the vehicle as a network road section.
  std::string vehicle_location_;
  // The destination of the representative vehicle corresponding to this state.
  // It is associated to the representative vehicle after the initial chance
  // node according to the od_demand distribution.
  std::string vehicle_destination_;

  // Size of the time step, used to convert travel times into number of game
  // time steps.
  const double time_step_length_;
  // Encodes maximum arrival time on any link in number of time steps.
  // Needed to enumerate all the possible state of a vehicle being on a link to
  // compute volume of cars on the link.
  const int max_travel_time_;
  // Whether to perform sanity checks, derived from `MeanFieldRoutingGame`.
  const bool perform_sanity_checks_;
  // An array of OriginDestinationDemand derived from `MeanFieldRoutingGame`,
  // owned by the corresponding game.
  const std::vector<OriginDestinationDemand>* od_demand_;
  // Network owned by the corresponding game.
  const Network* network_;
  // Total number of vehicles as the sum of the od_demand.
  const double total_num_vehicle_;
  // Chance outcomes based on the initial probability distribution.
  const ActionsAndProbs chance_outcomes_;

  friend class MeanFieldRoutingGame;
};

// In the implementation of the mean field routing game, the representative
// vehicle/player is represented as a tuple current location, current waiting
// time and destination. When the waiting time is negative, the vehicle chooses
// the successor link it would like to go. When arriving on the link, a
// waiting time is assigned to the player based on the distribution of players
// on the link. The vehicle arrival time is equal to the time step when they
// first reach their destination. See module docstring for more information.
class MeanFieldRoutingGame : public Game {
 public:
  // Constructor of the game.
  // Args:
  //   `params`: game parameters. It should define max_num_time_step,
  //     time_step_length, network and perform_sanity_checks.
  explicit MeanFieldRoutingGame(const GameParameters& params);

  // There is only 1 chance node (the initial node).
  int MaxChanceNodesInHistory() const override { return 1; }
  // Maximum number of possible actions.
  // This is equal to the number of links + 1
  //   (corresponding to having no possible action kNoPossibleAction).
  int NumDistinctActions() const override {
    return game_info_.num_distinct_actions;
  }
  // The number of vehicles.
  // Should be 1 as this mean field game is a one population game.
  int NumPlayers() const override {
    SPIEL_CHECK_EQ(game_info_.num_players, 1);
    return game_info_.num_players;
  }
  // Minimum utility is the opposite of the maximum arrival time.
  // Set to - max_game_length - 1.
  double MinUtility() const override {
    SPIEL_CHECK_EQ(game_info_.min_utility, -1 * game_info_.max_game_length - 1);
    return game_info_.min_utility;
  }
  // Maximum utility is the opposite of the minimum arrival time. Set to 0.
  double MaxUtility() const override { return game_info_.max_utility; }
  // Maximum number of time step played. Passed during construction.
  int MaxGameLength() const override { return game_info_.max_game_length; }
  // Maximum number of chance actions. Set to the length of
  // od_demand_, i.e. the number of `OriginDestinationDemand`s.
  int MaxChanceOutcomes() const override {
    return game_info_.max_chance_outcomes;
  }
  // If true, sanity checks are done during the game, should be set to false to
  // speed up the game.
  bool perform_sanity_checks() const { return perform_sanity_checks_; }

  // Creates a new initial state of the MeanFieldRoutingGame.
  std::unique_ptr<State> NewInitialState() const override {
    return MeanFieldRoutingGameState::CreateNewInitialState(
        shared_from_this(), time_step_length_, od_demand_.get(), network_.get(),
        perform_sanity_checks_);
  }

  // Returns the tensor shape for observation.
  std::vector<int> ObservationTensorShape() const override {
    int num_locations = network_->num_actions();
    int max_num_time_step = MaxGameLength();
    return {num_locations * 2 + max_num_time_step + 1 + 1};
  }

  // Deserialize a formatted string to MeanFieldRoutingGameState.
  std::unique_ptr<State> DeserializeState(
      const std::string& str) const override;

 private:
  std::string network_name_;
  std::unique_ptr<Network> network_;
  // A list of the vehicle. Their origin and their destination should be road
  // sections of the game.
  std::unique_ptr<std::vector<OriginDestinationDemand>> od_demand_;
  // If true, sanity checks are done during the game, should be set to false to
  // speed up the game.
  bool perform_sanity_checks_;
  // Is used to convert travel times into number of game time steps.
  double time_step_length_;
  GameInfo game_info_;
};

}  // namespace open_spiel::dynamic_routing

#endif  // OPEN_SPIEL_GAMES_MFG_DYNAMIC_ROUTING_H_
