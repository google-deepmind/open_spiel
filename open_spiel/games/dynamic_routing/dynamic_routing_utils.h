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

// Utils for dynamic routing game and mean field routing game.
// This module has three main classes:
//  - Network
//  - Vehicle
//  - OriginDestinationDemand

#ifndef OPEN_SPIEL_GAMES_DYNAMIC_ROUTING_DYNAMIC_ROUTING_UTILS_H_
#define OPEN_SPIEL_GAMES_DYNAMIC_ROUTING_DYNAMIC_ROUTING_UTILS_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "open_spiel/abseil-cpp/absl/container/flat_hash_map.h"
#include "open_spiel/abseil-cpp/absl/container/flat_hash_set.h"
#include "open_spiel/abseil-cpp/absl/strings/string_view.h"

namespace open_spiel::dynamic_routing {

// In case one vehicle has reached a end node, then it cannot do anything. In
// this case its action is 0. Action 0 is reserved to encode no possible action
// as requested by Open Spiel.
inline constexpr int kNoPossibleAction = 0;

// Creates a road section "A->B" from two nodes "A" and "B".
std::string RoadSectionFromNodes(absl::string_view origin,
                                 absl::string_view destination);

// Creates a vector of two nodes {"A", "B"} from a road section "A->B".
std::vector<std::string> NodesFromRoadSection(std::string road_section);

// A Vehicle is one origin and one destination.
//
// Both the origin and the destination of the vehicle are road section,
// therefore they are string formatted as "{str}->{str}".
// Attributes:
//    origin: origin of the vehicle.
//    destination: destination of the vehicle.
//    departure_time: departure time of the vehicle.
struct Vehicle {
  Vehicle(absl::string_view origin, absl::string_view destination,
          float departure_time = 0)
      : origin(origin),
        destination(destination),
        departure_time(departure_time) {}

  const std::string origin;
  const std::string destination;
  const float departure_time;
};

// Number of trips from origin to destination for a specific departure time.
// Both the origin and the destination of the vehicle are road section,
// therefore they are string formatted as "{str}->{str}".
struct OriginDestinationDemand {
  explicit OriginDestinationDemand(absl::string_view origin,
                                   absl::string_view destination,
                                   float departure_time, float counts)
      : vehicle{origin, destination, departure_time}, counts(counts) {}

  // The vehicles in the origin destination demand with the same origin,
  // destination and departure time.
  Vehicle vehicle;
  // The number of vehicles with the origin, destination and departure time.
  const float counts;
};

// Network implementation.
//
// A network is a directed graph with a volume delay function on each
// of its edges. Each vertex is referred to as a string (for example "A") and
// each edge as a string f"{node1}->{node2}" (for example "A->B"). The network
// is created from an adjacency list. Each road section is mapped to an action
// index (positive integer) in road_section_to_action_, and vice versa in
// action_to_road_section_. The volume delay function on each road section rs
// is given by free_flow_travel_time_[rs]*(1+ a_[rs]*(v/capacity_[rs])**b_[rs])
// where v is the volume on the road section rs, according to the U.S. Bureau
// of Public Road (BPR). Such functions are called fundamental diagram of
// traffic flow.
class Network {
 public:
  // The factory function to create an instance of the Network class.
  static std::unique_ptr<Network> Create(
      const absl::flat_hash_map<std::string, std::vector<std::string>>&
          adjacency_list,
      const absl::flat_hash_map<std::string, std::pair<float, float>>&
          node_position = {},
      const absl::flat_hash_map<std::string, float>& bpr_a_coefficient = {},
      const absl::flat_hash_map<std::string, float>& bpr_b_coefficient = {},
      const absl::flat_hash_map<std::string, float>& capacity = {},
      const absl::flat_hash_map<std::string, float>& free_flow_travel_time =
          {});

  // Returns True if the road section has no successors.
  bool IsLocationASinkNode(absl::string_view road_section) const;

  // Returns travel time on the road section given the volume on it.
  // Volume unit should be the same as the capacity unit.
  // Travel time unit is the free flow travel time unit.
  // Args:
  //   road_section: the road section.
  //   volume: the volume on the road section.
  float GetTravelTime(absl::string_view road_section, float volume) const;

  // Maps two connected nodes to an action.
  int GetActionIdFromMovement(absl::string_view origin,
                              absl::string_view destination) const;

  // Returns the number of road sections.
  int num_links() const;

  // Returns the number of possible actions.
  int num_actions() const;

  // Returns the successor nodes of the node.
  std::vector<std::string> GetSuccessors(absl::string_view node) const;

  // Maps a action to the corresponding road section.
  std::string GetRoadSectionFromActionId(int action) const;

  // Returns the integer representation of the road section.
  int GetRoadSectionAsInt(std::string section) const;

  // Assert that an action as a int is valid.
  // The action should be a int between 1 and num_actions. In case road_section
  // is not null then it is test if the action correspond to going on a road
  // section which is a successor of road_section.
  void AssertValidAction(int action, std::string road_section = "") const;

  // Assert that OD demands have valid origin and destination.
  void CheckListOfOdDemandIsCorrect(
      std::vector<OriginDestinationDemand>* od_demands);

 private:
  explicit Network(
      absl::flat_hash_map<std::string, std::vector<std::string>> adjacency_list,
      absl::flat_hash_map<std::string, std::pair<float, float>> node_position,
      absl::flat_hash_map<std::string, float> bpr_a_coefficient,
      absl::flat_hash_map<std::string, float> bpr_b_coefficient,
      absl::flat_hash_map<std::string, float> capacity,
      absl::flat_hash_map<std::string, float> free_flow_travel_time);

  // flat_hash_map that maps road section string representation to its a.
  absl::flat_hash_map<std::string, float> bpr_a_coefficient_;
  // flat_hash_map that maps road section string representation to its b.
  absl::flat_hash_map<std::string, float> bpr_b_coefficient_;
  // flat_hash_map that maps road section string representation to its adjacency
  // list.
  absl::flat_hash_map<std::string, std::vector<std::string>> adjacency_list_;
  // flat_hash_map that maps road section string representation to its capacity.
  absl::flat_hash_map<std::string, float> capacity_;
  // flat_hash_map that maps road section string representation to its free flow
  // travel time.
  absl::flat_hash_map<std::string, float> free_flow_travel_time_;
  // flat_hash_map that maps road section string representation to couple of
  // float encoding x and y position of the node. None by default.
  absl::flat_hash_map<std::string, std::pair<float, float>> node_position_;
  // flat_hash_map that maps road section string representation to action.
  absl::flat_hash_map<std::string, int> action_by_road_section_;
  // vector that maps action to road section string representation.
  std::vector<std::string> road_section_by_action;
  // flat_hash_set that contains sink locations.
  absl::flat_hash_set<std::string> sink_road_sections_;
};
}  // namespace open_spiel::dynamic_routing

#endif  // OPEN_SPIEL_GAMES_DYNAMIC_ROUTING_DYNAMIC_ROUTING_UTILS_H_
