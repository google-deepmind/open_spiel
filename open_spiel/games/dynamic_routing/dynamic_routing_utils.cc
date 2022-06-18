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

#include "open_spiel/games/dynamic_routing/dynamic_routing_utils.h"

#include <math.h>

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "open_spiel/abseil-cpp/absl/container/btree_map.h"
#include "open_spiel/abseil-cpp/absl/container/flat_hash_map.h"
#include "open_spiel/abseil-cpp/absl/container/flat_hash_set.h"
#include "open_spiel/abseil-cpp/absl/memory/memory.h"
#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/abseil-cpp/absl/strings/str_split.h"
#include "open_spiel/abseil-cpp/absl/strings/string_view.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel::dynamic_routing {
namespace {

template <typename KeyType, typename ValueType>
absl::flat_hash_set<KeyType> GetKeySet(
    const absl::flat_hash_map<KeyType, ValueType>& m) {
  absl::flat_hash_set<std::string> keys;
  for (const auto& pair : m) {
    keys.emplace(pair.first);
  }
  return keys;
}

absl::flat_hash_map<std::string, float> AssignExistingOrDefaultValues(
    absl::flat_hash_map<std::string, float> dict_object,
    absl::flat_hash_set<std::string> road_sections, float default_value) {
  if (!dict_object.empty()) {
    SPIEL_CHECK_TRUE((GetKeySet<std::string, float>(dict_object)) ==
                     road_sections);
    return dict_object;
  }
  absl::flat_hash_map<std::string, float> dict_object_returned;
  for (const auto& key : road_sections) {
    dict_object_returned.emplace(key, default_value);
  }
  return dict_object_returned;
}
}  // namespace

std::string RoadSectionFromNodes(absl::string_view origin,
                                 absl::string_view destination) {
  return absl::StrCat(origin, "->", destination);
}

std::vector<std::string> NodesFromRoadSection(std::string road_section) {
  return absl::StrSplit(road_section, "->");
}

std::unique_ptr<Network> Network::Create(
    const absl::flat_hash_map<std::string, std::vector<std::string>>&
        adjacency_list,
    const absl::flat_hash_map<std::string, std::pair<float, float>>&
        node_position,
    const absl::flat_hash_map<std::string, float>& bpr_a_coefficient,
    const absl::flat_hash_map<std::string, float>& bpr_b_coefficient,
    const absl::flat_hash_map<std::string, float>& capacity,
    const absl::flat_hash_map<std::string, float>& free_flow_travel_time) {
  return absl::WrapUnique(new Network(adjacency_list, node_position,
                                      bpr_a_coefficient, bpr_b_coefficient,
                                      capacity, free_flow_travel_time));
}

Network::Network(
    absl::flat_hash_map<std::string, std::vector<std::string>> adjacency_list,
    absl::flat_hash_map<std::string, std::pair<float, float>> node_position,
    absl::flat_hash_map<std::string, float> bpr_a_coefficient,
    absl::flat_hash_map<std::string, float> bpr_b_coefficient,
    absl::flat_hash_map<std::string, float> capacity,
    absl::flat_hash_map<std::string, float> free_flow_travel_time) {
  adjacency_list_ = adjacency_list;
  // Sort the adjacency list to make the action id unique.
  absl::btree_map<std::string, std::vector<std::string>> sorted_adjacency_list;
  sorted_adjacency_list.insert(adjacency_list.begin(), adjacency_list.end());
  action_by_road_section_.clear();
  road_section_by_action.clear();
  road_section_by_action.emplace_back("");  // Dummy road section at index 0.
  int action_number = kNoPossibleAction + 1;
  for (auto& [origin, successors] : sorted_adjacency_list) {
    std::sort(successors.begin(), successors.end());
    for (const auto& destination : successors) {
      std::string road_section = RoadSectionFromNodes(origin, destination);
      SPIEL_CHECK_FALSE(action_by_road_section_.contains(road_section));
      action_by_road_section_.emplace(road_section, action_number);
      road_section_by_action.emplace_back(road_section);
      // Adds road_section with no successors to sink_road_sections_;
      if (sorted_adjacency_list.at(destination).empty()) {
        sink_road_sections_.emplace(road_section);
      }
      action_number++;
    }
  }
  node_position_ = node_position;
  absl::flat_hash_set<std::string> road_sections =
      GetKeySet<std::string, int>(action_by_road_section_);
  bpr_a_coefficient_ =
      AssignExistingOrDefaultValues(bpr_a_coefficient, road_sections, 0);
  bpr_b_coefficient_ =
      AssignExistingOrDefaultValues(bpr_b_coefficient, road_sections, 1);
  capacity_ = AssignExistingOrDefaultValues(capacity, road_sections, 1);
  free_flow_travel_time_ =
      AssignExistingOrDefaultValues(free_flow_travel_time, road_sections, 1);
}

float Network::GetTravelTime(absl::string_view road_section,
                             float volume) const {
  SPIEL_CHECK_TRUE(free_flow_travel_time_.contains(road_section));
  SPIEL_CHECK_TRUE(bpr_a_coefficient_.contains(road_section));
  SPIEL_CHECK_TRUE(bpr_b_coefficient_.contains(road_section));
  SPIEL_CHECK_TRUE(capacity_.contains(road_section));

  float free_flow_travel_time = free_flow_travel_time_.at(road_section);
  float a = bpr_a_coefficient_.at(road_section);
  float b = bpr_b_coefficient_.at(road_section);
  float capacity = capacity_.at(road_section);
  return free_flow_travel_time * (1.0 + a * pow(volume / capacity, b));
}

bool Network::IsLocationASinkNode(absl::string_view road_section) const {
  return sink_road_sections_.contains(road_section);
}

int Network::GetActionIdFromMovement(absl::string_view origin,
                                     absl::string_view destination) const {
  std::string section = RoadSectionFromNodes(origin, destination);
  SPIEL_CHECK_TRUE(action_by_road_section_.contains(section));
  return action_by_road_section_.at(section);
}

int Network::num_links() const { return this->action_by_road_section_.size(); }

int Network::num_actions() const { return 1 + this->num_links(); }

std::vector<std::string> Network::GetSuccessors(absl::string_view node) const {
  SPIEL_CHECK_TRUE(adjacency_list_.contains(node));
  return adjacency_list_.at(node);
}

std::string Network::GetRoadSectionFromActionId(int action) const {
  return road_section_by_action.at(action);
}

int Network::GetRoadSectionAsInt(std::string section) const {
  if (section.empty()) {
    return 0;
  }
  std::vector<std::string> nodes = NodesFromRoadSection(section);
  std::string start_node = nodes[0];
  std::string end_node = nodes[1];
  return GetActionIdFromMovement(start_node, end_node);
}

void Network::AssertValidAction(int action, std::string road_section) const {
  SPIEL_CHECK_GE(action, 1);
  SPIEL_CHECK_LT(action, num_actions());
  if (!road_section.empty()) {
    std::string new_road_section = GetRoadSectionFromActionId(action);
    std::vector<std::string> nodes = NodesFromRoadSection(new_road_section);
    std::string origin_new_section = nodes[0];
    std::string end_new_section = nodes[1];
    std::string end_section_node = NodesFromRoadSection(road_section)[1];
    SPIEL_CHECK_EQ(end_section_node, origin_new_section);
    std::vector<std::string> successors = GetSuccessors(origin_new_section);
    SPIEL_CHECK_TRUE(std::find(successors.begin(), successors.end(),
                               end_new_section) != successors.end());
  }
}

void Network::CheckListOfOdDemandIsCorrect(
    std::vector<OriginDestinationDemand>* od_demands) {
  for (const OriginDestinationDemand& od_demand : *od_demands) {
    SPIEL_CHECK_TRUE(
        action_by_road_section_.contains(od_demand.vehicle.origin));
    SPIEL_CHECK_TRUE(
        action_by_road_section_.contains(od_demand.vehicle.destination));
  }
}

}  // namespace open_spiel::dynamic_routing
