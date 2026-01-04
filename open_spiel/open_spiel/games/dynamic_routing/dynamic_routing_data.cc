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

#include "open_spiel/games/dynamic_routing/dynamic_routing_data.h"

#include <memory>
#include <utility>
#include <vector>

#include "open_spiel/abseil-cpp/absl/memory/memory.h"
#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/abseil-cpp/absl/strings/string_view.h"
#include "open_spiel/games/dynamic_routing/dynamic_routing_utils.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel::dynamic_routing {

std::unique_ptr<DynamicRoutingData> DynamicRoutingData::Create(
    DynamicRoutingDataName name) {
  std::unique_ptr<DynamicRoutingData> data =
      absl::make_unique<DynamicRoutingData>();
  switch (name) {
    case DynamicRoutingDataName::kLine: {
      absl::flat_hash_map<std::string, std::vector<std::string>>
          adjacency_list = {{"bef_O", {"O"}},
                            {"O", {"A"}},
                            {"A", {"D"}},
                            {"D", {"aft_D"}},
                            {"aft_D", {}}};
      data->network_ = Network::Create(adjacency_list);
      data->od_demand_ =
          absl::make_unique<std::vector<OriginDestinationDemand>>(std::vector{
              OriginDestinationDemand("bef_O->O", "D->aft_D", 0, 100)});
      return data;
    }
    case DynamicRoutingDataName::kBraess: {
      const int kBraessNumPlayer = 5;
      absl::flat_hash_map<std::string, std::vector<std::string>>
          adjacency_list = {{"O", {"A"}}, {"A", {"B", "C"}}, {"B", {"C", "D"}},
                            {"C", {"D"}}, {"D", {"E"}},      {"E", {}}};
      absl::flat_hash_map<std::string, std::pair<float, float>> node_position =
          {{"O", {0, 0}},  {"A", {1, 0}}, {"B", {2, 1}},
           {"C", {2, -1}}, {"D", {3, 0}}, {"E", {4, 0}}};
      absl::flat_hash_map<std::string, float> bpr_a_coefficient = {
          {"O->A", 0}, {"A->B", 1.0}, {"A->C", 0}, {"B->C", 0},
          {"B->D", 0}, {"C->D", 1.0}, {"D->E", 0}};
      absl::flat_hash_map<std::string, float> bpr_b_coefficient = {
          {"O->A", 1.0}, {"A->B", 1.0}, {"A->C", 1.0}, {"B->C", 1.0},
          {"B->D", 1.0}, {"C->D", 1.0}, {"D->E", 1.0}};
      absl::flat_hash_map<std::string, float> capacity = {
          {"O->A", kBraessNumPlayer}, {"A->B", kBraessNumPlayer},
          {"A->C", kBraessNumPlayer}, {"B->C", kBraessNumPlayer},
          {"B->D", kBraessNumPlayer}, {"C->D", kBraessNumPlayer},
          {"D->E", kBraessNumPlayer}};
      absl::flat_hash_map<std::string, float> free_flow_travel_time = {
          {"O->A", 0},   {"A->B", 1.0}, {"A->C", 2.0}, {"B->C", 0.25},
          {"B->D", 2.0}, {"C->D", 1.0}, {"D->E", 0}};
      data->network_ =
          Network::Create(adjacency_list, node_position, bpr_a_coefficient,
                          bpr_b_coefficient, capacity, free_flow_travel_time);
      data->od_demand_ =
          absl::make_unique<std::vector<OriginDestinationDemand>>(std::vector{
              OriginDestinationDemand("O->A", "D->E", 0, kBraessNumPlayer)});
      return data;
    }
    default:
      open_spiel::SpielFatalError(
          absl::StrCat("Unknown Dynamic Routing Data Name: ", name));
  }
  return data;
}

}  // namespace open_spiel::dynamic_routing
