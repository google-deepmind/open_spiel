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

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "open_spiel/abseil-cpp/absl/container/flat_hash_map.h"
#include "open_spiel/abseil-cpp/absl/memory/memory.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel::dynamic_routing {

namespace {

using ::open_spiel::dynamic_routing::RoadSectionFromNodes;
using ::open_spiel::dynamic_routing::NodesFromRoadSection;

void TestRoadSectionFromNodes() {
  std::string road_section = RoadSectionFromNodes("A", "B");
  SPIEL_CHECK_TRUE(road_section == "A->B");
}

void TestNodesFromRoadSection() {
  std::string road_section = "A->B";
  std::vector<std::string> nodes = NodesFromRoadSection(road_section);
  std::vector<std::string> expected{"A", "B"};
  SPIEL_CHECK_TRUE(nodes == expected);
}

void TestVehicleInstanciation1() {
  auto vehicle = absl::make_unique<Vehicle>("O->A", "B->D");
  SPIEL_CHECK_EQ(vehicle->origin, "O->A");
  SPIEL_CHECK_EQ(vehicle->destination, "B->D");
  SPIEL_CHECK_FLOAT_EQ(vehicle->departure_time, 0);
}

void TestVehicleInstanciation2() {
  auto vehicle = absl::make_unique<Vehicle>("O->A", "B->D", 10.5);
  SPIEL_CHECK_EQ(vehicle->origin, "O->A");
  SPIEL_CHECK_EQ(vehicle->destination, "B->D");
  SPIEL_CHECK_FLOAT_EQ(vehicle->departure_time, 10.5);
}

void TestOdDemandInstanciation1() {
  auto od_demand =
      absl::make_unique<OriginDestinationDemand>("O->A", "B->D", 0, 30);
  SPIEL_CHECK_EQ(od_demand->vehicle.origin, "O->A");
  SPIEL_CHECK_EQ(od_demand->vehicle.destination, "B->D");
  SPIEL_CHECK_FLOAT_EQ(od_demand->vehicle.departure_time, 0);
  SPIEL_CHECK_FLOAT_EQ(od_demand->counts, 30);
}

void TestOdDemandInstanciation2() {
  auto od_demand =
      absl::make_unique<OriginDestinationDemand>("O->A", "B->D", 10.5, 43.2);
  SPIEL_CHECK_EQ(od_demand->vehicle.origin, "O->A");
  SPIEL_CHECK_EQ(od_demand->vehicle.destination, "B->D");
  SPIEL_CHECK_FLOAT_EQ(od_demand->vehicle.departure_time, 10.5);
  SPIEL_CHECK_FLOAT_EQ(od_demand->counts, 43.2);
}

void TestNetworkInitWithEmpty() {
  absl::flat_hash_map<std::string, std::vector<std::string>> adjacency_list =
      {};
  auto network = Network::Create(adjacency_list);
}

std::unique_ptr<Network> InitNetwork() {
  absl::flat_hash_map<std::string, std::vector<std::string>> adjacency_list;
  adjacency_list["O"] = std::vector<std::string>{"A"};
  adjacency_list["A"] = std::vector<std::string>{"D"};
  adjacency_list["D"] = std::vector<std::string>{};
  return Network::Create(adjacency_list);
}

void TestNetworkAdjacencyListInit() {
  auto network = InitNetwork();
  SPIEL_CHECK_EQ(network->GetActionIdFromMovement("O", "A"), 2);
  SPIEL_CHECK_EQ(network->GetActionIdFromMovement("A", "D"), 1);
  SPIEL_CHECK_EQ(network->num_links(), 2);
  SPIEL_CHECK_EQ(network->GetSuccessors("O"), std::vector<std::string>{"A"});
  SPIEL_CHECK_EQ(network->GetSuccessors("A"), std::vector<std::string>{"D"});
  SPIEL_CHECK_EQ(network->GetSuccessors("D"), std::vector<std::string>{});
  SPIEL_CHECK_TRUE(network->IsLocationASinkNode("A->D"));
  SPIEL_CHECK_FALSE(network->IsLocationASinkNode("O->A"));
  SPIEL_CHECK_EQ(network->GetRoadSectionFromActionId(2), "O->A");
  SPIEL_CHECK_EQ(network->GetRoadSectionFromActionId(1), "A->D");
}

// Exceptions are checked in the code with SPIEL_CHECK_TRUE.

}  // namespace
}  // namespace open_spiel::dynamic_routing

int main(int argc, char** argv) {
  open_spiel::dynamic_routing::TestRoadSectionFromNodes();
  open_spiel::dynamic_routing::TestNodesFromRoadSection();
  open_spiel::dynamic_routing::TestVehicleInstanciation1();
  open_spiel::dynamic_routing::TestVehicleInstanciation2();
  open_spiel::dynamic_routing::TestOdDemandInstanciation1();
  open_spiel::dynamic_routing::TestOdDemandInstanciation2();
  open_spiel::dynamic_routing::TestNetworkInitWithEmpty();
  open_spiel::dynamic_routing::TestNetworkAdjacencyListInit();
}
