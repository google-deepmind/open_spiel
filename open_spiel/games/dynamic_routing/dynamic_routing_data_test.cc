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

#include "open_spiel/games/dynamic_routing/dynamic_routing_utils.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel::dynamic_routing {

namespace {
float GetTravelTime(float free_flow_travel_time, float a, float b,
                    float capacity, float volume) {
  return free_flow_travel_time * (1.0 + a * pow(volume / capacity, b));
}
void TestGetDynamicRoutingDataLine() {
  std::unique_ptr<DynamicRoutingData> data =
      DynamicRoutingData::Create(DynamicRoutingDataName::kLine);
  Network* network = data->network_.get();
  OriginDestinationDemand od_demand = data->od_demand_->at(0);
  SPIEL_CHECK_EQ(network->num_links(), 4);
  SPIEL_CHECK_EQ(network->GetSuccessors("bef_O"),
                 std::vector<std::string>{"O"});
  SPIEL_CHECK_EQ(network->GetSuccessors("O"), std::vector<std::string>{"A"});
  SPIEL_CHECK_EQ(network->GetSuccessors("A"), std::vector<std::string>{"D"});
  SPIEL_CHECK_EQ(network->GetSuccessors("D"),
                 std::vector<std::string>{"aft_D"});
  SPIEL_CHECK_EQ(network->GetSuccessors("aft_D"), std::vector<std::string>{});
  SPIEL_CHECK_FALSE(network->IsLocationASinkNode("bef_O->O"));
  SPIEL_CHECK_FALSE(network->IsLocationASinkNode("O->A"));
  SPIEL_CHECK_FALSE(network->IsLocationASinkNode("A->D"));
  SPIEL_CHECK_TRUE(network->IsLocationASinkNode("D->aft_D"));
  SPIEL_CHECK_EQ(od_demand.vehicle.origin, "bef_O->O");
  SPIEL_CHECK_EQ(od_demand.vehicle.destination, "D->aft_D");
  SPIEL_CHECK_EQ(od_demand.vehicle.departure_time, 0);
  SPIEL_CHECK_EQ(od_demand.counts, 100);
}

void TestGetDynamicRoutingDataBraess() {
  std::unique_ptr<DynamicRoutingData> data =
      DynamicRoutingData::Create(DynamicRoutingDataName::kBraess);
  Network* network = data->network_.get();
  OriginDestinationDemand od_demand = data->od_demand_->at(0);
  SPIEL_CHECK_EQ(network->num_links(), 7);
  SPIEL_CHECK_EQ(network->GetSuccessors("O"), (std::vector<std::string>{"A"}));
  SPIEL_CHECK_EQ(network->GetSuccessors("A"),
                 (std::vector<std::string>{"B", "C"}));
  SPIEL_CHECK_EQ(network->GetSuccessors("B"),
                 (std::vector<std::string>{"C", "D"}));
  SPIEL_CHECK_EQ(network->GetSuccessors("C"), (std::vector<std::string>{"D"}));
  SPIEL_CHECK_EQ(network->GetSuccessors("D"), (std::vector<std::string>{"E"}));
  SPIEL_CHECK_EQ(network->GetSuccessors("E"), (std::vector<std::string>{}));
  SPIEL_CHECK_FALSE(network->IsLocationASinkNode("A->B"));
  SPIEL_CHECK_FALSE(network->IsLocationASinkNode("B->C"));
  SPIEL_CHECK_FALSE(network->IsLocationASinkNode("C->D"));
  SPIEL_CHECK_TRUE(network->IsLocationASinkNode("D->E"));
  SPIEL_CHECK_EQ(od_demand.vehicle.origin, "O->A");
  SPIEL_CHECK_EQ(od_demand.vehicle.destination, "D->E");
  SPIEL_CHECK_EQ(od_demand.vehicle.departure_time, 0);
  SPIEL_CHECK_EQ(od_demand.counts, 5);
  SPIEL_CHECK_EQ(network->GetTravelTime("O->A", 1.0), 0);
  SPIEL_CHECK_EQ(network->GetTravelTime("A->B", 1.0),
                 GetTravelTime(1.0, 1.0, 1.0, 5.0, 1.0));
  SPIEL_CHECK_EQ(network->GetTravelTime("A->C", 1.0),
                 GetTravelTime(2.0, 0, 1.0, 5.0, 1.0));
  SPIEL_CHECK_EQ(network->GetTravelTime("B->C", 1.0),
                 GetTravelTime(0.25, 0, 1.0, 5.0, 1.0));
  SPIEL_CHECK_EQ(network->GetTravelTime("B->D", 1.0),
                 GetTravelTime(2.0, 0, 1.0, 5.0, 1.0));
  SPIEL_CHECK_EQ(network->GetTravelTime("C->D", 1.0),
                 GetTravelTime(1.0, 1.0, 1.0, 5.0, 1.0));
  SPIEL_CHECK_EQ(network->GetTravelTime("D->E", 1.0), 0);
}

}  // namespace
}  // namespace open_spiel::dynamic_routing

int main(int argc, char** argv) {
  open_spiel::dynamic_routing::TestGetDynamicRoutingDataLine();
  open_spiel::dynamic_routing::TestGetDynamicRoutingDataBraess();
}
