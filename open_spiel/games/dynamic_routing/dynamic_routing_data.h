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

#ifndef OPEN_SPIEL_GAMES_DYNAMIC_ROUTING_DYNAMIC_ROUTING_DATA_H_
#define OPEN_SPIEL_GAMES_DYNAMIC_ROUTING_DYNAMIC_ROUTING_DATA_H_

#include <memory>

#include "open_spiel/abseil-cpp/absl/container/flat_hash_map.h"
#include "open_spiel/abseil-cpp/absl/strings/string_view.h"
#include "open_spiel/games/dynamic_routing/dynamic_routing_utils.h"

namespace open_spiel::dynamic_routing {

// The enum for supported Dynamic Routing Data.
enum class DynamicRoutingDataName { kLine, kBraess };

// Data of the Dynamic Routing Game
class DynamicRoutingData {
 public:
  // Creates data for the specific dynamic routing game.
  static std::unique_ptr<DynamicRoutingData> Create(
      DynamicRoutingDataName name);

  std::unique_ptr<Network> network_;
  std::unique_ptr<std::vector<OriginDestinationDemand>> od_demand_;
};

}  // namespace open_spiel::dynamic_routing

#endif  // OPEN_SPIEL_GAMES_DYNAMIC_ROUTING_DYNAMIC_ROUTING_DATA_H_
