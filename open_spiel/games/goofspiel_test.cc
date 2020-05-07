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

#include "open_spiel/games/goofspiel.h"

#include "open_spiel/game_parameters.h"
#include "open_spiel/game_transforms/turn_based_simultaneous_game.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/tests/basic_tests.h"

namespace open_spiel {
namespace goofspiel {
namespace {

namespace testing = open_spiel::testing;

void BasicGoofspielTests() {
  testing::LoadGameTest("goofspiel");
  testing::ChanceOutcomesTest(*LoadGame("goofspiel"));
  testing::RandomSimTest(*LoadGame("goofspiel"), 100);
  for (Player players = 3; players <= 5; players++) {
    for (const std::string& returns_type :
         {"win_loss", "point_difference", "total_points"}) {
      testing::RandomSimTest(
          *LoadGame("goofspiel",
                    {{"players", GameParameter(players)},
                     {"returns_type", GameParameter(returns_type)}}),
          10);
    }
  }
}

void LegalActionsValidAtEveryState() {
  GameParameters params;
  params["imp_info"] = GameParameter(true);
  params["num_cards"] = GameParameter(4);
  params["points_order"] = GameParameter(std::string("descending"));
  std::shared_ptr<const Game> game = LoadGameAsTurnBased("goofspiel", params);
  testing::RandomSimTest(*game, /*num_sims=*/10);
}

}  // namespace
}  // namespace goofspiel
}  // namespace open_spiel

int main(int argc, char **argv) {
  open_spiel::goofspiel::BasicGoofspielTests();
  open_spiel::goofspiel::LegalActionsValidAtEveryState();
}
