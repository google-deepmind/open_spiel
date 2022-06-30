// Copyright 2019 DeepMind Technologies Limited
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

void GoofspielWithLimitedTurns() {
  GameParameters params;
  params["imp_info"] = GameParameter(true);
  params["num_cards"] = GameParameter(13);
  params["num_turns"] = GameParameter(3);
  params["points_order"] = GameParameter(std::string("descending"));
  testing::RandomSimTest(*LoadGame("goofspiel", params), 10);
}

void EgocentricViewOfSymmetricActions() {
  GameParameters params;
  params["imp_info"] = GameParameter(true);
  params["egocentric"] = GameParameter(true);
  params["num_cards"] = GameParameter(4);
  params["players"] = GameParameter(3);
  params["points_order"] = GameParameter(std::string("descending"));
  std::shared_ptr<const Game> game = LoadGame("goofspiel", params);

  std::unique_ptr<open_spiel::State> state = game->NewInitialState();

  // Three action sequences each played by one player.
  std::vector<Action> seq1{3, 2, 0 /*, 1 */};
  std::vector<Action> seq2{0, 1, 2 /*, 3 */};
  std::vector<Action> seq3{2, 3, 1 /*, 0 */};

  // Accumulate info state histories form the perspective of `seq1` when playing
  // as one of the three players.
  std::vector<std::vector<std::vector<float>>> info_state_histories(
      game->NumPlayers());
  for (int as_player = 0; as_player < game->NumPlayers(); as_player++) {
    for (int t = 0; t < game->MaxGameLength() - 1; t++) {
      std::vector<Action> joint_actions(game->NumPlayers(), -1);
      joint_actions[as_player] = seq1[t];
      joint_actions[(as_player + 1) % game->NumPlayers()] = seq2[t];
      joint_actions[(as_player + 2) % game->NumPlayers()] = seq3[t];
      state->ApplyActions(std::move(joint_actions));
      auto info_state = state->InformationStateTensor(as_player);
      info_state_histories[as_player].push_back(std::move(info_state));
    }
    state = game->NewInitialState();
  }

  // Verify that the observations remain identical regardless of which player
  // `seq1` was executed for.
  SPIEL_CHECK_EQ(info_state_histories.size(), game->NumPlayers());
  SPIEL_CHECK_EQ(info_state_histories[0], info_state_histories[1]);
  SPIEL_CHECK_EQ(info_state_histories[1], info_state_histories[2]);
}

}  // namespace
}  // namespace goofspiel
}  // namespace open_spiel

int main(int argc, char **argv) {
  open_spiel::goofspiel::BasicGoofspielTests();
  open_spiel::goofspiel::LegalActionsValidAtEveryState();
  open_spiel::goofspiel::GoofspielWithLimitedTurns();
  open_spiel::goofspiel::EgocentricViewOfSymmetricActions();
}
