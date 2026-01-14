// Copyright 2021 DeepMind Technologies Limited
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

#include <memory>
#include <random>

#include "open_spiel/abseil-cpp/absl/random/uniform_int_distribution.h"
#include "open_spiel/matrix_game.h"
#include "open_spiel/spiel.h"

int main(int argc, char** argv) {
  // Random number generator.
  std::mt19937 rng;

  // Create the game with its default parameter settings.
  std::cerr << "Creating game..\n" << std::endl;
  std::shared_ptr<const open_spiel::Game> game(
      new open_spiel::matrix_game::MatrixGame(
          {/*short_name=*/"matrix_pd",
           /*long_name=*/"Prisoner's Dilemma",
           open_spiel::GameType::Dynamics::kSimultaneous,
           open_spiel::GameType::ChanceMode::kDeterministic,
           open_spiel::GameType::Information::kPerfectInformation,
           open_spiel::GameType::Utility::kGeneralSum,
           open_spiel::GameType::RewardModel::kTerminal,
           /*max_num_players=*/2,
           /*min_num_players=*/2,
           /*provides_information_state_string=*/true,
           /*provides_information_state_tensor=*/true,
           /*parameter_specification=*/{}},
          {},                       // Empty parameters
          {"Cooperate", "Defect"},  // (Row) Player 0's actions
          {"Cooperate", "Defect"},  // (Column) Player 2's actions
          {5, 0, 10, 1},            // Player 1's utilities in row-major order
          {5, 10, 0, 1}             // Player 2's utilities in row-major order
          ));

  // Note: matrix games can also be registered through the main factory, just
  // like the other games in spiel, and then loaded through
  // open_spiel::LoadGame. See games/matrix_games.cc for how to register matrix
  // games.

  std::cerr << "Starting new game..." << std::endl;
  std::unique_ptr<open_spiel::State> state = game->NewInitialState();

  std::vector<open_spiel::Action> row_actions = state->LegalActions(0);
  std::vector<open_spiel::Action> col_actions = state->LegalActions(1);

  open_spiel::Action row_action =
      row_actions[absl::uniform_int_distribution<int>(
          0, row_actions.size() - 1)(rng)];
  open_spiel::Action col_action =
      col_actions[absl::uniform_int_distribution<int>(
          0, col_actions.size() - 1)(rng)];

  std::cerr << "Joint action is: (" << state->ActionToString(0, row_action)
            << "," << state->ActionToString(1, row_action) << ")" << std::endl;

  state->ApplyActions({row_action, col_action});

  SPIEL_CHECK_TRUE(state->IsTerminal());

  auto returns = state->Returns();
  for (int p = 0; p < game->NumPlayers(); p++) {
    std::cerr << "Terminal return to player " << p << " is " << returns[p]
              << std::endl;
  }
}
