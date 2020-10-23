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

#include "open_spiel/algorithms/ortools/sequence_form_lp.h"
#include "open_spiel/algorithms/tabular_exploitability.h"
#include "open_spiel/game_transforms/turn_based_simultaneous_game.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace algorithms {
namespace ortools {
namespace {

constexpr double kErrorTolerance = 1e-14;

void TestGameValueAndExploitability(const std::string& game_name,
                                    double expected_game_value) {
  std::shared_ptr<const Game> game = LoadGame(game_name);
  ZeroSumSequentialGameSolution solution = SolveZeroSumSequentialGame(*game);
  SPIEL_CHECK_FLOAT_NEAR(solution.game_value, expected_game_value,
                         kErrorTolerance);

  if (game->GetType().dynamics == GameType::Dynamics::kSimultaneous)
    return;
  SPIEL_CHECK_FLOAT_NEAR(Exploitability(*game, solution.policy), 0.,
                         kErrorTolerance);
}

void TestKuhnPokerRootCfvs() {
  std::shared_ptr<const Game> game = LoadGame("kuhn_poker");
  // Request the states after distributing a card to the first player.
  std::unique_ptr<State> state = game->NewInitialState();
  std::vector<const State*> starting_states;
  std::unique_ptr<State> state_J = state->Child(0);
  starting_states.push_back(state_J.get());
  std::unique_ptr<State> state_Q = state->Child(1);
  starting_states.push_back(state_Q.get());
  std::unique_ptr<State> state_K = state->Child(2);
  starting_states.push_back(state_K.get());

  std::array<std::vector<float>, 2> player_ranges = {
      std::vector<float>{1., 1., 1.},
      std::vector<float>{1.}  // Player 2 didn't get any card yet.
  };
  std::vector<float> chance_range = {1. / 3., 1. / 3., 1. / 3.};

  ZeroSumSequentialGameSolution solution = SolveZeroSumSequentialGame(
      game->MakeObserver(kInfoStateObsType, {}),
      absl::MakeSpan(starting_states),
      {absl::MakeSpan(player_ranges[0]),
       absl::MakeSpan(player_ranges[1])},
      absl::MakeSpan(chance_range),
      /*solve_only_player=*/{},
      /*collect_tabular_policy=*/true,
      /*collect_root_cfvs=*/true);

  SPIEL_CHECK_FLOAT_NEAR(solution.game_value, -1 / 18., kErrorTolerance);
  SPIEL_CHECK_FLOAT_NEAR(Exploitability(*game, solution.policy), 0.,
                         kErrorTolerance);
  // Check cf values of each individual subtree, i.e. after dealing a card
  // for the player. Notice that these sum up to -1 / 18.
  SPIEL_CHECK_FLOAT_NEAR(solution.root_cfvs[0][0], -1 / 3., kErrorTolerance);
  SPIEL_CHECK_FLOAT_NEAR(solution.root_cfvs[0][1], -1 / 9., kErrorTolerance);
  SPIEL_CHECK_FLOAT_NEAR(solution.root_cfvs[0][2], 7 / 18., kErrorTolerance);
  // Second player didn't receive a card yet, so the utility is -game_value.
  SPIEL_CHECK_FLOAT_NEAR(solution.root_cfvs[1][0], 1 / 18., kErrorTolerance);
}

}  // namespace
}  // namespace ortools
}  // namespace algorithms
}  // namespace open_spiel

namespace algorithms = open_spiel::algorithms;

int main(int argc, char** argv) {
  algorithms::ortools::TestGameValueAndExploitability("matrix_mp", 0.);
  algorithms::ortools::TestGameValueAndExploitability("kuhn_poker", -1 / 18.);
  algorithms::ortools::TestGameValueAndExploitability(
      "leduc_poker", -0.085606424078);
  algorithms::ortools::TestGameValueAndExploitability(
      "goofspiel(players=2,num_cards=3,imp_info=True)", 0.);

  algorithms::ortools::TestKuhnPokerRootCfvs();
}
