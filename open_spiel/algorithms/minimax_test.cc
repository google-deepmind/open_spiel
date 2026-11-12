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

#include "open_spiel/algorithms/minimax.h"

#include <cmath>
#include <memory>
#include <utility>
#include <vector>

#include "open_spiel/games/pig/pig.h"
#include "open_spiel/games/tic_tac_toe/tic_tac_toe.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace algorithms {
namespace {

// See also the examples/minimax_example.cc for example usage.

void AlphaBetaSearchTest_TicTacToe() {
  std::shared_ptr<const Game> game = LoadGame("tic_tac_toe");
  std::pair<double, BestActions> value_and_actions =
      AlphaBetaSearch(*game, nullptr, {}, -1, kInvalidPlayer);

  const float value = value_and_actions.first;
  SPIEL_CHECK_EQ(0.0, value);

  const BestActions& actions = value_and_actions.second;
  std::vector<Action> true_best_actions = {0, 1, 2, 3, 4, 5, 6, 7, 8};
  SPIEL_CHECK_TRUE(actions.Equals(true_best_actions));
}

void AlphaBetaSearchTest_TicTacToe_Win() {
  std::shared_ptr<const Game> game = LoadGame("tic_tac_toe");
  std::unique_ptr<State> state = game->NewInitialState();
  state->ApplyAction(4);
  state->ApplyAction(1);

  // Construct:
  // .o.
  // .x.
  // ...
  // Optimal actions: 0 (R1C1), 2 (R1C3), 3 (R2C1), 5 (R2C3), 6 (R3C1),
  // 8 (R3C3). Action 7 (R3C2) is legal but only draws.
  std::pair<double, BestActions> value_and_actions =
      AlphaBetaSearch(*game, state.get(), {}, -1, kInvalidPlayer);

  const float value = value_and_actions.first;
  SPIEL_CHECK_EQ(1.0, value_and_actions.first);

  const BestActions& actions = value_and_actions.second;
  std::vector<Action> true_best_actions = {0, 2, 3, 5, 6, 8};
  SPIEL_CHECK_TRUE(actions.Equals(true_best_actions));
}

void AlphaBetaSearchTest_TicTacToe_Loss() {
  std::shared_ptr<const Game> game = LoadGame("tic_tac_toe");
  std::unique_ptr<State> state = game->NewInitialState();

  // Construct:
  // ...
  // xox
  // ..o
  // Optimal actions: 0 (R1C1), 1 (R1C2), 2 (R1C3), 6 (R3C1), 7 (R3C2).
  state->ApplyAction(5);
  state->ApplyAction(4);
  state->ApplyAction(3);
  state->ApplyAction(8);

  std::pair<double, BestActions> value_and_actions =
      AlphaBetaSearch(*game, state.get(), {}, -1, kInvalidPlayer);

  const float value = value_and_actions.first;
  SPIEL_CHECK_EQ(-1.0, value);

  const BestActions& actions = value_and_actions.second;
  std::vector<Action> true_best_actions = {0, 1, 2, 6, 7};
  SPIEL_CHECK_TRUE(actions.Equals(true_best_actions));
}

void AlphaBetaSearchTest_TicTacToe_SingleAction() {
  std::shared_ptr<const Game> game = LoadGame("tic_tac_toe");
  std::unique_ptr<State> state = game->NewInitialState();

  // Construct:
  // xox
  // ...
  // ...
  // Optimal actions: 4 (R2C2) only. Every other legal action loses.
  state->ApplyAction(0);
  state->ApplyAction(1);
  state->ApplyAction(2);

  std::pair<double, BestActions> value_and_actions =
      AlphaBetaSearch(*game, state.get(), {}, -1, kInvalidPlayer);

  const float value = value_and_actions.first;
  SPIEL_CHECK_EQ(0.0, value);

  const BestActions& actions = value_and_actions.second;
  std::vector<Action> true_best_actions = {4};
  SPIEL_CHECK_TRUE(actions.Equals(true_best_actions));
}

int FirstPlayerAdvantage(const State& state) {
  const auto& pstate = down_cast<const open_spiel::pig::PigState&>(state);
  return pstate.score(0) - pstate.score(1);
}

void ExpectiminimaxSearchTest_Pig() {
  std::shared_ptr<const Game> game =
      LoadGame("pig", {{"diceoutcomes", GameParameter(3)}});
  std::pair<double, BestActions> value_and_actions = ExpectiminimaxSearch(
      *game, nullptr, FirstPlayerAdvantage, 2, kInvalidPlayer);

  const double value = value_and_actions.first;
  const double true_value = 1.0 / 3 * 2 + 1.0 / 3 * 3;
  SPIEL_CHECK_EQ(true_value, value);

  const BestActions& actions = value_and_actions.second;
  const std::vector<Action> true_best_actions = {/*kRoll=*/0};
  SPIEL_CHECK_TRUE(actions.Equals(true_best_actions));
}

}  // namespace
}  // namespace algorithms
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::algorithms::AlphaBetaSearchTest_TicTacToe();
  open_spiel::algorithms::AlphaBetaSearchTest_TicTacToe_Win();
  open_spiel::algorithms::AlphaBetaSearchTest_TicTacToe_Loss();
  open_spiel::algorithms::AlphaBetaSearchTest_TicTacToe_SingleAction();
  open_spiel::algorithms::ExpectiminimaxSearchTest_Pig();
}
