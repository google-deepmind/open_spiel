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

#include "open_spiel/games/pig.h"
#include "open_spiel/games/tic_tac_toe.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace algorithms {
namespace {

// See also the examples/minimax_example.cc for example usage.

void AlphaBetaSearchTest_TicTacToe() {
  std::shared_ptr<const Game> game = LoadGame("tic_tac_toe");
  std::pair<double, Action> value_and_action =
      AlphaBetaSearch(*game, nullptr, {}, -1, kInvalidPlayer);
  SPIEL_CHECK_EQ(0.0, value_and_action.first);
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
  std::pair<double, Action> value_and_action =
      AlphaBetaSearch(*game, state.get(), {}, -1, kInvalidPlayer);
  SPIEL_CHECK_EQ(1.0, value_and_action.first);
}

void AlphaBetaSearchTest_TicTacToe_Loss() {
  std::shared_ptr<const Game> game = LoadGame("tic_tac_toe");
  std::unique_ptr<State> state = game->NewInitialState();

  // Construct:
  // ...
  // xox
  // ..o
  state->ApplyAction(5);
  state->ApplyAction(4);
  state->ApplyAction(3);
  state->ApplyAction(8);

  std::pair<double, Action> value_and_action =
      AlphaBetaSearch(*game, state.get(), {}, -1, kInvalidPlayer);
  SPIEL_CHECK_EQ(-1.0, value_and_action.first);
}

int FirstPlayerAdvantage(const State& state) {
  const auto& pstate = down_cast<const open_spiel::pig::PigState&>(state);
  return pstate.score(0) - pstate.score(1);
}

void ExpectiminimaxSearchTest_Pig() {
  std::shared_ptr<const Game> game =
      LoadGame("pig", {{"diceoutcomes", GameParameter(3)}});
  std::pair<double, Action> value_and_action = ExpectiminimaxSearch(
      *game, nullptr, FirstPlayerAdvantage, 2, kInvalidPlayer);
  SPIEL_CHECK_EQ(1.0 / 3 * 2 + 1.0 / 3 * 3, value_and_action.first);
  SPIEL_CHECK_EQ(/*kRoll=*/0, value_and_action.second);
}

}  // namespace
}  // namespace algorithms
}  // namespace open_spiel

int main(int argc, char **argv) {
  open_spiel::algorithms::AlphaBetaSearchTest_TicTacToe();
  open_spiel::algorithms::AlphaBetaSearchTest_TicTacToe_Win();
  open_spiel::algorithms::AlphaBetaSearchTest_TicTacToe_Loss();
  open_spiel::algorithms::ExpectiminimaxSearchTest_Pig();
}
