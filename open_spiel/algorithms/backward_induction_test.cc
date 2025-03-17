// Copyright 2023 DeepMind Technologies Limited
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

#include "open_spiel/algorithms/backward_induction.h"

#include <memory>
#include <string>
#include <vector>

#include "open_spiel/games/tic_tac_toe/tic_tac_toe.h"
#include "open_spiel/games/goofspiel/goofspiel.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace algorithms {
namespace {

void BackwardInductionTest_TicTacToe() {
  std::shared_ptr<const Game> game = LoadGame("tic_tac_toe");
  std::vector<double> values = BackwardInductionValues(*game);
  
  // Tic-tac-toe is a draw with optimal play
  SPIEL_CHECK_EQ(values[0], 0.0);
  SPIEL_CHECK_EQ(values[1], 0.0);
  
  // Create a state with an obvious winning move for the first player
  std::unique_ptr<State> state = game->NewInitialState();
  // X . .
  // X O .
  // O . .
  state->ApplyAction(0);  // X in top-left
  state->ApplyAction(4);  // O in middle
  state->ApplyAction(3);  // X in middle-left
  state->ApplyAction(6);  // O in bottom-left
  
  // Now X can win by playing in top-right
  auto [result_values, policy] = BackwardInduction(*game, state.get());
  SPIEL_CHECK_EQ(result_values[0], 1.0);  // X wins
  SPIEL_CHECK_EQ(result_values[1], -1.0);  // O loses
  
  // The best action for X should be the winning move (top-right = position 2)
  SPIEL_CHECK_EQ(policy[state->ToString()], 2);
}

void BackwardInductionTest_TicTacToe_AllOptimalActions() {
  std::shared_ptr<const Game> game = LoadGame("tic_tac_toe");
  
  // Create a new game
  std::unique_ptr<State> state = game->NewInitialState();
  
  // Get all optimal actions from the initial state
  auto [values, all_actions] = BackwardInductionAllOptimalActions(*game, state.get());
  
  // With optimal play, tic-tac-toe is a draw
  SPIEL_CHECK_EQ(values[0], 0.0);
  SPIEL_CHECK_EQ(values[1], 0.0);
  
  // From the initial state, the center and corners are all optimal
  // (there should be multiple optimal first moves)
  const auto& initial_actions = all_actions[state->ToString()];
  SPIEL_CHECK_GE(initial_actions.size(), 1);
  
  // Test tie-breaking policies
  auto [first_values, first_policy] = BackwardInduction(
      *game, state.get(), TieBreakingPolicy::kFirstAction);
  auto [last_values, last_policy] = BackwardInduction(
      *game, state.get(), TieBreakingPolicy::kLastAction);
  
  // The values should be the same regardless of tie-breaking
  SPIEL_CHECK_EQ(first_values[0], 0.0);
  SPIEL_CHECK_EQ(first_values[1], 0.0);
  SPIEL_CHECK_EQ(last_values[0], 0.0);
  SPIEL_CHECK_EQ(last_values[1], 0.0);
  
  // But the chosen actions might be different if there are multiple optimal actions
  if (initial_actions.size() > 1) {
    Action first_action = first_policy[state->ToString()];
    Action last_action = last_policy[state->ToString()];
    // Check that both actions are in the set of all optimal actions
    SPIEL_CHECK_TRUE(std::find(initial_actions.begin(), initial_actions.end(), 
                            first_action) != initial_actions.end());
    SPIEL_CHECK_TRUE(std::find(initial_actions.begin(), initial_actions.end(), 
                            last_action) != initial_actions.end());
  }
}

// Test backward induction on a sequential-move variant of Goofspiel
void BackwardInductionTest_SequentialGoofspiel() {
  // Create a small sequential version of Goofspiel
  GameParameters params;
  params["num_cards"] = GameParameter(3);
  params["points_order"] = GameParameter(std::string("descending"));
  params["returns_type"] = GameParameter(std::string("win_loss"));
  params["players"] = GameParameter(2);
  
  // Important: use sequential to make it a perfect information game
  params["imp_info"] = GameParameter(false);
  
  std::shared_ptr<const Game> game = LoadGame("goofspiel", params);
  SPIEL_CHECK_EQ(game->GetType().information, 
                 GameType::Information::kPerfectInformation);
  
  auto [values, policy] = BackwardInduction(*game);
  
  // With optimal play, the game is either a win for player 0, a draw, or 
  // a win for player 1. We don't verify the exact value, just that it's valid.
  SPIEL_CHECK_GE(values[0], -1.0);
  SPIEL_CHECK_LE(values[0], 1.0);
  SPIEL_CHECK_GE(values[1], -1.0);
  SPIEL_CHECK_LE(values[1], 1.0);
  
  // Check that the values are consistent (zero-sum)
  SPIEL_CHECK_EQ(values[0], -values[1]);
}

}  // namespace
}  // namespace algorithms
}  // namespace open_spiel

int main(int argc, char **argv) {
  open_spiel::algorithms::BackwardInductionTest_TicTacToe();
  open_spiel::algorithms::BackwardInductionTest_TicTacToe_AllOptimalActions();
  open_spiel::algorithms::BackwardInductionTest_SequentialGoofspiel();
  
  // If we made it here without any CHECK/DCHECK failures, the test passed.
  return 0;
} 