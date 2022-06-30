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

#include "open_spiel/algorithms/mcts.h"

#include <memory>
#include <utility>

#include "open_spiel/abseil-cpp/absl/strings/string_view.h"
#include "open_spiel/algorithms/evaluate_bots.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_bots.h"
#include "open_spiel/spiel_utils.h"

using open_spiel::algorithms::Evaluator;
using open_spiel::algorithms::RandomRolloutEvaluator;

namespace open_spiel {
namespace {

constexpr double UCT_C = 2;

std::unique_ptr<open_spiel::Bot> InitBot(const open_spiel::Game& game,
                                         int max_simulations,
                                         std::shared_ptr<Evaluator> evaluator) {
  return std::make_unique<open_spiel::algorithms::MCTSBot>(
      game, std::move(evaluator), UCT_C, max_simulations,
      /*max_memory_mb=*/5, /*solve=*/true, /*seed=*/42, /*verbose=*/false);
}

void MCTSTest_CanPlayTicTacToe() {
  auto game = LoadGame("tic_tac_toe");
  int max_simulations = 100;
  auto evaluator = std::make_shared<RandomRolloutEvaluator>(20, 42);
  auto bot0 = InitBot(*game, max_simulations, evaluator);
  auto bot1 = InitBot(*game, max_simulations, evaluator);
  auto results =
      EvaluateBots(game->NewInitialState().get(), {bot0.get(), bot1.get()}, 42);
  SPIEL_CHECK_EQ(results[0] + results[1], 0);
}

void MCTSTest_CanPlayTicTacToe_LowSimulations() {
  auto game = LoadGame("tic_tac_toe");
  // Setting max_simulations to 0 or 1 is equivalent to sampling from the prior.
  for (const int max_simulations : {0, 1}) {
    auto evaluator = std::make_shared<RandomRolloutEvaluator>(20, 42);
    auto bot0 = InitBot(*game, max_simulations, evaluator);
    auto bot1 = InitBot(*game, max_simulations, evaluator);
    auto results = EvaluateBots(game->NewInitialState().get(),
                                {bot0.get(), bot1.get()}, 42);
    SPIEL_CHECK_EQ(results[0] + results[1], 0);
  }
}

void MCTSTest_CanPlayBothSides() {
  auto game = LoadGame("tic_tac_toe");
  int max_simulations = 100;
  auto evaluator = std::make_shared<RandomRolloutEvaluator>(20, 42);
  auto bot = InitBot(*game, max_simulations, evaluator);
  auto results =
      EvaluateBots(game->NewInitialState().get(), {bot.get(), bot.get()}, 42);
  SPIEL_CHECK_EQ(results[0] + results[1], 0);
}

void MCTSTest_CanPlaySinglePlayer() {
  auto game = LoadGame("catch");
  int max_simulations = 100;
  auto evaluator = std::make_shared<RandomRolloutEvaluator>(20, 42);
  auto bot = InitBot(*game, max_simulations, evaluator);
  auto results = EvaluateBots(game->NewInitialState().get(), {bot.get()}, 42);
  SPIEL_CHECK_GT(results[0], 0);
}

void MCTSTest_CanPlayThreePlayerStochasticGames() {
  auto game = LoadGame("pig(players=3,winscore=20,horizon=30)");
  int max_simulations = 1000;
  auto evaluator = std::make_shared<RandomRolloutEvaluator>(20, 42);
  auto bot0 = InitBot(*game, max_simulations, evaluator);
  auto bot1 = InitBot(*game, max_simulations, evaluator);
  auto bot2 = InitBot(*game, max_simulations, evaluator);
  auto results = EvaluateBots(game->NewInitialState().get(),
                              {bot0.get(), bot1.get(), bot2.get()}, 42);
  SPIEL_CHECK_FLOAT_EQ(results[0] + results[1] + results[2], 0);
}

open_spiel::Action GetAction(const open_spiel::State& state,
                             const absl::string_view action_str) {
  for (open_spiel::Action action : state.LegalActions()) {
    if (action_str == state.ActionToString(state.CurrentPlayer(), action))
      return action;
  }
  open_spiel::SpielFatalError(absl::StrCat("Illegal action: ", action_str));
}

std::pair<std::unique_ptr<algorithms::SearchNode>, std::unique_ptr<State>>
SearchTicTacToeState(const absl::string_view initial_actions) {
  auto game = LoadGame("tic_tac_toe");
  std::unique_ptr<State> state = game->NewInitialState();
  for (const auto& action_str : absl::StrSplit(initial_actions, ' ')) {
    state->ApplyAction(GetAction(*state, action_str));
  }
  auto evaluator = std::make_shared<RandomRolloutEvaluator>(20, 42);
  algorithms::MCTSBot bot(*game, evaluator, UCT_C,
                          /*max_simulations=*/ 10000,
                          /*max_memory_mb=*/ 10,
                          /*solve=*/ true,
                          /*seed=*/ 42,
                          /*verbose=*/ false);
  return {bot.MCTSearch(*state), std::move(state)};
}

void MCTSTest_SolveDraw() {
  auto [root, state] = SearchTicTacToeState("x(1,1) o(0,0) x(2,2)");
  SPIEL_CHECK_EQ(state->ToString(), "o..\n.x.\n..x");
  SPIEL_CHECK_EQ(root->outcome[root->player], 0);
  for (const algorithms::SearchNode& c : root->children)
    SPIEL_CHECK_LE(c.outcome[c.player], 0);  // No winning moves.
  const algorithms::SearchNode& best = root->BestChild();
  SPIEL_CHECK_EQ(best.outcome[best.player], 0);
  std::string action_str = state->ActionToString(best.player, best.action);
  if (action_str != "o(2,0)" && action_str != "o(0,2)")  // All others lose.
    SPIEL_CHECK_EQ(action_str, "o(2,0)");  // "o(0,2)" is also valid.
}

void MCTSTest_SolveLoss() {
  auto [root, state] =
      SearchTicTacToeState("x(1,1) o(0,0) x(2,2) o(0,1) x(0,2)");
  SPIEL_CHECK_EQ(state->ToString(), "oox\n.x.\n..x");
  SPIEL_CHECK_EQ(root->outcome[root->player], -1);
  for (const algorithms::SearchNode& c : root->children)
    SPIEL_CHECK_EQ(c.outcome[c.player], -1);  // All losses.
}

void MCTSTest_SolveWin() {
  auto [root, state] = SearchTicTacToeState("x(0,1) o(2,2)");
  SPIEL_CHECK_EQ(state->ToString(), ".x.\n...\n..o");
  SPIEL_CHECK_EQ(root->outcome[root->player], 1);
  const algorithms::SearchNode& best = root->BestChild();
  SPIEL_CHECK_EQ(best.outcome[best.player], 1);
  SPIEL_CHECK_EQ(state->ActionToString(best.player, best.action), "x(0,2)");
}

void MCTSTest_GarbageCollect() {
  auto game = LoadGame("tic_tac_toe");
  std::unique_ptr<State> state = game->NewInitialState();
  auto evaluator = std::make_shared<RandomRolloutEvaluator>(1, 42);
  algorithms::MCTSBot bot(*game, evaluator, UCT_C,
                          /*max_simulations=*/ 1000000,
                          /*max_memory_mb=*/ 1,
                          /*solve=*/ true,
                          /*seed=*/ 42,
                          /*verbose=*/ true);  // Verify the log output.
  std::unique_ptr<algorithms::SearchNode> root = bot.MCTSearch(*state);
  SPIEL_CHECK_TRUE(root->outcome.size() == 2 ||
                   root->explore_count == 1000000);
}

}  // namespace
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::MCTSTest_CanPlayTicTacToe();
  open_spiel::MCTSTest_CanPlayTicTacToe_LowSimulations();
  open_spiel::MCTSTest_CanPlayBothSides();
  open_spiel::MCTSTest_CanPlaySinglePlayer();
  open_spiel::MCTSTest_CanPlayThreePlayerStochasticGames();
  open_spiel::MCTSTest_SolveDraw();
  open_spiel::MCTSTest_SolveLoss();
  open_spiel::MCTSTest_SolveWin();
  open_spiel::MCTSTest_GarbageCollect();
}
