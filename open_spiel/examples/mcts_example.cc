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

#include <memory>
#include <random>
#include <string>

#include "open_spiel/abseil-cpp/absl/flags/flag.h"
#include "open_spiel/abseil-cpp/absl/flags/parse.h"
#include "open_spiel/algorithms/mcts.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

ABSL_FLAG(std::string, game, "tic_tac_toe", "The name of the game to play.");
ABSL_FLAG(open_spiel::Player, mcts_player, 0, "The player to play with mcts.");
ABSL_FLAG(int, rollout_count, 100, "How many rollouts per evaluation.");
ABSL_FLAG(int, max_search_nodes, 100, "How many search nodes to expand.");

// Example code for using MCTS agent to play a game
int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);

  std::string game_name = absl::GetFlag(FLAGS_game);
  auto mcts_player = absl::GetFlag(FLAGS_mcts_player);
  int rollout_count = absl::GetFlag(FLAGS_rollout_count);
  int max_search_nodes = absl::GetFlag(FLAGS_max_search_nodes);

  std::cerr << "game: " << game_name << std::endl;
  // Exploration parameter for UCT
  const double uct_c = 2;

  // Create the game
  open_spiel::GameParameters params;
  std::shared_ptr<const open_spiel::Game> game =
      open_spiel::LoadGame(game_name, params);

  std::unique_ptr<open_spiel::State> state = game->NewInitialState();

  // Check that the game satisfies conditions for the implemented MCTS
  // algorithm.
  SPIEL_CHECK_TRUE(game->NumPlayers() == 1 || game->NumPlayers() == 2);
  if (game->NumPlayers() == 2) {
    SPIEL_CHECK_EQ(game->GetType().utility,
                   open_spiel::GameType::Utility::kZeroSum);
  }

  // Create MCTS Bot
  open_spiel::algorithms::RandomRolloutEvaluator evaluator(rollout_count);
  open_spiel::algorithms::MCTSBot mcts_bot(*game, mcts_player, uct_c,
                                           max_search_nodes, evaluator);

  // Create Random Bot
  std::shared_ptr<open_spiel::Bot> random_bot =
      open_spiel::MakeUniformRandomBot(
          *game, open_spiel::Player{1 - mcts_player}, /*seed=*/1234);

  open_spiel::Bot* bots[2];
  if (mcts_player == open_spiel::Player{0}) {
    bots[0] = &mcts_bot;
    bots[1] = random_bot.get();
  } else {
    bots[1] = &mcts_bot;
    bots[0] = random_bot.get();
  }

  // Random number generator.
  std::mt19937 rng;

  while (!state->IsTerminal()) {
    std::cerr << "player " << state->CurrentPlayer() << std::endl;

    if (state->IsChanceNode()) {
      // Chance node; sample one according to underlying distribution.
      std::vector<std::pair<open_spiel::Action, double>> outcomes =
          state->ChanceOutcomes();
      open_spiel::Action action = open_spiel::SampleChanceOutcome(
          outcomes, std::uniform_real_distribution<double>(0.0, 1.0)(rng));
      std::cerr << "sampled outcome: "
                << state->ActionToString(open_spiel::kChancePlayerId, action)
                << std::endl;
      state->ApplyAction(action);
    } else if (state->IsSimultaneousNode()) {
      open_spiel::SpielFatalError(
          "MCTS not supported for games with simultaneous actions.");
    } else {
      // Decision node, ask the right bot to make its action
      open_spiel::Player player = state->CurrentPlayer();
      if (game->GetType().provides_information_state_as_normalized_vector) {
        std::vector<double> infostate;
        state->InformationStateAsNormalizedVector(player, &infostate);
        std::cerr << "player " << player << ": "
                  << absl::StrJoin(infostate, " ") << std::endl;
      }
      if (game->GetType().provides_information_state) {
        std::cerr << "player " << player << ": "
                  << state->InformationState(player) << std::endl;
      }

      auto bot_choice = bots[player]->Step(*state);
      auto action = bot_choice.second;
      std::cerr << "chose action: " << state->ActionToString(player, action)
                << std::endl;
      state->ApplyAction(action);
    }

    std::cerr << "State: " << std::endl << state->ToString() << std::endl;
  }

  auto returns = state->Returns();
  for (int p = 0; p < game->NumPlayers(); p++) {
    std::cerr << "Terminal return to player " << p << " is " << returns[p]
              << std::endl;
  }
}
