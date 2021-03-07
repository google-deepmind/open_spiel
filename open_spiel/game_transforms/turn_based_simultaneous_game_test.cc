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

#include "open_spiel/game_transforms/turn_based_simultaneous_game.h"

#include <random>
#include <string>

#include "open_spiel/abseil-cpp/absl/random/uniform_int_distribution.h"
#include "open_spiel/spiel.h"

namespace open_spiel {
namespace {

void SimulateGames(std::mt19937* rng, const Game& game, State* sim_state,
                   State* turn_based_state) {
  while (!sim_state->IsTerminal()) {
    const State* wrapped_sim_state =
        dynamic_cast<TurnBasedSimultaneousState*>(turn_based_state)
            ->SimultaneousGameState();

    // Now check that the states are identical via the ToString().
    SPIEL_CHECK_EQ(sim_state->ToString(), wrapped_sim_state->ToString());

    if (sim_state->IsChanceNode()) {
      SPIEL_CHECK_TRUE(turn_based_state->IsChanceNode());

      // Chance node; sample one according to underlying distribution
      std::vector<std::pair<Action, double>> outcomes =
          sim_state->ChanceOutcomes();
      Action action =
          open_spiel::SampleAction(
              outcomes, std::uniform_real_distribution<double>(0.0, 1.0)(*rng))
              .first;

      std::cout << "sampled outcome: %s\n"
                << sim_state->ActionToString(kChancePlayerId, action)
                << std::endl;

      sim_state->ApplyAction(action);
      turn_based_state->ApplyAction(action);
    } else if (sim_state->CurrentPlayer() == kSimultaneousPlayerId) {
      SPIEL_CHECK_EQ(wrapped_sim_state->CurrentPlayer(), kSimultaneousPlayerId);

      // Players choose simultaneously.
      std::vector<Action> joint_action;

      // Sample an action for each player
      for (auto p = Player{0}; p < game.NumPlayers(); p++) {
        if (game.GetType().provides_information_state_string) {
          // Check the information states to each player are consistent.
          SPIEL_CHECK_EQ(sim_state->InformationStateString(p),
                         wrapped_sim_state->InformationStateString(p));
        }

        std::vector<Action> actions;
        actions = sim_state->LegalActions(p);
        absl::uniform_int_distribution<> dis(0, actions.size() - 1);
        Action action = actions[dis(*rng)];
        joint_action.push_back(action);
        std::cout << "player " << p << " chose "
                  << sim_state->ActionToString(p, action) << std::endl;
        SPIEL_CHECK_EQ(p, turn_based_state->CurrentPlayer());
        turn_based_state->ApplyAction(action);
      }

      sim_state->ApplyActions(joint_action);
    } else {
      SPIEL_CHECK_EQ(sim_state->CurrentPlayer(),
                     wrapped_sim_state->CurrentPlayer());
      SPIEL_CHECK_EQ(sim_state->CurrentPlayer(),
                     turn_based_state->CurrentPlayer());

      Player p = sim_state->CurrentPlayer();

      if (game.GetType().provides_information_state_string) {
        // Check the information states to each player are consistent.
        SPIEL_CHECK_EQ(sim_state->InformationStateString(p),
                       wrapped_sim_state->InformationStateString(p));
      }

      std::vector<Action> actions;
      actions = sim_state->LegalActions(p);
      absl::uniform_int_distribution<> dis(0, actions.size() - 1);
      Action action = actions[dis(*rng)];

      std::cout << "player " << p << " chose "
                << sim_state->ActionToString(p, action) << std::endl;

      turn_based_state->ApplyAction(action);
      sim_state->ApplyAction(action);
    }

    std::cout << "State: " << std::endl << sim_state->ToString() << std::endl;
  }

  SPIEL_CHECK_TRUE(turn_based_state->IsTerminal());

  auto sim_returns = sim_state->Returns();
  auto turn_returns = turn_based_state->Returns();

  for (auto player = Player{0}; player < game.NumPlayers(); player++) {
    double utility = sim_returns[player];
    SPIEL_CHECK_GE(utility, game.MinUtility());
    SPIEL_CHECK_LE(utility, game.MaxUtility());
    std::cout << "Utility to player " << player << " is " << utility
              << std::endl;

    double other_utility = turn_returns[player];
    SPIEL_CHECK_EQ(utility, other_utility);
  }
}

void BasicTurnBasedSimultaneousTests() {
  std::mt19937 rng;

  for (const GameType& type : RegisteredGameTypes()) {
    if (!type.ContainsRequiredParameters() && type.default_loadable) {
      std::string name = type.short_name;
      if (type.dynamics == GameType::Dynamics::kSimultaneous) {
        std::cout << "TurnBasedSimultaneous: Testing " << name << std::endl;
        for (int i = 0; i < 100; ++i) {
          std::shared_ptr<const Game> sim_game = LoadGame(name);
          std::shared_ptr<const Game> turn_based_game =
              ConvertToTurnBased(*LoadGame(name));
          auto sim_state = sim_game->NewInitialState();
          auto turn_based_state = turn_based_game->NewInitialState();
          SimulateGames(&rng, *sim_game, sim_state.get(),
                        turn_based_state.get());
        }
      }
    }
  }
}

}  // namespace
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::BasicTurnBasedSimultaneousTests();
}
