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

#include "open_spiel/game_transforms/restricted_nash_response.h"

#include <string>

#include "open_spiel/abseil-cpp/absl/random/uniform_int_distribution.h"
#include "open_spiel/algorithms/cfr.h"
#include "open_spiel/game_transforms/turn_based_simultaneous_game.h"
#include "open_spiel/spiel.h"
#include "open_spiel/tests/basic_tests.h"
#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"

namespace open_spiel {
namespace {

void SimulateGame(std::mt19937 *rng, const Game &game,
                  std::unique_ptr<State> normal_state,
                  std::unique_ptr<State> rnr_state, bool fixed,
                  Player fixed_player) {
  // Now check that the states are identical via the ToString().
  std::string state_prefix =
      fixed ? "Rnr state string of state in fixed part with underlying state:\n"
            : "Rnr state string of state in free part with underlying state:\n";
  std::string infostate_prefix = fixed ? "[Rnr: fixed]" : "[Rnr: free]";
  while (!normal_state->IsTerminal()) {
    SPIEL_CHECK_EQ(state_prefix + normal_state->ToString(),
                   rnr_state->ToString());
    if (game.GetType().provides_information_state_string) {
      // Check the information states to each player are consistent.
      for (auto p = Player{0}; p < game.NumPlayers(); p++) {
        SPIEL_CHECK_EQ((p == fixed_player ? infostate_prefix : "") +
                           normal_state->InformationStateString(p),
                       rnr_state->InformationStateString(p));
      }
    }

    if (normal_state->IsChanceNode()) {
      SPIEL_CHECK_TRUE(rnr_state->IsChanceNode());

      // Chance node; sample one according to underlying distribution
      std::vector<std::pair<Action, double>> outcomes =
          normal_state->ChanceOutcomes();
      Action action =
          open_spiel::SampleAction(
              outcomes, std::uniform_real_distribution<double>(0.0, 1.0)(*rng))
              .first;

      normal_state->ApplyAction(action);
      rnr_state->ApplyAction(action);
    } else if (normal_state->CurrentPlayer() == kSimultaneousPlayerId) {
      SPIEL_CHECK_EQ(rnr_state->CurrentPlayer(), kSimultaneousPlayerId);

      // Players choose simultaneously.
      std::vector<Action> joint_action;

      // Sample an action for each player
      for (auto p = Player{0}; p < game.NumPlayers(); p++) {
        std::vector<Action> actions;
        actions = normal_state->LegalActions(p);
        absl::uniform_int_distribution<> dis(0, actions.size() - 1);
        Action action = actions[dis(*rng)];
        joint_action.push_back(action);
      }

      normal_state->ApplyActions(joint_action);
      rnr_state->ApplyActions(joint_action);
    } else {
      // Chance or player node
      SPIEL_CHECK_EQ(normal_state->CurrentPlayer(), rnr_state->CurrentPlayer());

      Player p = normal_state->CurrentPlayer();

      std::vector<Action> actions;
      actions = normal_state->LegalActions(p);
      absl::uniform_int_distribution<> dis(0, actions.size() - 1);
      Action action = actions[dis(*rng)];

      normal_state->ApplyAction(action);
      rnr_state->ApplyAction(action);
    }
  }

  SPIEL_CHECK_TRUE(rnr_state->IsTerminal());

  auto sim_returns = normal_state->Returns();
  auto turn_returns = rnr_state->Returns();

  for (auto player = Player{0}; player < sim_returns.size(); player++) {
    double utility = sim_returns[player];
    SPIEL_CHECK_GE(utility, game.MinUtility());
    SPIEL_CHECK_LE(utility, game.MaxUtility());

    double other_utility = turn_returns[player];
    SPIEL_CHECK_EQ(utility, other_utility);
  }
}

void BasicRNRTests() {
  for (const std::string& name :
     {"blotto", "goofspiel", "kuhn_poker", "tiny_hanabi", "phantom_ttt",
      "matrix_rps", "leduc_poker"}) {
    std::cout << "Basic RNR Test for " << name << std::endl;
    std::string full_game_str =
        absl::StrCat("restricted_nash_response(game=",
                                               name, "())");
    testing::RandomSimTest(*LoadGame(full_game_str), 10, /*serialize*/false,
                           /*verbose*/false, /*mask_test*/true);
  }
}

void TestBasicCreation() {
  std::mt19937 rng;

  // Create different games for RNR and check the simulation
  for (const std::string& name :
       {"blotto", "goofspiel", "kuhn_poker", "tiny_hanabi", "phantom_ttt",
        "matrix_rps", "leduc_poker"}) {
    std::cout << "RestrictedNashResponse: Testing " << name << std::endl;
    for (Player fixed_player = 0; fixed_player < 2; fixed_player++) {
      for (int i = 0; i < 100; ++i) {
        std::shared_ptr<const Game> normal_game_game = LoadGame(name);
        std::shared_ptr<const Game> rnr_game =
            ConvertToRNR(*LoadGame(name), fixed_player, 0.5);
        auto normal_init_fixed = normal_game_game->NewInitialState();
        auto rnr_init_fixed = rnr_game->NewInitialState();
        rnr_init_fixed->ApplyAction(Action(kFixedAction));
        SimulateGame(&rng, *normal_game_game, std::move(normal_init_fixed),
                     std::move(rnr_init_fixed), true, fixed_player);

        auto rnr_init_free = rnr_game->NewInitialState();
        auto normal_init_free = normal_game_game->NewInitialState();
        rnr_init_free->ApplyAction(Action(kFreeAction));
        SimulateGame(&rng, *normal_game_game, std::move(normal_init_free),
                     std::move(rnr_init_free), false, fixed_player);
      }
    }
  }
}

void TestMatchingPenniesCreation() {
  // Check the creation of matching pennies game
  Player fixed_player = 1;
  std::shared_ptr<const Game> game = LoadGame("matrix_mp");
  std::shared_ptr<const Game> rnr_game = ConvertToRNR(*game, fixed_player, 0.4);
  SPIEL_CHECK_EQ(game->MaxGameLength() + 1, rnr_game->MaxGameLength());
  SPIEL_CHECK_EQ(rnr_game->NumPlayers(), game->NumPlayers());
  SPIEL_CHECK_EQ(rnr_game->MaxUtility(), game->MaxUtility());
  SPIEL_CHECK_EQ(rnr_game->MinUtility(), game->MinUtility());
  auto state = rnr_game->NewInitialState();
  SPIEL_CHECK_EQ("Initial restricted Nash response state.", state->ToString());
  SPIEL_CHECK_EQ(state->LegalActions().size(), 2);

  auto chance_outcomes = state->ChanceOutcomes();
  SPIEL_CHECK_EQ(chance_outcomes[0].second, 0.4);
  SPIEL_CHECK_EQ(chance_outcomes[1].second, 0.6);

  // Fixed part
  auto fixed_child = state->Child(kFixedAction);
  SPIEL_CHECK_EQ(fixed_child->CurrentPlayer(), kSimultaneousPlayerId);

  // Free part
  auto free_child = state->Child(kFreeAction);
  SPIEL_CHECK_EQ(free_child->CurrentPlayer(), kSimultaneousPlayerId);

  for (Action joint_action : free_child->LegalActions()) {
    auto new_fixed_child = fixed_child->Child(joint_action);
    auto new_free_child = free_child->Child(joint_action);
    SPIEL_CHECK_EQ(new_fixed_child->Rewards(), new_free_child->Rewards());
    SPIEL_CHECK_EQ(new_fixed_child->InformationStateString(1 - fixed_player),
                   new_free_child->InformationStateString(1 - fixed_player));
    SPIEL_CHECK_NE(new_fixed_child->InformationStateString(fixed_player),
                   new_free_child->InformationStateString(fixed_player));
  }
}

void TestFixedPolicyGame() {
  // Check the RNR which automatically puts the strategy in the game as chance
  // nodes Setup
  Player fixed_player = 1;
  std::shared_ptr<const Game> game = LoadGameAsTurnBased("matrix_mp");
  std::shared_ptr<TabularPolicy> fixed_policy =
      std::make_shared<TabularPolicy>(*game);
  auto initial_state = game->NewInitialState();
  initial_state->ApplyAction(0);
  fixed_policy->SetStatePolicy(initial_state->InformationStateString(),
                               {{0, 1}, {1, 0}});
  // P 0.6 case when the resulting strategy is pure
  std::shared_ptr<const Game> rnr_game =
      ConvertToRNR(*game, fixed_player, 0.6, fixed_policy);
  algorithms::CFRPlusSolver solver(*rnr_game);
  for (int i = 0; i < 1000; i++) {
    solver.EvaluateAndUpdatePolicy();
  }
  const std::shared_ptr<Policy> average_policy = solver.AveragePolicy();
  auto player_two_policy = average_policy->GetStatePolicy(
      "[Rnr: free]Current player: 1\nObserving player: 1. Non-terminal");
  for (int i = 0; i < player_two_policy.size(); i++) {
    SPIEL_CHECK_FLOAT_NEAR(player_two_policy[i].second, i, 0.001);
  }
  auto player_one_policy = average_policy->GetStatePolicy(
      "Current player: 0\nObserving player: 0. Non-terminal");
  for (int i = 0; i < player_one_policy.size(); i++) {
    SPIEL_CHECK_FLOAT_NEAR(player_one_policy[i].second, 1 - i, 0.001);
  }
  // P 0.6 case when the resulting strategy is pure
  rnr_game = ConvertToRNR(*game, fixed_player, 0.4, fixed_policy);
  algorithms::CFRPlusSolver solver_two(*rnr_game);
  for (int i = 0; i < 1000; i++) {
    solver_two.EvaluateAndUpdatePolicy();
  }
  const std::shared_ptr<Policy> average_policy_two = solver_two.AveragePolicy();
  auto player_two_policy_two = average_policy_two->GetStatePolicy(
      "[Rnr: free]Current player: 1\nObserving player: 1. Non-terminal");
  double check_policy[] = {1. / 6, 5. / 6};
  for (int i = 0; i < player_two_policy_two.size(); i++) {
    SPIEL_CHECK_FLOAT_NEAR(player_two_policy_two[i].second, check_policy[i],
                           0.001);
  }
  auto player_one_policy_two = average_policy_two->GetStatePolicy(
      "Current player: 0\nObserving player: 0. Non-terminal");
  check_policy[0] = check_policy[1] = 0.5;
  for (int i = 0; i < player_one_policy_two.size(); i++) {
    SPIEL_CHECK_FLOAT_NEAR(player_one_policy_two[i].second, check_policy[i],
                           0.001);
  }
}

}  // namespace
}  // namespace open_spiel

int main(int argc, char **argv) {
  open_spiel::BasicRNRTests();
  open_spiel::TestBasicCreation();
  open_spiel::TestMatchingPenniesCreation();
  open_spiel::TestFixedPolicyGame();
}
