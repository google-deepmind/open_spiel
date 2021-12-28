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

#ifndef OPEN_SPIEL_TESTS_BASIC_TESTS_H_
#define OPEN_SPIEL_TESTS_BASIC_TESTS_H_

#include <random>
#include <string>

#include "open_spiel/game_parameters.h"
#include "open_spiel/policy.h"
#include "open_spiel/spiel.h"

namespace open_spiel {
namespace testing {

constexpr int kDefaultNumSimsForPolicyTests = 10;

// Default state checker function (does nothing).
void DefaultStateChecker(const State& state);

// Checks that the game can be loaded.
void LoadGameTest(const std::string& game_name);

// Test to ensure that there are chance outcomes.
void ChanceOutcomesTest(const Game& game);

// Test to ensure that there are no chance outcomes.
void NoChanceOutcomesTest(const Game& game);

// Perform num_sims random simulations of the specified game. The optional
// state_checker_fn is called at every state (including chance nodes and
// terminals), and is intended to be an easy way to pass context-specific
// testing functions to the simulation tests.
void RandomSimTest(const Game& game, int num_sims, bool serialize = true,
                   bool verbose = true, bool mask_test = true,
                   const std::function<void(const State&)>& state_checker_fn =
                       &DefaultStateChecker,
                   int mean_field_population = -1);

// Perform num_sims random simulations of the specified game. Also tests the
// Undo function. Note: for every step in the simulation, the entire simulation
// up to that point is rolled backward all the way to the beginning via undo,
// checking that the states match the ones along the history. Therefore, this
// is very slow! Please use sparingly.
void RandomSimTestWithUndo(const Game& game, int num_sims);

// Check that chance outcomes are valid and consistent.
// Performs an exhaustive search of the game tree, so should only be
// used for smallish games.
void CheckChanceOutcomes(const Game& game);

// Same as above but without checking the serialization functions. Every game
// should support serialization: only use this function when developing a new
// game, in order to test the implementation using the basic tests before having
// to implement the custom serialization (only useful for games that have chance
// mode kSampledStochastic).
void RandomSimTestNoSerialize(const Game& game, int num_sims);

void RandomSimTestCustomObserver(const Game& game,
                                 const std::shared_ptr<Observer> observer);
// Verifies that ResampleFromInfostate is correctly implemented.
void ResampleInfostateTest(const Game& game, int num_sims);

using TabularPolicyGenerator = std::function<TabularPolicy(const Game&)>;

void TestPoliciesCanPlay(
    TabularPolicyGenerator policy_generator, const Game& game,
    int numSims = kDefaultNumSimsForPolicyTests);
void TestPoliciesCanPlay(
    const Policy& policy, const Game& game,
    int numSims = kDefaultNumSimsForPolicyTests);
void TestEveryInfostateInPolicy(TabularPolicyGenerator policy_generator,
    const Game& game);

// Checks that the legal actions list is sorted.
void CheckLegalActionsAreSorted(const Game& game, State& state);

}  // namespace testing
}  // namespace open_spiel

#endif  // OPEN_SPIEL_TESTS_BASIC_TESTS_H_
