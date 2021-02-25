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

#include "open_spiel/games/leduc_poker_dummy.h"
#include "open_spiel/policy.h"
#include "open_spiel/spiel.h"
#include "open_spiel/tests/basic_tests.h"

namespace open_spiel {
namespace leduc_poker_dummy {
namespace {

namespace testing = open_spiel::testing;

void BasicLeducTests() {
  testing::LoadGameTest("leduc_poker_dummy");
  testing::ChanceOutcomesTest(*LoadGame("leduc_poker_dummy"));
  testing::RandomSimTest(*LoadGame("leduc_poker_dummy"), 100);
  testing::RandomSimTest(*LoadGame("leduc_poker_dummy",
                         {{"action_mapping", GameParameter(true)}}), 100);
  testing::RandomSimTest(*LoadGame("leduc_poker_dummy",
                         {{"suit_isomorphism", GameParameter(true)}}), 100);
  for (Player players = 3; players <= 5; players++) {
    testing::RandomSimTest(
        *LoadGame("leduc_poker_dummy", {{"players", GameParameter(players)}}), 100);
  }
  testing::ResampleInfostateTest(*LoadGame("leduc_poker_dummy"), /*num_sims=*/100);
  auto observer = LoadGame("leduc_poker_dummy")
                      ->MakeObserver(kDefaultObsType,
                                     GameParametersFromString("single_tensor"));
  testing::RandomSimTestCustomObserver(*LoadGame("leduc_poker_dummy"), observer);
}

void PolicyTest() {
  using PolicyGenerator = std::function<TabularPolicy(const Game& game)>;
  std::vector<PolicyGenerator> policy_generators = {
      GetAlwaysFoldPolicy,
      GetAlwaysCallPolicy,
      GetAlwaysRaisePolicy
  };

  std::shared_ptr<const Game> game = LoadGame("leduc_poker_dummy");
  for (const auto& policy_generator : policy_generators) {
    testing::TestEveryInfostateInPolicy(policy_generator, *game);
    testing::TestPoliciesCanPlay(policy_generator, *game);
  }
}

}  // namespace
}  // namespace leduc_poker_dummy
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::leduc_poker_dummy::BasicLeducTests();
  open_spiel::leduc_poker_dummy::PolicyTest();
}
