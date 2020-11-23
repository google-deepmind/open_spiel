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

// Copyright 2020 DeepMind Technologies Ltd. All rights reserved.
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

#include "open_spiel/games/sheriff.h"

#include <iostream>
#include <limits>

#include "open_spiel/abseil-cpp/absl/container/flat_hash_set.h"
#include "open_spiel/algorithms/expected_returns.h"
#include "open_spiel/algorithms/get_all_states.h"
#include "open_spiel/algorithms/tabular_exploitability.h"
#include "open_spiel/policy.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/tests/basic_tests.h"

namespace open_spiel {
namespace sheriff {
namespace {

namespace testing = open_spiel::testing;

void BasicSheriffTest() {
  for (int num_rounds = 1; num_rounds <= 6; ++num_rounds) {
    const std::shared_ptr<const Game> game =
        LoadGame("sheriff", {{"item_penalty", GameParameter(2.0)},
                             {"item_value", GameParameter(1.5)},
                             {"sheriff_penalty", GameParameter(3.14)},
                             {"max_bribe", GameParameter(10)},
                             {"max_items", GameParameter(10)},
                             {"num_rounds", GameParameter(num_rounds)}});
    testing::RandomSimTestWithUndo(*game, 100);
    testing::NoChanceOutcomesTest(*game);
  }
}

}  // namespace
}  // namespace sheriff
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::testing::LoadGameTest("sheriff");
}
