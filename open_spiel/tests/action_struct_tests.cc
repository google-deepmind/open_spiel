// Copyright 2025 DeepMind Technologies Limited
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

#include "open_spiel/tests/action_struct_tests.h"

#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "open_spiel/spiel.h"
#include "open_spiel/spiel_globals.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/utils/status.h"

namespace open_spiel {
namespace testing {

void SimulateRandomGame(std::shared_ptr<const open_spiel::Game> game,
                        int seed, bool serialize) {
  std::unique_ptr<State> state = game->NewInitialState();
  std::unique_ptr<ActionStructSampler> sampler =
      state->GetActionStructSampler(/*seed=*/seed);
  std::mt19937 rng(seed);

  while (!state->IsTerminal()) {
    if (serialize) {
      std::string serialized_state = state->Serialize();
      std::unique_ptr<State> deserialized_state =
          game->DeserializeState(serialized_state);
      SPIEL_CHECK_EQ(state->Serialize(), deserialized_state->Serialize());
    }

    if (state->IsChanceNode()) {
      std::vector<std::pair<open_spiel::Action, double>> outcomes =
          state->ChanceOutcomes();
      open_spiel::Action action = open_spiel::SampleAction(outcomes, rng).first;
      std::cerr << "sampled outcome: "
                << state->ActionToString(open_spiel::kChancePlayerId, action)
                << std::endl;
      state->ApplyAction(action);
    } else {
      std::unique_ptr<ActionStruct> action = sampler->SampleActionStruct();
      if (action == nullptr) {
        SpielFatalError("SimulateRandomGame: failed to sample action.");
      }
      Status status = state->ApplyActionStruct(*action);
      if (!status.ok()) {
        SpielFatalError("SimulateRandomGame: failed to apply action: " +
                        status.ToString());
      }
      std::cout << "State: \n" << state->ToString() << "\n";
    }
  }
}

}  // namespace testing
}  // namespace open_spiel
