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

#include "open_spiel/spiel_bots.h"

namespace open_spiel {

std::vector<double> EvaluateBots(State* state, const std::vector<Bot*>& bots,
                                 int seed) {
  const int num_players = bots.size();
  std::mt19937 rng(seed);
  std::uniform_real_distribution<> uniform(0, 1);
  std::vector<Action> joint_actions(bots.size());
  for (auto bot : bots) bot->Restart(*state);
  while (!state->IsTerminal()) {
    if (state->IsChanceNode()) {
      state->ApplyAction(
          SampleChanceOutcome(state->ChanceOutcomes(), uniform(rng)));
    } else if (state->IsSimultaneousNode()) {
      for (int i = 0; i < num_players; ++i) {
        if (state->LegalActions(i).empty()) {
          joint_actions[i] = kInvalidAction;
        } else {
          joint_actions[i] = bots[i]->Step(*state).second;
        }
      }
      state->ApplyActions(joint_actions);
    } else {
      state->ApplyAction(bots[state->CurrentPlayer()]->Step(*state).second);
    }
  }

  // Return terminal utility.
  return state->Returns();
}

}  // namespace open_spiel
