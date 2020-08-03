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

#include "open_spiel/games/kuhn_poker.h"

#include <iostream>
#include <memory>
#include <utility>

#include "open_spiel/public_states/games/kuhn_poker.h"
#include "open_spiel/public_states/public_states.h"

namespace open_spiel {
namespace public_states {

// TODO(author13): we need to implement many more tests.
// High-level items:
// - Regression playthroughs.
// - Equivalence tests to the Base API.
// - Resampling is consistent.
// - Correct propagation of cf. values.
// - Correct update of reach probs.
// - Public state type one of chance/player/terminal.
// - Terminal values: GetPublicSet at terminal public state
//     -> Returns() * Chance reaches
//     == TerminalCfValues of appropriate reach probs.

namespace kuhn_poker {
namespace {

std::vector<std::vector<Action>> StatesToHistories(
    const std::vector<std::unique_ptr<State>>& states) {
  std::vector<std::vector<Action>> histories;
  histories.reserve(states.size());
  for (const auto& state : states) {
    histories.push_back(state->History());
  }
  return histories;
}

void CheckPublicSet(PublicState* s,
                    std::vector<std::vector<Action>> expected_histories) {
  std::vector<std::unique_ptr<State>> actual_set = s->GetPublicSet();
  std::vector<std::vector<Action>> actual_histories =
      StatesToHistories(actual_set);
  SPIEL_CHECK_EQ(actual_histories, expected_histories);
}

void TestGeneratePublicSets() {
  std::shared_ptr<const GameWithPublicStates> game =
      LoadGameWithPublicStates("kuhn_poker(players=2)");

  // We use down_cast for easier navigation in IDEs
  std::unique_ptr<PublicState> public_state = game->NewInitialPublicState();
  auto* s = down_cast<KuhnPublicState*>(public_state.get());

  SPIEL_CHECK_TRUE(s->IsChance());
  CheckPublicSet(s, {{}});  // Start of game
  s->ApplyPublicTransition("Deal to player 0");

  SPIEL_CHECK_TRUE(s->IsChance());
  CheckPublicSet(s, {{2}, {1}, {0}});
  s->ApplyPublicTransition("Deal to player 1");

  SPIEL_CHECK_TRUE(s->IsPlayer());
  SPIEL_CHECK_EQ(s->ActingPlayers(), std::vector<int>{0});
  CheckPublicSet(s, {{2, 1}, {2, 0}, {1, 2}, {1, 0}, {0, 1}, {0, 2}});
  s->ApplyPublicTransition("Pass");

  SPIEL_CHECK_TRUE(s->IsPlayer());
  SPIEL_CHECK_EQ(s->ActingPlayers(), std::vector<int>{1});
  CheckPublicSet(s, {{2, 1, 0}, {2, 0, 0}, {1, 2, 0},
                     {1, 0, 0}, {0, 1, 0}, {0, 2, 0}});
  s->ApplyPublicTransition("Bet");

  SPIEL_CHECK_TRUE(s->IsPlayer());
  SPIEL_CHECK_EQ(s->ActingPlayers(), std::vector<int>{0});
  CheckPublicSet(s, {{2, 1, 0, 1}, {2, 0, 0, 1}, {1, 2, 0, 1},
                     {1, 0, 0, 1}, {0, 1, 0, 1}, {0, 2, 0, 1}});
  s->ApplyPublicTransition("Bet");

  SPIEL_CHECK_TRUE(s->IsTerminal());
  CheckPublicSet(s, {{2, 1, 0, 1, 1}, {2, 0, 0, 1, 1}, {1, 2, 0, 1, 1},
                     {1, 0, 0, 1, 1}, {0, 1, 0, 1, 1}, {0, 2, 0, 1, 1}});
}

}  // namespace
}  // namespace kuhn_poker
}  // namespace public_states
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::public_states::kuhn_poker::TestGeneratePublicSets();
}
