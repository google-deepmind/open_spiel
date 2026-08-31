// Copyright 2022 DeepMind Technologies Limited
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

#include "open_spiel/games/dou_dizhu/dou_dizhu.h"

#include <memory>
#include <random>
#include <vector>

#include "open_spiel/abseil-cpp/absl/algorithm/container.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/tests/basic_tests.h"

namespace open_spiel {
namespace dou_dizhu {
namespace {

void BasicGameTests() {
  testing::LoadGameTest("dou_dizhu");
  testing::RandomSimTest(*LoadGame("dou_dizhu"), 20);
}

// Regression test for
// https://github.com/google-deepmind/open_spiel/issues/1358.
// If every player passes during the auction then nobody becomes the dizhu.
// The terminal state's ToString() goes through OriginalDeal(), which used to
// hand the three left-over cards to dizhu_ == kInvalidPlayer and so wrote
// outside the deal array.
void AuctionAllPassTest() {
  std::shared_ptr<const Game> game = LoadGame("dou_dizhu");
  std::unique_ptr<State> state = game->NewInitialState();
  std::mt19937 rng(0);

  while (state->IsChanceNode()) {
    state->ApplyAction(SampleAction(state->ChanceOutcomes(), rng).first);
  }
  SPIEL_CHECK_FALSE(state->IsTerminal());

  // Nobody bids.
  while (!state->IsTerminal()) {
    std::vector<Action> legal_actions = state->LegalActions();
    SPIEL_CHECK_TRUE(absl::c_linear_search(legal_actions, kPass));
    state->ApplyAction(kPass);
  }

  // No dizhu was chosen, so nobody scores.
  for (double returns : state->Returns()) SPIEL_CHECK_EQ(returns, 0.0);

  // Formatting the terminal state must not index the deal with
  // kInvalidPlayer. The out-of-bounds write this guards against is only
  // visible under a sanitizer, so this is coverage for those builds; it is
  // shaped like the games_sim_test.py check that reported the crash.
  std::unique_ptr<State> clone = state->Clone();
  SPIEL_CHECK_EQ(state->ToString(), clone->ToString());
  SPIEL_CHECK_FALSE(state->ToString().empty());
}

}  // namespace
}  // namespace dou_dizhu
}  // namespace open_spiel

int main() {
  open_spiel::dou_dizhu::BasicGameTests();
  open_spiel::dou_dizhu::AuctionAllPassTest();
}
