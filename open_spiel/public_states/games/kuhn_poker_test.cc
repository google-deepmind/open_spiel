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
#include <utility>
#include <iostream>

#include "open_spiel/games/kuhn_poker.h"
#include "open_spiel/public_states/games/kuhn_poker.h"
#include "open_spiel/utils/down_cast.h"

namespace open_spiel {
namespace public_states {

// TODO(sustr): we need to implement many more tests.
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
  for (const auto& state: states) {
    histories.push_back(state->History());
  }
  return histories;
}

void TestGeneratePublicSets() {
  std::shared_ptr<const GameWithPublicStates> game =
      LoadGameWithPublicStates("kuhn_poker(players=2)");

  // down_cast for easier navigation in IDEs
  auto public_state = down_cast<KuhnPublicState>(
      std::move(game->NewInitialPublicState()));

  // Start of game
  {
    auto actual_set = public_state->GetPublicSet();
    auto actual_histories = StatesToHistories(actual_set);
    auto expected_histories = std::vector<std::vector<Action>>{{}};
    std::cout << ": " << actual_set << std::endl;
    SPIEL_CHECK_EQ(actual_histories, expected_histories
    );
  }
  // Deal 0
  public_state->ApplyPublicTransition("deal 0");
  {
    auto actual_set = public_state->GetPublicSet();
    auto actual_histories = StatesToHistories(actual_set);
    auto expected_histories = std::vector<std::vector<Action>>{
        {3}, {2}, {1}};
    std::cout << "Deal 0: " << actual_set << std::endl;
    SPIEL_CHECK_EQ(actual_histories, expected_histories);
  }
  // Deal 1
  public_state->ApplyPublicTransition("deal 1");
  {
    auto actual_set = public_state->GetPublicSet();
    auto actual_histories = StatesToHistories(actual_set);
    auto expected_histories = std::vector<std::vector<Action>>{
        {3, 2}, {3, 1}, {2, 3}, {2, 1}, {1, 2}, {1, 3}};
    std::cout << "Deal 1: " << actual_set << std::endl;
    SPIEL_CHECK_EQ(actual_histories, expected_histories);
  }
  // Player 0 pass
  public_state->ApplyPublicTransition("0");
  {
    auto actual_set = public_state->GetPublicSet();
    auto actual_histories = StatesToHistories(actual_set);
    auto expected_histories = std::vector<std::vector<Action>>{
        {3, 2, 0}, {3, 1, 0}, {2, 3, 0}, {2, 1, 0}, {1, 2, 0}, {1, 3, 0}};
    std::cout << "Player 0 pass: " << actual_set << std::endl;
    SPIEL_CHECK_EQ(actual_histories, expected_histories);
  }
  // Player 1 bet
  public_state->ApplyPublicTransition("1");
  {
    auto actual_set = public_state->GetPublicSet();
    auto actual_histories = StatesToHistories(actual_set);
    auto expected_histories = std::vector<std::vector<Action>>{
        {3, 2, 0, 1}, {3, 1, 0, 1}, {2, 3, 0, 1},
        {2, 1, 0, 1}, {1, 2, 0, 1}, {1, 3, 0, 1}};
    std::cout << "Player 1 bet: " << actual_set << std::endl;
    SPIEL_CHECK_EQ(actual_histories, expected_histories);
  }
  // Player 0 bet
  public_state->ApplyPublicTransition("1");
  {
    auto actual_set = public_state->GetPublicSet();
    auto actual_histories = StatesToHistories(actual_set);
    auto expected_histories = std::vector<std::vector<Action>>{
        {3, 2, 0, 1, 1}, {3, 1, 0, 1, 1}, {2, 3, 0, 1, 1},
        {2, 1, 0, 1, 1}, {1, 2, 0, 1, 1}, {1, 3, 0, 1, 1}};
    std::cout << "Player 0 bet: " << actual_set << std::endl;
    SPIEL_CHECK_EQ(actual_histories, expected_histories);
  }
}

}  // namespace
}  // namespace kuhn_poker
}  // namespace public_states
}  // namespace open_spiel


int main(int argc, char** argv) {
  open_spiel::public_states::kuhn_poker::TestGeneratePublicSets();
}
