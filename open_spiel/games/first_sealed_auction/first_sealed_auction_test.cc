// Copyright 2019 DeepMind Technologies Limited
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

#include "open_spiel/games/first_sealed_auction.h"

#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/tests/basic_tests.h"

namespace open_spiel {
namespace first_sealed_auction {
namespace {

namespace testing = open_spiel::testing;

void BasicFPSBATests(const GameParameters& params) {
  testing::LoadGameTest("first_sealed_auction");
  testing::ChanceOutcomesTest(*LoadGame("first_sealed_auction", params));
  testing::RandomSimTest(*LoadGame("first_sealed_auction", params), 100);
  testing::CheckChanceOutcomes(*LoadGame("first_sealed_auction", params));
}

void TieBreak() {
  std::shared_ptr<const Game> game = LoadGame(
      "first_sealed_auction", {{"players", open_spiel::GameParameter(3)},
                               {"max_value", open_spiel::GameParameter(5)}});
  std::vector<int64_t> action({1, 2, 3, 4, 5});
  auto state = game->NewInitialState();
  SPIEL_CHECK_EQ(state->LegalActions(), std::vector<int64_t>({1, 2, 3, 4, 5}));
  state->ApplyAction(5);
  SPIEL_CHECK_EQ(state->LegalActions(), std::vector<int64_t>({1, 2, 3, 4, 5}));
  state->ApplyAction(2);
  SPIEL_CHECK_EQ(state->LegalActions(), std::vector<int64_t>({1, 2, 3, 4, 5}));
  state->ApplyAction(4);
  SPIEL_CHECK_EQ(state->LegalActions(), std::vector<int64_t>({0, 1, 2, 3, 4}));
  state->ApplyAction(2);
  SPIEL_CHECK_EQ(state->LegalActions(), std::vector<int64_t>({0, 1}));
  state->ApplyAction(1);
  SPIEL_CHECK_EQ(state->LegalActions(), std::vector<int64_t>({0, 1, 2, 3}));
  state->ApplyAction(2);
  SPIEL_CHECK_EQ(state->LegalActions(), std::vector<int64_t>({0, 2}));
  state->ApplyAction(2);
  SPIEL_CHECK_TRUE(state->IsTerminal());
  SPIEL_CHECK_EQ(state->Returns(), std::vector<double>({0, 0, 2}));
}
}  // namespace
}  // namespace first_sealed_auction
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::first_sealed_auction::BasicFPSBATests({});
  open_spiel::first_sealed_auction::BasicFPSBATests(
      {{"players", open_spiel::GameParameter(1)},
       {"max_value", open_spiel::GameParameter(1)}});
  open_spiel::first_sealed_auction::BasicFPSBATests(
      {{"players", open_spiel::GameParameter(10)},
       {"max_value", open_spiel::GameParameter(2)}});
  open_spiel::first_sealed_auction::BasicFPSBATests(
      {{"players", open_spiel::GameParameter(2)},
       {"max_value", open_spiel::GameParameter(40)}});
  open_spiel::first_sealed_auction::TieBreak();
}
