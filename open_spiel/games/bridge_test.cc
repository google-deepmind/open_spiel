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

#include "open_spiel/games/bridge/bridge_scoring.h"
#include "open_spiel/games/bridge_uncontested_bidding.h"
#include "open_spiel/spiel.h"
#include "open_spiel/tests/basic_tests.h"

namespace open_spiel {
namespace bridge {
namespace {

void ScoringTests() {
  SPIEL_CHECK_EQ(Score({4, kHearts, kUndoubled}, 11, true), 650);
  SPIEL_CHECK_EQ(Score({4, kDiamonds, kUndoubled}, 10, true), 130);
  SPIEL_CHECK_EQ(Score({3, kNone, kUndoubled}, 6, false), -150);
  SPIEL_CHECK_EQ(Score({3, kNone, kDoubled}, 6, false), -500);
  SPIEL_CHECK_EQ(Score({2, kSpades, kDoubled}, 8, true), 670);
}

void BasicGameTests() {
  testing::LoadGameTest("bridge_uncontested_bidding");
  testing::NoChanceOutcomesTest(*LoadGame("bridge_uncontested_bidding"));
  testing::RandomSimTest(*LoadGame("bridge_uncontested_bidding"), 3);
}

void DeserializeStateTest() {
  auto game = LoadGame("bridge_uncontested_bidding");
  auto state = game->DeserializeState("AKQJ.543.QJ8.T92 97532.A2.9.QJ853");
  SPIEL_CHECK_EQ(state->ToString(), "AKQJ.543.QJ8.T92 97532.A2.9.QJ853 ");
}

}  // namespace
}  // namespace bridge
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::bridge::DeserializeStateTest();
  open_spiel::bridge::ScoringTests();
  open_spiel::bridge::BasicGameTests();
}
