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

#include "open_spiel/games/bridge.h"

#include "open_spiel/abseil-cpp/absl/strings/str_replace.h"
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
  SPIEL_CHECK_EQ(Score({3, kNoTrump, kUndoubled}, 6, false), -150);
  SPIEL_CHECK_EQ(Score({3, kNoTrump, kDoubled}, 6, false), -500);
  SPIEL_CHECK_EQ(Score({2, kSpades, kDoubled}, 8, true), 670);
}

void BasicGameTests() {
  testing::LoadGameTest("bridge_uncontested_bidding(num_redeals=1)");
  testing::RandomSimTest(*LoadGame("bridge_uncontested_bidding(num_redeals=1)"),
                         3);
  testing::LoadGameTest("bridge");
  testing::RandomSimTest(*LoadGame("bridge"), 3);
  testing::RandomSimTest(*LoadGame("bridge(use_double_dummy_result=false)"), 3);
}

void DeserializeStateTest() {
  auto game = LoadGame("bridge_uncontested_bidding(num_redeals=1)");
  auto state = game->DeserializeState("AKQJ.543.QJ8.T92 97532.A2.9.QJ853");
  SPIEL_CHECK_EQ(state->ToString(), "AKQJ.543.QJ8.T92 97532.A2.9.QJ853 ");
}

void SerializeDoubleDummyResults() {
  auto game = LoadGame("bridge");
  auto state = game->NewInitialState();
  for (auto action : {33, 25, 3,  44, 47, 28, 23, 46, 1,  43, 30, 26, 29, 48,
                      24, 42, 13, 21, 17, 8,  5,  34, 6,  7,  37, 49, 11, 38,
                      51, 32, 20, 9,  0,  14, 35, 22, 10, 50, 15, 45, 39, 16,
                      12, 18, 27, 31, 41, 40, 4,  36, 19, 2,  52, 59, 52, 61}) {
    state->ApplyAction(action);
  }
  auto str = state->Serialize();
  str = absl::StrReplaceAll(str, {{"\n", ","}});
  SPIEL_CHECK_EQ(str,
                 "33,25,3,44,47,28,23,46,1,43,30,26,29,48,"
                 "24,42,13,21,17,8,5,34,6,7,37,49,11,38,51,"
                 "32,20,9,0,14,35,22,10,50,15,45,39,16,12,"
                 "18,27,31,41,40,4,36,19,2,52,59,52,61,"
                 "Double Dummy Results,"
                 "0,12,0,12,7,5,7,5,0,12,0,12,8,5,8,5,0,7,0,7,");
}

void DeserializeDoubleDummyResults() {
  auto game = LoadGame("bridge");
  // These results intentionally incorrect to check that the
  // implementation is using them rather than wastefully recomputing them.
  std::string serialized =
      "33,25,3,44,47,28,23,46,1,43,30,26,29,48,"
      "24,42,13,21,17,8,5,34,6,7,37,49,11,38,51,"
      "32,20,9,0,14,35,22,10,50,15,45,39,16,12,"
      "18,27,31,41,40,4,36,19,2,52,59,52,61,"
      "Double Dummy Results,"
      "12,12,0,12,7,5,7,5,9,12,0,12,6,5,8,5,3,7,0,7,";
  serialized = absl::StrReplaceAll(serialized, {{",", "\n"}});
  auto new_state = game->DeserializeState(serialized);
  SPIEL_CHECK_EQ(serialized, new_state->Serialize());
}

}  // namespace
}  // namespace bridge
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::bridge::DeserializeStateTest();
  open_spiel::bridge::ScoringTests();
  open_spiel::bridge::BasicGameTests();
  open_spiel::bridge::SerializeDoubleDummyResults();
  open_spiel::bridge::DeserializeDoubleDummyResults();
}
