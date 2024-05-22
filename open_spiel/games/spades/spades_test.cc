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

#include "open_spiel/games/spades/spades.h"

#include "open_spiel/abseil-cpp/absl/strings/str_replace.h"
#include "open_spiel/games/spades/spades_scoring.h"
#include "open_spiel/spiel.h"
#include "open_spiel/tests/basic_tests.h"

namespace open_spiel {
namespace spades {
namespace {

void ScoringTests() {
  // Score returns difference in score (reward), not new overall score
  SPIEL_CHECK_EQ(Score({4, 5, 5, 0}, {5, 3, 5, 0}, {0, 0})[0], 91);
  SPIEL_CHECK_EQ(Score({13, 5, 0, 1}, {4, 6, 1, 2}, {0, 0})[0], -230);
  SPIEL_CHECK_EQ(Score({3, 3, 3, 2}, {4, 2, 5, 2}, {99, 0})[0], -37);
  SPIEL_CHECK_EQ(Score({2, 3, 3, 3}, {2, 4, 2, 5}, {0, 99})[1], -37);
}

void BasicGameTests() {
  testing::LoadGameTest("spades");
  testing::RandomSimTest(*LoadGame("spades"), 3);
  testing::RandomSimTest(*LoadGame("spades(score_partnership_0=59,score_partnership_1=99)"), 3);
}


}  // namespace
}  // namespace spades
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::spades::ScoringTests();
  open_spiel::spades::BasicGameTests();
}
