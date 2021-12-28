// Copyright 2021 DeepMind Technologies Limited
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

#include "open_spiel/algorithms/deterministic_policy.h"

#include "open_spiel/games/kuhn_poker.h"
#include "open_spiel/games/leduc_poker.h"

namespace open_spiel {
namespace algorithms {
namespace {

void KuhnDeterministicPolicyTest() {
  std::shared_ptr<const Game> game = LoadGame("kuhn_poker");

  int p0_policies = 1;
  int p1_policies = 1;

  DeterministicTabularPolicy p0_policy(*game, Player{0});
  while (p0_policy.NextPolicy()) {
    p0_policies += 1;
  }
  SPIEL_CHECK_EQ(p0_policies, 64);  // 2^6

  DeterministicTabularPolicy p1_policy(*game, Player{1});
  while (p1_policy.NextPolicy()) {
    p1_policies += 1;
  }
  SPIEL_CHECK_EQ(p1_policies, 64);  // 2^6
}

void NumDeterministicPoliciesTest() {
  // In Kuhn, each player has 6 information states with 2 actions each.
  std::shared_ptr<const Game> game = LoadGame("kuhn_poker");
  SPIEL_CHECK_EQ(NumDeterministicPolicies(*game, 0), 64);
  SPIEL_CHECK_EQ(NumDeterministicPolicies(*game, 1), 64);

  // Leduc poker has larger than 2^64 - 1, so -1 will be returned.
  game = LoadGame("leduc_poker");
  SPIEL_CHECK_EQ(NumDeterministicPolicies(*game, 0), -1);
  SPIEL_CHECK_EQ(NumDeterministicPolicies(*game, 1), -1);
}

}  // namespace
}  // namespace algorithms
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::algorithms::KuhnDeterministicPolicyTest();
  open_spiel::algorithms::NumDeterministicPoliciesTest();
}
