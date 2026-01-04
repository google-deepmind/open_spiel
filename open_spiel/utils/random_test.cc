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


#include "open_spiel/utils/random.h"

#include <random>

#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace {

void TestRandomUtility() {
  {
    RandomFixedSequence r{0.};
    SPIEL_CHECK_EQ(r.RandomUniform(), 0.);
    SPIEL_CHECK_EQ(r.RandomUniform(), 0.);
    SPIEL_CHECK_EQ(r.RandomUniform(), 0.);
  }

  {
    RandomFixedSequence r{0., 1., 2.};
    SPIEL_CHECK_EQ(r.RandomUniform(), 0.);
    SPIEL_CHECK_EQ(r.RandomUniform(), 1.);
    SPIEL_CHECK_EQ(r.RandomUniform(), 2.);
    SPIEL_CHECK_EQ(r.RandomUniform(), 0.);
    SPIEL_CHECK_EQ(r.RandomUniform(), 1.);
    SPIEL_CHECK_EQ(r.RandomUniform(), 2.);
  }

  {
    std::mt19937 gen(0);
    std::uniform_real_distribution<double> uniformDist;

    RandomMT r(0);
    SPIEL_CHECK_EQ(r.RandomUniform(), uniformDist(gen));
    SPIEL_CHECK_EQ(r.RandomUniform(), uniformDist(gen));
    SPIEL_CHECK_EQ(r.RandomUniform(), uniformDist(gen));
  }
}

}  // namespace
}  // namespace open_spiel

int main(int argc, char **argv) { open_spiel::TestRandomUtility(); }
