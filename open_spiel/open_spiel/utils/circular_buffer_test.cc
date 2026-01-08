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

#include "open_spiel/utils/circular_buffer.h"

#include <random>

#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace {

void TestCircularBuffer() {
  CircularBuffer<int> buffer(4);
  std::mt19937 rng;
  std::vector<int> sample;

  SPIEL_CHECK_TRUE(buffer.Empty());
  SPIEL_CHECK_EQ(buffer.Size(), 0);

  buffer.Add(13);
  SPIEL_CHECK_FALSE(buffer.Empty());
  SPIEL_CHECK_EQ(buffer.Size(), 1);
  SPIEL_CHECK_EQ(buffer.TotalAdded(), 1);
  SPIEL_CHECK_EQ(buffer[0], 13);

  sample = buffer.Sample(&rng, 1);
  SPIEL_CHECK_EQ(sample.size(), 1);
  SPIEL_CHECK_EQ(sample[0], 13);

  buffer.Add(14);
  buffer.Add(15);
  buffer.Add(16);

  SPIEL_CHECK_EQ(buffer.Size(), 4);
  SPIEL_CHECK_EQ(buffer.TotalAdded(), 4);

  sample = buffer.Sample(&rng, 2);
  SPIEL_CHECK_EQ(sample.size(), 2);
  SPIEL_CHECK_GE(sample[0], 13);
  SPIEL_CHECK_LE(sample[0], 16);
  SPIEL_CHECK_GE(sample[1], 13);
  SPIEL_CHECK_LE(sample[1], 16);

  buffer.Add(17);
  buffer.Add(18);

  SPIEL_CHECK_EQ(buffer.Size(), 4);
  SPIEL_CHECK_EQ(buffer.TotalAdded(), 6);

  sample = buffer.Sample(&rng, 1);
  SPIEL_CHECK_EQ(sample.size(), 1);
  SPIEL_CHECK_GE(sample[0], 15);
  SPIEL_CHECK_LE(sample[0], 18);
}

}  // namespace
}  // namespace open_spiel

int main(int argc, char** argv) { open_spiel::TestCircularBuffer(); }
