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

#include "open_spiel/utils/stats.h"

#include "open_spiel/spiel_utils.h"
#include "open_spiel/utils/json.h"

namespace open_spiel {
namespace {

void TestBasicStats() {
  BasicStats stats;

  SPIEL_CHECK_EQ(stats.Num(), 0);
  SPIEL_CHECK_EQ(stats.Min(), 0);
  SPIEL_CHECK_EQ(stats.Max(), 0);
  SPIEL_CHECK_EQ(stats.Avg(), 0);
  SPIEL_CHECK_EQ(stats.StdDev(), 0);

  stats.Add(10);

  SPIEL_CHECK_EQ(stats.Num(), 1);
  SPIEL_CHECK_EQ(stats.Min(), 10);
  SPIEL_CHECK_EQ(stats.Max(), 10);
  SPIEL_CHECK_EQ(stats.Avg(), 10);
  SPIEL_CHECK_EQ(stats.StdDev(), 0);

  stats.Add(30);

  SPIEL_CHECK_EQ(stats.Num(), 2);
  SPIEL_CHECK_EQ(stats.Min(), 10);
  SPIEL_CHECK_EQ(stats.Max(), 30);
  SPIEL_CHECK_EQ(stats.Avg(), 20);
  SPIEL_CHECK_FLOAT_EQ(stats.StdDev(), 14.14213562);

  stats.Add(20);

  SPIEL_CHECK_EQ(stats.Num(), 3);
  SPIEL_CHECK_EQ(stats.Min(), 10);
  SPIEL_CHECK_EQ(stats.Max(), 30);
  SPIEL_CHECK_EQ(stats.Avg(), 20);
  SPIEL_CHECK_FLOAT_EQ(stats.StdDev(), 10);

  SPIEL_CHECK_EQ(stats.ToJson(), json::Object({
      {"num", 3},
      {"min", 10.0},
      {"max", 30.0},
      {"avg", 20.0},
      {"std_dev", 10.0},
  }));

  stats.Reset();

  SPIEL_CHECK_EQ(stats.Num(), 0);
  SPIEL_CHECK_EQ(stats.Min(), 0);
  SPIEL_CHECK_EQ(stats.Max(), 0);
  SPIEL_CHECK_EQ(stats.Avg(), 0);
  SPIEL_CHECK_EQ(stats.StdDev(), 0);
}

void TestHistogramNumbered() {
  HistogramNumbered hist(3);
  hist.Add(0);
  hist.Add(1);
  hist.Add(2);
  hist.Add(2);
  hist.Add(2);

  SPIEL_CHECK_EQ(hist.ToJson(), json::Array({1, 1, 3}));

  hist.Reset();

  SPIEL_CHECK_EQ(hist.ToJson(), json::Array({0, 0, 0}));
}

void TestHistogramNamed() {
  HistogramNamed hist({"win", "loss", "draw"});
  hist.Add(0);
  hist.Add(1);
  hist.Add(2);
  hist.Add(2);
  hist.Add(2);

  SPIEL_CHECK_EQ(hist.ToJson(), json::Object({
      {"counts", json::Array({1, 1, 3})},
      {"names", json::Array({"win", "loss", "draw"})},
  }));

  hist.Reset();

  SPIEL_CHECK_EQ(hist.ToJson(), json::Object({
      {"counts", json::Array({0, 0, 0})},
      {"names", json::Array({"win", "loss", "draw"})},
  }));
}

}  // namespace
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::TestBasicStats();
  open_spiel::TestHistogramNumbered();
  open_spiel::TestHistogramNamed();
}
