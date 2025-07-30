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

#include "open_spiel/evaluation/elo.h"

#include <map>
#include <string>
#include <vector>

#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace {

using evaluation::ComputeRatingsFromMatrices;
using evaluation::IntArray2D;

void TestSimpleTransitiveCase() {
  IntArray2D win_matrix = {{0, 2}, {1, 0}};
  IntArray2D draw_matrix = {{0, 0}, {0, 0}};
  std::vector<double> ratings1 =
      ComputeRatingsFromMatrices(win_matrix, draw_matrix);
  SPIEL_CHECK_GT(ratings1[0], ratings1[1]);
  SPIEL_CHECK_FLOAT_NEAR(ratings1[1], 0.0, 1e-6);

  // Testing default when excluding draws matrix.
  std::vector<double> ratings2 = ComputeRatingsFromMatrices(win_matrix);
  SPIEL_CHECK_FLOAT_NEAR(ratings1[0], ratings2[0], 1e-6);
  SPIEL_CHECK_FLOAT_NEAR(ratings1[1], ratings2[1], 1e-6);
}

void TestMeeplePentathlon() {
  // Meeple Pentathlon example from the VasE paper
  // (https://arxiv.org/abs/2312.03121)
  //    1: A > B > C
  //    1: A > C > B
  //    2: C > A > B
  //    1: B > C > A
  // Here, the first and last player have provably equal Elo ratings.
  IntArray2D win_matrix = {{0, 4, 2}, {1, 0, 2}, {3, 3, 0}};
  std::vector<double> ratings = ComputeRatingsFromMatrices(win_matrix);
  SPIEL_CHECK_FLOAT_NEAR(ratings[0], ratings[2], 1e-6);
  SPIEL_CHECK_LT(ratings[1], ratings[0]);
  SPIEL_CHECK_LT(ratings[1], ratings[2]);

  // Now, from match records.
  std::vector<evaluation::MatchRecord> match_records = {
      // A > B > C
      {"A", "B"}, {"A", "C"}, {"B", "C"},
      // A > C > B
      {"A", "C"}, {"A", "B"}, {"C", "B"},
      // 2: C > A > B
      {"C", "A"}, {"C", "B"}, {"A", "B"},
      {"C", "A"}, {"C", "B"}, {"A", "B"},
      // B > C > A
      {"B", "C"}, {"B", "A"}, {"C", "A"},
  };
  std::map<std::string, double> ratings_map =
      ComputeRatingsFromMatchRecords(match_records);
  SPIEL_CHECK_FLOAT_NEAR(ratings_map["A"], ratings[0], 1e-6);
  SPIEL_CHECK_FLOAT_NEAR(ratings_map["B"], ratings[1], 1e-6);
  SPIEL_CHECK_FLOAT_NEAR(ratings_map["C"], ratings[2], 1e-6);
}

void TestSCOPaperSec41Example() {
  // Example from Section 4.1 of the SCO paper
  // (https://arxiv.org/pdf/2411.00119)
  //    2: A > B > C
  //    3: C > A > B
  // Here, the first player's Elo is higher due to higher win rate.
  IntArray2D win_matrix = {{0, 5, 2}, {0, 0, 2}, {3, 3, 0}};
  std::vector<double> ratings = ComputeRatingsFromMatrices(win_matrix);
  SPIEL_CHECK_GT(ratings[0], ratings[2]);
  SPIEL_CHECK_LT(ratings[1], ratings[0]);
  SPIEL_CHECK_LT(ratings[1], ratings[2]);
}

}  // namespace
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::TestSimpleTransitiveCase();
  open_spiel::TestMeeplePentathlon();
  open_spiel::TestSCOPaperSec41Example();
}
