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

#ifndef OPEN_SPIEL_EVALUATION_ELO_H_
#define OPEN_SPIEL_EVALUATION_ELO_H_

#include <map>
#include <string>
#include <vector>

namespace open_spiel {
namespace evaluation {

using IntArray2D = std::vector<std::vector<int>>;
using DoubleArray2D = std::vector<std::vector<double>>;

constexpr int kDefaultMaxIterations = 2000;
constexpr double kDefaultSmoothingFactor = 0.01;
constexpr double kDefaultConvergenceDelta = 1e-10;
constexpr double kStandardScaleFactor = 400.0;
constexpr double kDefaultMinimumRating = 0.0;

enum MatchOutcome {
  kFirstPlayerWin = 0,
  kFirstPlayerLoss = 1,  // second player won
  kDraw = 2,
};

// A record of a match between two players. Default outcome is kFirstPlayerWin,
// so adding records of type A beats B is equivalent to simply constructing as
// MatchRecord("A", "B").
struct MatchRecord {
  std::string first_player_name;
  std::string second_player_name;
  MatchOutcome outcome;
  MatchRecord(std::string _first_player_name, std::string _second_player_name,
              MatchOutcome _outcome = kFirstPlayerWin)
      : first_player_name(_first_player_name),
        second_player_name(_second_player_name),
        outcome(_outcome) {}
};

struct EloOptions {
  double smoothing_factor = kDefaultSmoothingFactor;
  int max_iterations = kDefaultMaxIterations;
  double convergence_delta = kDefaultConvergenceDelta;
  double scale_factor = kStandardScaleFactor;
  double minimum_rating = kDefaultMinimumRating;
};

EloOptions DefaultEloOptions();

std::vector<double> ComputeRatingsFromMatrices(
    const IntArray2D& win_matrix, const IntArray2D& draw_matrix = {},
    const EloOptions& options = DefaultEloOptions());

std::map<std::string, double> ComputeRatingsFromMatchRecords(
    const std::vector<MatchRecord>& match_records,
    const EloOptions& options = DefaultEloOptions());

}  // namespace evaluation
}  // namespace open_spiel

#endif  // OPEN_SPIEL_EVALUATION_H_
