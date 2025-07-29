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

// An implementation of the Elo rating system via Majorization-Minorization
// algorithm of Hunter, MM Algorithms for Generalized Bradley-Terry Models,
// The Annals of Statistics 2004, Vol. 32, No. 1, 384--406.

#include "open_spiel/evaluation/elo.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <vector>

#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace evaluation {
namespace {

void Normalize(std::vector<double>* vec) {
  double total =
      std::accumulate(vec->begin(), vec->end(), static_cast<double>(0.0));
  for (int i = 0; i < vec->size(); ++i) {
    (*vec)[i] = std::max<double>((*vec)[i] / total,
                                 std::numeric_limits<double>::denorm_min());
  }
}

// Returns the maximum absolute difference between the two vectors.
double MaxAbsDiff(const std::vector<double>& values_left,
                  const std::vector<double>& values_right) {
  double max_diff = 0;
  for (int idx = 0; idx < values_left.size(); ++idx) {
    max_diff =
        std::max(max_diff, std::abs(values_left[idx] - values_right[idx]));
  }
  return max_diff;
}

int GetNumDraws(const IntArray2D& draw_matrix, int row, int col) {
  if (draw_matrix.empty()) {
    return 0;
  } else {
    return draw_matrix[row][col];
  }
}
}  // namespace

std::vector<double> ComputeElo(const IntArray2D& win_matrix,
                               const IntArray2D& draw_matrix,
                               double smoothing_factor, int max_iterations,
                               double convergence_delta,
                               double scale_factor) {
  const int num_players = win_matrix.size();

  // First, compute W by converting to win scores (1.0 for each win,
  // 0.5 for draw) + smoothing factor.
  DoubleArray2D W_mat(num_players,
                      std::vector<double>(num_players, smoothing_factor));
  for (int i = 0; i < num_players; ++i) {
    SPIEL_CHECK_EQ(win_matrix[i].size(), num_players);
    if (!draw_matrix.empty()) {
      SPIEL_CHECK_EQ(draw_matrix[i].size(), num_players);
    }
    for (int j = 0; j < num_players; ++j) {
      W_mat[i][j] += (win_matrix[i][j] + GetNumDraws(draw_matrix, i, j) * 0.5);
      if (!draw_matrix.empty()) {
        SPIEL_CHECK_EQ(draw_matrix[i][j], draw_matrix[j][i]);
      }
    }
  }

  // Then compute total wins per player and N matrix (num matches). The number
  // of matches is the sum of the diagonal and the off-diagonal entries.
  std::vector<double> total_wins(num_players, 0.0);
  DoubleArray2D N_mat(num_players, std::vector<double>(num_players, 0.0));
  for (int i = 0; i < num_players; ++i) {
    for (int j = i + 1; j < num_players; ++j) {
      total_wins[i] += W_mat[i][j];
      total_wins[j] += W_mat[j][i];
      N_mat[i][j] = N_mat[j][i] = W_mat[i][j] + W_mat[j][i];
    }
  }

  std::vector<double> gammas_t(num_players, 1.0);
  std::vector<double> gammas_tp1(num_players, 0.0);

  for (int iter = 0; iter < max_iterations; ++iter) {
    for (int player_idx = 0; player_idx < num_players; ++player_idx) {
      double total_ratios = 0;
      double player_gamma = gammas_t[player_idx];
      for (int opp_idx = 0; opp_idx < num_players; ++opp_idx) {
        if (player_idx == opp_idx) {
          continue;
        }
        double total_gamma = player_gamma + gammas_t[opp_idx];
        total_ratios += N_mat[player_idx][opp_idx] / total_gamma;
      }
      gammas_tp1[player_idx] = total_wins[player_idx] / total_ratios;
    }
    Normalize(&gammas_tp1);
    if (MaxAbsDiff(gammas_t, gammas_tp1) < convergence_delta) {
      break;
    }
    gammas_t.swap(gammas_tp1);
  }

  std::vector<double> ratings(num_players, 0.0);
  for (int i = 0; i < num_players; ++i) {
    ratings[i] = scale_factor * std::log10(gammas_t[i]);
  }

  // Define the minimum Elo to be zero.
  double min_elo = *std::min_element(ratings.begin(), ratings.end());
  for (int i = 0; i < num_players; ++i) {
    ratings[i] -= min_elo;
  }

  return ratings;
}

}  // namespace evaluation
}  // namespace open_spiel
