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
#include <map>
#include <numeric>
#include <string>
#include <vector>

#include "open_spiel/abseil-cpp/absl/container/flat_hash_map.h"
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

void BuildPlayerNameMap(
    const std::vector<MatchRecord>& match_records,
    absl::flat_hash_map<std::string, int>* name_to_index_map,
    std::vector<std::string>* player_names) {
  int cur_id = 0;
  for (const MatchRecord& match_record : match_records) {
    if (!name_to_index_map->contains(match_record.first_player_name)) {
      player_names->push_back(match_record.first_player_name);
      name_to_index_map->insert({match_record.first_player_name, cur_id});
      ++cur_id;
    }
    if (!name_to_index_map->contains(match_record.second_player_name)) {
      player_names->push_back(match_record.second_player_name);
      name_to_index_map->insert({match_record.second_player_name, cur_id});
      ++cur_id;
    }
  }
}

}  // namespace

EloOptions DefaultEloOptions() { return EloOptions(); }

std::vector<double> ComputeRatingsFromMatrices(const IntArray2D& win_matrix,
                                               const IntArray2D& draw_matrix,
                                               const EloOptions& options) {
  // Implements the algorithm described in Equation (4) of the Hunter paper.
  // https://projecteuclid.org/journals/annals-of-statistics/volume-32/issue-1/MM-algorithms-for-generalized-Bradley-Terry-models/10.1214/aos/1079120141.full
  const int num_players = win_matrix.size();

  // First, compute W by converting to win scores (1.0 for each win,
  // 0.5 for draw) + smoothing factor.
  DoubleArray2D W_mat(
      num_players, std::vector<double>(num_players, options.smoothing_factor));
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

  for (int iter = 0; iter < options.max_iterations; ++iter) {
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
    if (MaxAbsDiff(gammas_t, gammas_tp1) < options.convergence_delta) {
      break;
    }
    gammas_t.swap(gammas_tp1);
  }

  std::vector<double> ratings(num_players, 0.0);
  for (int i = 0; i < num_players; ++i) {
    ratings[i] = options.scale_factor * std::log10(gammas_t[i]);
  }

  // Subtract the minimum computed rating (which shifts everything to
  // be relative to 0), then shift upward by the specified minimum rating.
  double min_computed_rating = *std::min_element(ratings.begin(),
                                                 ratings.end());
  for (int i = 0; i < num_players; ++i) {
    ratings[i] = ratings[i] - min_computed_rating + options.minimum_rating;
  }

  return ratings;
}

std::map<std::string, double> ComputeRatingsFromMatchRecords(
    const std::vector<MatchRecord>& match_records, const EloOptions& options) {
  absl::flat_hash_map<std::string, int> name_to_index_map;
  std::vector<std::string> player_names;
  BuildPlayerNameMap(match_records, &name_to_index_map, &player_names);
  const int num_players = player_names.size();
  SPIEL_CHECK_GE(num_players, 2);
  IntArray2D win_matrix(num_players, std::vector<int>(num_players, 0));
  IntArray2D draw_matrix(num_players, std::vector<int>(num_players, 0));
  for (const MatchRecord& match_record : match_records) {
    int first_player_idx = name_to_index_map[match_record.first_player_name];
    int second_player_idx = name_to_index_map[match_record.second_player_name];
    if (match_record.outcome == kFirstPlayerWin) {
      ++win_matrix[first_player_idx][second_player_idx];
    } else if (match_record.outcome == kFirstPlayerLoss) {
      ++win_matrix[second_player_idx][first_player_idx];
    } else if (match_record.outcome == kDraw) {
      ++draw_matrix[first_player_idx][second_player_idx];
      ++draw_matrix[second_player_idx][first_player_idx];
    }
  }
  std::vector<double> ratings =
      ComputeRatingsFromMatrices(win_matrix, draw_matrix, options);
  std::map<std::string, double> ratings_map;
  for (int i = 0; i < num_players; ++i) {
    ratings_map[player_names[i]] = ratings[i];
  }
  return ratings_map;
}

}  // namespace evaluation
}  // namespace open_spiel
