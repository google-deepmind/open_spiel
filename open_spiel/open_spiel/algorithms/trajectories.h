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

#ifndef OPEN_SPIEL_ALGORITHMS_TRAJECTORIES_H_
#define OPEN_SPIEL_ALGORITHMS_TRAJECTORIES_H_

#include <stdint.h>

#include <memory>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include "open_spiel/policy.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace algorithms {

// The content of a trajectory. The idea is to represent a batch of trajectories
// of shape [B, T]. Each trajectory will be padded to have the same length,
// which is equal to the length of the longest episode in the batch.
struct BatchedTrajectory {
  // Initializes a BatchedTrajectory of size [batch_size, T].
  BatchedTrajectory(int batch_size);

  // Moves the trajectory fields into the current trajectory.
  void MoveTrajectory(int index, BatchedTrajectory* trajectory);

  // Pads fields to make sure that they're all the same shape, i.e. [B, T, N],
  // where N = the size of each field. If size is -1, i.e. the default, then
  // we resize to the max trajectory length in the batch.
  void ResizeFields(int length = -1);

  int batch_size;

  // Observations is an optional field that corresponds to the results of
  // calling State::InformationStateTensor. Only one of observations
  // and state_indices will be filled out for any given instance of
  // BatchedTrajectory.
  std::vector<std::vector<std::vector<float>>> observations;

  // The indices corresponding to the viewed state.
  std::vector<std::vector<int>> state_indices;

  // Stores the result of open_spiel::State::LegalActionMask.
  std::vector<std::vector<std::vector<int>>> legal_actions;
  std::vector<std::vector<Action>> actions;
  std::vector<std::vector<std::vector<double>>> player_policies;
  std::vector<std::vector<int>> player_ids;

  // This is a tensor of shape [B, T], where rewards[b][n] is the terminal
  // reward for episode b for player n.
  std::vector<std::vector<double>> rewards;

  // Tensor of shape [B, T]. valid[b][n] is true if actions[b][n] was actually
  // taken during a rollout, and false if it is just padding.
  std::vector<std::vector<int>> valid;

  // This is false everywhere except for the last state of the trajectory.
  std::vector<std::vector<int>> next_is_terminal;
  uint64_t max_trajectory_length = 0;
};

// If include_full_observations is true, then we record the result of
// open_spiel::State::InformationStateTensor(); otherwise, we store
// the index (taken from state_to_index).
BatchedTrajectory RecordTrajectory(
    const Game& game, const std::vector<TabularPolicy>& policies,
    const State& initial_state,
    const std::unordered_map<std::string, int>& state_to_index,
    bool include_full_observations, std::mt19937* rng_ptr);

BatchedTrajectory RecordBatchedTrajectory(
    const Game& game, const std::vector<TabularPolicy>& policies,
    const State& initial_state,
    const std::unordered_map<std::string, int>& state_to_index, int batch_size,
    bool include_full_observations, std::mt19937* rng_ptr,
    int max_unroll_length = -1);

BatchedTrajectory RecordTrajectory(
    const Game& game, const std::vector<TabularPolicy>& policies,
    const std::unordered_map<std::string, int>& state_to_index,
    bool include_full_observations, std::mt19937* rng_ptr);

BatchedTrajectory RecordBatchedTrajectory(
    const Game& game, const std::vector<TabularPolicy>& policies,
    const std::unordered_map<std::string, int>& state_to_index, int batch_size,
    bool include_full_observations, std::mt19937* rng_ptr,
    int max_unroll_length = -1);

BatchedTrajectory RecordBatchedTrajectory(
    const Game& game, const std::vector<TabularPolicy>& policies,
    const std::unordered_map<std::string, int>& state_to_index, int batch_size,
    bool include_full_observations, int seed, int max_unroll_length = -1);

// Stateful version of RecordTrajectory. There are several optimisations that
// this allows. Currently, the only optimisation is preventing making multiple
// copies of the state_to_index class. When state_to_index.empty() is false,
// then we default to setting the full observations field and not setting the
// state_indices field.
class TrajectoryRecorder {
 public:
  TrajectoryRecorder(const Game& game,
                     const std::unordered_map<std::string, int>& state_to_index,
                     int seed)
      : game_(game.shared_from_this()),
        state_to_index_(state_to_index),
        rng_(std::mt19937(seed)) {}

  BatchedTrajectory RecordBatch(const std::vector<TabularPolicy>& policies,
                                int batch_size, int max_unroll_length) {
    const bool include_full_observations = state_to_index_.empty();
    std::unique_ptr<State> root = game_->NewInitialState();
    return RecordBatchedTrajectory(*game_, policies, *root, state_to_index_,
                                   batch_size, include_full_observations, &rng_,
                                   max_unroll_length);
  }

 private:
  std::shared_ptr<const Game> game_;

  // Note: The key here depends on the game, and is implemented by the
  // StateKey method.
  std::unordered_map<std::string, int> state_to_index_;

  std::mt19937 rng_;
};

}  // namespace algorithms
}  // namespace open_spiel

#endif  // OPEN_SPIEL_ALGORITHMS_TRAJECTORIES_H_
