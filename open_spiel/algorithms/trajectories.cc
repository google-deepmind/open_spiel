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

#include "open_spiel/algorithms/trajectories.h"

#include <algorithm>
#include <chrono>  // NOLINT
#include <cstdint>
#include <random>
#include <unordered_map>
#include <vector>

#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace algorithms {
namespace {
std::string StateKey(const Game& game, const State& state,
                     Player player = kInvalidPlayer) {
  if (game.GetType().provides_information_state_string) {
    if (player == kInvalidPlayer) return state.InformationStateString();
    return state.InformationStateString(player);
  } else if (game.GetType().provides_observation_string) {
    if (player == kInvalidPlayer) return state.ObservationString();
    return state.ObservationString(player);
  }
  return state.ToString();
}
}  // namespace

// Initializes a BatchedTrajectory of size [batch_size, T].
BatchedTrajectory::BatchedTrajectory(int batch_size) : batch_size(batch_size) {
  observations.resize(batch_size);
  state_indices.resize(batch_size);
  legal_actions.resize(batch_size);
  actions.resize(batch_size);
  player_policies.resize(batch_size);
  player_ids.resize(batch_size);
  rewards.resize(batch_size);
  next_is_terminal.resize(batch_size);
  valid.resize(batch_size);
}

// Moves a trajectory of size [1, T] into the current trajectory at index.
void BatchedTrajectory::MoveTrajectory(int index,
                                       BatchedTrajectory* trajectory) {
  // The passed trajectory must have a batch size of 1.
  SPIEL_CHECK_EQ(trajectory->batch_size, 1);
  max_trajectory_length =
      std::max(max_trajectory_length, trajectory->max_trajectory_length);
  observations[index] = std::move(trajectory->observations[0]);
  state_indices[index] = std::move(trajectory->state_indices[0]);
  legal_actions[index] = std::move(trajectory->legal_actions[0]);
  actions[index] = std::move(trajectory->actions[0]);
  player_policies[index] = std::move(trajectory->player_policies[0]);
  player_ids[index] = std::move(trajectory->player_ids[0]);
  rewards[index] = trajectory->rewards[0];
  next_is_terminal[index] = std::move(trajectory->next_is_terminal[0]);
  valid[index] = std::move(trajectory->valid[0]);
}

// Pads fields to make sure that they're all the same shape, i.e. [B, T, N],
// where N = the size of each field.
void BatchedTrajectory::ResizeFields(int length) {
  if (length > 0) {
    SPIEL_CHECK_GE(length, max_trajectory_length);
    // We adjust max_trajectory_length as it's no longer correct.
    max_trajectory_length = length;
  }
  // Only works for batches with at least one trajectory as otherwise we can't
  // infer the field size.
  SPIEL_CHECK_GT(batch_size, 0);
  // TODO(author1): Replace this with a multi-threaded version.
  for (int i = 0; i < batch_size; ++i) {
    // Each field has shape [B, T, field_size], where N is a parameter that is
    // fixed for each (game, field) pair. We thus have to get the size of N from
    // the existing vectors.
    if (!observations[0].empty()) {
      observations[i].resize(max_trajectory_length,
                             std::vector<float>(observations[0][0].size(), 0));
    }
    state_indices[i].resize(max_trajectory_length, 0);
    legal_actions[i].resize(max_trajectory_length,
                            std::vector<int>(legal_actions[0][0].size(), 1));

    // Actions has shape [B, T, 1]
    actions[i].resize(max_trajectory_length, 0);

    // legal_actions has shape [B, T, num_distinct_actions], while
    // player_policies[0][0].size() <= num_distinct_actions.
    player_policies[i].resize(
        max_trajectory_length,
        std::vector<double>(legal_actions[0][0].size(), 1));
    player_ids[i].resize(max_trajectory_length, 0);
    next_is_terminal[i].resize(max_trajectory_length, false);
    valid[i].resize(max_trajectory_length, false);
  }
}

BatchedTrajectory RecordBatchedTrajectory(
    const Game& game, const std::vector<TabularPolicy>& policies,
    const State& initial_state,
    const std::unordered_map<std::string, int>& state_to_index, int batch_size,
    bool include_full_observations, std::mt19937* rng_ptr,
    int max_unroll_length) {
  SPIEL_CHECK_GT(batch_size, 0);
  if (state_to_index.empty()) SPIEL_CHECK_TRUE(include_full_observations);
  BatchedTrajectory batched_trajectory(batch_size);
  // TODO(author1): Replace this with a multi-threaded version.
  for (int i = 0; i < batch_size; ++i) {
    BatchedTrajectory trajectory =
        RecordTrajectory(game, policies, initial_state, state_to_index,
                         include_full_observations, rng_ptr);
    SPIEL_CHECK_FALSE(trajectory.rewards[0].empty());
    batched_trajectory.MoveTrajectory(i, &trajectory);
  }
  batched_trajectory.ResizeFields(max_unroll_length);
  return batched_trajectory;
}

BatchedTrajectory RecordTrajectory(
    const Game& game, const std::vector<TabularPolicy>& policies,
    const State& initial_state,
    const std::unordered_map<std::string, int>& state_to_index,
    bool include_full_observations, std::mt19937* rng) {
  if (state_to_index.empty()) SPIEL_CHECK_TRUE(include_full_observations);
  BatchedTrajectory trajectory(/*batch_size=*/1);
  std::unique_ptr<open_spiel::State> state = initial_state.Clone();
  bool find_index = !state_to_index.empty();
  while (!state->IsTerminal()) {
    Action action = kInvalidAction;
    if (state->IsChanceNode()) {
      action = open_spiel::SampleAction(
                   state->ChanceOutcomes(),
                   std::uniform_real_distribution<double>(0.0, 1.0)(*rng))
                   .first;
    } else if (state->IsSimultaneousNode()) {
      open_spiel::SpielFatalError(
          "We do not support games with simultaneous actions.");
    } else {
      // Then we're at a decision node.
      trajectory.legal_actions[0].push_back(state->LegalActionsMask());
      if (find_index) {
        auto it = state_to_index.find(StateKey(game, *state));
        SPIEL_CHECK_TRUE(it != state_to_index.end());
        trajectory.state_indices[0].push_back(it->second);
      } else {
        trajectory.observations[0].push_back(state->InformationStateTensor());
      }
      ActionsAndProbs policy =
          policies.at(state->CurrentPlayer())
              .GetStatePolicy(state->InformationStateString());
      if (policy.size() > state->LegalActions().size()) {
        std::string policy_str = "";
        for (const auto& item : policy) {
          absl::StrAppend(&policy_str, "(", item.first, ",", item.second, ") ");
        }
        SpielFatalError(absl::StrCat(
            "There are more actions than legal actions from ",
            typeid(policies.at(state->CurrentPlayer())).name(),
            "\n Legal actions are: ", absl::StrJoin(state->LegalActions(), " "),
            " \n Available probabilities were:", policy_str));
      }
      std::vector<double> probs(game.NumDistinctActions(), 0.);
      for (const std::pair<Action, double>& pair : policy) {
        probs[pair.first] = pair.second;
      }
      trajectory.player_policies[0].push_back(probs);
      trajectory.player_ids[0].push_back(state->CurrentPlayer());
      action = SampleAction(policy, *rng).first;
      trajectory.actions[0].push_back(action);
    }
    SPIEL_CHECK_NE(action, kInvalidAction);
    state->ApplyAction(action);
  }
  trajectory.valid[0] = std::vector<int>(trajectory.actions[0].size(), true);
  trajectory.rewards[0] = state->Returns();
  trajectory.next_is_terminal[0].resize(trajectory.actions[0].size(), false);
  trajectory.next_is_terminal[0][trajectory.next_is_terminal[0].size() - 1] =
      true;

  // We arbitrarily set max_trajectory_length based on the actions field. All
  // the fields should have the same length.
  trajectory.max_trajectory_length = trajectory.actions[0].size();
  return trajectory;
}

BatchedTrajectory RecordBatchedTrajectory(
    const Game& game, const std::vector<TabularPolicy>& policies,
    const std::unordered_map<std::string, int>& state_to_index, int batch_size,
    bool include_full_observations, std::mt19937* rng_ptr,
    int max_unroll_length) {
  if (state_to_index.empty()) SPIEL_CHECK_TRUE(include_full_observations);
  std::unique_ptr<State> state = game.NewInitialState();
  return RecordBatchedTrajectory(game, policies, *state, state_to_index,
                                 batch_size, include_full_observations, rng_ptr,
                                 max_unroll_length);
}

BatchedTrajectory RecordBatchedTrajectory(
    const Game& game, const std::vector<TabularPolicy>& policies,
    const std::unordered_map<std::string, int>& state_to_index, int batch_size,
    bool include_full_observations, int seed, int max_unroll_length) {
  std::mt19937 rng(seed);
  return RecordBatchedTrajectory(game, policies, state_to_index, batch_size,
                                 include_full_observations, &rng,
                                 max_unroll_length);
}

BatchedTrajectory RecordTrajectory(
    const Game& game, const std::vector<TabularPolicy>& policies,
    const std::unordered_map<std::string, int>& state_to_index,
    bool include_full_observations, std::mt19937* rng_ptr) {
  if (state_to_index.empty()) SPIEL_CHECK_TRUE(include_full_observations);
  std::unique_ptr<State> state = game.NewInitialState();
  return RecordTrajectory(game, policies, *state, state_to_index,
                          include_full_observations, rng_ptr);
}

}  // namespace algorithms
}  // namespace open_spiel
