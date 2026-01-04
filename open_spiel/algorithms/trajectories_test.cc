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

#include <unordered_map>

#include "open_spiel/policy.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace algorithms {
namespace {

constexpr int kBatchSize = 32;

std::unordered_map<std::string, int> GetStatesToIndices(const Game& game) {
  std::unordered_map<std::string, int> state_index;
  std::vector<std::unique_ptr<State>> to_visit;
  to_visit.push_back(game.NewInitialState());
  int index = 0;
  while (!to_visit.empty()) {
    std::unique_ptr<State> state = std::move(to_visit.back());
    to_visit.pop_back();
    if (!state->IsChanceNode() && !state->IsTerminal()) {
      state_index[state->InformationStateString()] = index;
    }
    ++index;
    for (Action action : state->LegalActions()) {
      to_visit.push_back(state->Child(action));
    }
  }
  return state_index;
}

void RecordTrajectoryEveryFieldHasSameLength(const std::string& game_name) {
  std::shared_ptr<const Game> game = LoadGame(game_name);
  std::unordered_map<std::string, int> states_to_indices =
      GetStatesToIndices(*game);
  std::vector<TabularPolicy> policies(2, GetUniformPolicy(*game));
  std::mt19937 rng;
  BatchedTrajectory trajectory =
      RecordTrajectory(*game, policies, states_to_indices,
                       /*include_full_observations=*/false, &rng);
  int num_steps = trajectory.state_indices[0].size();
  SPIEL_CHECK_EQ(num_steps, trajectory.legal_actions[0].size());
  SPIEL_CHECK_EQ(num_steps, trajectory.actions[0].size());
  SPIEL_CHECK_EQ(num_steps, trajectory.player_policies[0].size());
  SPIEL_CHECK_EQ(num_steps, trajectory.player_ids[0].size());
  SPIEL_CHECK_EQ(num_steps, trajectory.next_is_terminal[0].size());
  SPIEL_CHECK_EQ(num_steps, trajectory.valid[0].size());
  SPIEL_CHECK_EQ(trajectory.rewards.size(), 1);
}

void RecordTrajectoryLegalActionsIsCorrect(const std::string& game_name) {
  std::shared_ptr<const Game> game = LoadGame(game_name);
  std::vector<TabularPolicy> policies(2, GetUniformPolicy(*game));
  std::unordered_map<std::string, int> states_to_indices =
      GetStatesToIndices(*game);
  std::mt19937 rng;

  BatchedTrajectory trajectory =
      RecordTrajectory(*game, policies, states_to_indices,
                       /*include_full_observations=*/false, &rng);
  std::unique_ptr<open_spiel::State> state = game->NewInitialState();
  for (int i = 0; i < trajectory.actions[0].size(); ++i) {
    while (state->IsChanceNode()) state->ApplyAction(state->LegalActions()[0]);
    if (!state->IsTerminal() && !state->IsChanceNode()) {
      SPIEL_CHECK_EQ(state->LegalActionsMask(), trajectory.legal_actions[0][i]);
    }
    state->ApplyAction(trajectory.actions[0][i]);
  }
}

void RecordTrajectoryNextIsTerminalIsCorrect(const std::string& game_name) {
  std::shared_ptr<const Game> game = LoadGame(game_name);
  std::vector<TabularPolicy> policies(2, GetUniformPolicy(*game));
  std::mt19937 rng;
  std::unordered_map<std::string, int> states_to_indices =
      GetStatesToIndices(*game);
  BatchedTrajectory trajectory =
      RecordTrajectory(*game, policies, states_to_indices,
                       /*include_full_observations=*/false, &rng);
  std::unique_ptr<open_spiel::State> state = game->NewInitialState();
  for (int i = 0; i < trajectory.actions[0].size(); ++i) {
    while (state->IsChanceNode()) state->ApplyAction(state->LegalActions()[0]);
    state->ApplyAction(trajectory.actions[0][i]);
    SPIEL_CHECK_EQ(state->IsTerminal(), trajectory.next_is_terminal[0][i]);
  }
}

void RecordTrajectoryPlayerIdsIsCorrect(const std::string& game_name) {
  std::shared_ptr<const Game> game = LoadGame(game_name);
  std::vector<TabularPolicy> policies(2, GetUniformPolicy(*game));
  std::mt19937 rng;
  std::unordered_map<std::string, int> states_to_indices =
      GetStatesToIndices(*game);

  BatchedTrajectory trajectory =
      RecordTrajectory(*game, policies, states_to_indices,
                       /*include_full_observations=*/false, &rng);
  std::unique_ptr<open_spiel::State> state = game->NewInitialState();
  for (int i = 0; i < trajectory.actions[0].size(); ++i) {
    while (state->IsChanceNode()) state->ApplyAction(state->LegalActions()[0]);
    if (!state->IsTerminal() && !state->IsChanceNode()) {
      SPIEL_CHECK_EQ(trajectory.player_ids[0][i], state->CurrentPlayer());
    }
    state->ApplyAction(trajectory.actions[0][i]);
  }
}

void RecordBatchedTrajectoryEveryFieldHasSameLength(
    const std::string& game_name) {
  std::shared_ptr<const Game> game = LoadGame(game_name);
  std::vector<TabularPolicy> policies(2, GetUniformPolicy(*game));
  std::unordered_map<std::string, int> states_to_indices =
      GetStatesToIndices(*game);
  std::mt19937 rng;
  BatchedTrajectory trajectory = RecordBatchedTrajectory(
      *game, policies, states_to_indices, kBatchSize,
      /*include_full_observations=*/false, /*rng_ptr=*/&rng);
  int batch_size = trajectory.batch_size;
  SPIEL_CHECK_EQ(batch_size, trajectory.legal_actions.size());
  SPIEL_CHECK_EQ(batch_size, trajectory.actions.size());
  SPIEL_CHECK_EQ(batch_size, trajectory.player_policies.size());
  SPIEL_CHECK_EQ(batch_size, trajectory.player_ids.size());
  SPIEL_CHECK_EQ(batch_size, trajectory.next_is_terminal.size());
}

void RecordBatchedTrajectoryLegalActionsIsCorrect(
    const std::string& game_name) {
  std::shared_ptr<const Game> game = LoadGame(game_name);
  std::vector<TabularPolicy> policies(2, GetUniformPolicy(*game));
  std::unordered_map<std::string, int> states_to_indices =
      GetStatesToIndices(*game);
  std::mt19937 rng;
  BatchedTrajectory trajectory = RecordBatchedTrajectory(
      *game, policies, states_to_indices, kBatchSize,
      /*include_full_observations=*/false, /*rng_ptr=*/&rng);
  for (int t = 0; t < trajectory.batch_size; ++t) {
    std::unique_ptr<open_spiel::State> state = game->NewInitialState();
    for (int i = 0; i < trajectory.actions[t].size(); ++i) {
      while (state->IsChanceNode()) {
        state->ApplyAction(state->LegalActions()[0]);
      }
      if (!state->IsTerminal() && !state->IsChanceNode()) {
        SPIEL_CHECK_EQ(state->LegalActionsMask(),
                       trajectory.legal_actions[t][i]);
      }
      state->ApplyAction(trajectory.actions[t][i]);
      if (state->IsTerminal()) break;
    }
  }
}

void RecordBatchedTrajectoryNextIsTerminalIsCorrect(
    const std::string& game_name) {
  std::shared_ptr<const Game> game = LoadGame(game_name);
  std::vector<TabularPolicy> policies(2, GetUniformPolicy(*game));
  std::unordered_map<std::string, int> states_to_indices =
      GetStatesToIndices(*game);
  std::mt19937 rng;
  BatchedTrajectory trajectory = RecordBatchedTrajectory(
      *game, policies, states_to_indices, kBatchSize,
      /*include_full_observations=*/false, /*rng_ptr=*/&rng);
  for (int t = 0; t < trajectory.batch_size; ++t) {
    std::unique_ptr<open_spiel::State> state = game->NewInitialState();
    for (int i = 0; i < trajectory.actions[t].size(); ++i) {
      while (state->IsChanceNode()) {
        state->ApplyAction(state->LegalActions()[0]);
      }
      state->ApplyAction(trajectory.actions[t][i]);
      SPIEL_CHECK_EQ(state->IsTerminal(), trajectory.next_is_terminal[t][i]);
      if (state->IsTerminal()) break;
    }
  }
}

void RecordBatchedTrajectoryPlayerIdsIsCorrect(const std::string& game_name) {
  std::shared_ptr<const Game> game = LoadGame(game_name);
  std::vector<TabularPolicy> policies(2, GetUniformPolicy(*game));
  std::unordered_map<std::string, int> states_to_indices =
      GetStatesToIndices(*game);
  std::mt19937 rng;
  BatchedTrajectory trajectory = RecordBatchedTrajectory(
      *game, policies, states_to_indices, kBatchSize,
      /*include_full_observations=*/false, /*rng_ptr=*/&rng);
  for (int t = 0; t < trajectory.batch_size; ++t) {
    std::unique_ptr<open_spiel::State> state = game->NewInitialState();
    for (int i = 0; i < trajectory.actions[t].size(); ++i) {
      while (state->IsChanceNode())
        state->ApplyAction(state->LegalActions()[0]);
      if (!state->IsTerminal() && !state->IsChanceNode()) {
        SPIEL_CHECK_EQ(trajectory.player_ids[t][i], state->CurrentPlayer());
      }
      state->ApplyAction(trajectory.actions[t][i]);
      if (state->IsTerminal()) break;
    }
  }
}

void BatchedTrajectoryResizesCorrectly(const std::string& game_name) {
  std::shared_ptr<const Game> game = LoadGame(game_name);
  const std::vector<TabularPolicy> policies(2, GetUniformPolicy(*game));
  std::unordered_map<std::string, int> states_to_indices =
      GetStatesToIndices(*game);
  std::mt19937 rng;
  BatchedTrajectory trajectory = RecordBatchedTrajectory(
      *game, policies, states_to_indices, kBatchSize,
      /*include_full_observations=*/false, /*rng_ptr=*/&rng);
  for (int b = 0; b < trajectory.batch_size; ++b) {
    SPIEL_CHECK_EQ(trajectory.valid[b].size(), trajectory.actions[b].size());
  }
  trajectory.ResizeFields(game->MaxGameLength());
  SPIEL_CHECK_EQ(trajectory.batch_size, kBatchSize);
  SPIEL_CHECK_EQ(trajectory.actions.size(), kBatchSize);
  SPIEL_CHECK_EQ(trajectory.player_ids.size(), kBatchSize);
  SPIEL_CHECK_EQ(trajectory.rewards.size(), kBatchSize);
  SPIEL_CHECK_EQ(trajectory.legal_actions.size(), kBatchSize);
  SPIEL_CHECK_EQ(trajectory.player_policies.size(), kBatchSize);
  SPIEL_CHECK_EQ(trajectory.next_is_terminal.size(), kBatchSize);
  SPIEL_CHECK_EQ(trajectory.valid.size(), kBatchSize);
  for (int b = 0; b < trajectory.batch_size; ++b) {
    SPIEL_CHECK_EQ(trajectory.actions[b].size(),
                   trajectory.max_trajectory_length);
    SPIEL_CHECK_EQ(trajectory.valid[b].size(),
                   trajectory.max_trajectory_length);
    SPIEL_CHECK_EQ(trajectory.player_ids[b].size(),
                   trajectory.max_trajectory_length);
    SPIEL_CHECK_EQ(trajectory.next_is_terminal[b].size(),
                   trajectory.max_trajectory_length);
    SPIEL_CHECK_EQ(trajectory.rewards[b].size(), game->NumPlayers());
    for (int t = 0; t < trajectory.max_trajectory_length; ++t) {
      SPIEL_CHECK_EQ(trajectory.legal_actions[b][t].size(),
                     game->NumDistinctActions());

      // We have to check for <= as some policies omit actions with zero
      // probability.
      SPIEL_CHECK_LE(trajectory.player_policies[b][t].size(),
                     game->NumDistinctActions());
    }
  }
}

}  // namespace
}  // namespace algorithms
}  // namespace open_spiel

namespace alg = open_spiel::algorithms;
int main() {
  // We test these games as they're all games that have implemented the
  // necessary methods. tic_tac_toe, for instance, has not.it
  for (const std::string& game_name :
       {"kuhn_poker", "leduc_poker", "liars_dice"}) {
    alg::RecordTrajectoryEveryFieldHasSameLength(game_name);
    alg::RecordTrajectoryLegalActionsIsCorrect(game_name);
    alg::RecordTrajectoryPlayerIdsIsCorrect(game_name);
    alg::RecordTrajectoryNextIsTerminalIsCorrect(game_name);
    alg::RecordTrajectoryEveryFieldHasSameLength(game_name);
    alg::RecordTrajectoryLegalActionsIsCorrect(game_name);
    alg::RecordTrajectoryPlayerIdsIsCorrect(game_name);
    alg::RecordTrajectoryNextIsTerminalIsCorrect(game_name);
    alg::RecordBatchedTrajectoryEveryFieldHasSameLength(game_name);
    alg::RecordBatchedTrajectoryLegalActionsIsCorrect(game_name);
    alg::RecordBatchedTrajectoryPlayerIdsIsCorrect(game_name);
    alg::RecordBatchedTrajectoryNextIsTerminalIsCorrect(game_name);
    alg::BatchedTrajectoryResizesCorrectly(game_name);
  }
}
