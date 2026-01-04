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

#include "open_spiel/bots/pimc_bot.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <utility>
#include <vector>

#include "open_spiel/abseil-cpp/absl/container/flat_hash_map.h"
#include "open_spiel/abseil-cpp/absl/random/distributions.h"
#include "open_spiel/algorithms/maxn.h"
#include "open_spiel/algorithms/minimax.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_globals.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {

PIMCBot::PIMCBot(
    std::function<double(const State&, Player player)> value_function,
    Player player_id, uint32_t seed, int num_determinizations, int depth_limit)
    : rng_(seed),
      value_function_(value_function),
      player_id_(player_id),
      num_determinizations_(num_determinizations),
      depth_limit_(depth_limit) {}

Action PIMCBot::Step(const State& state) {
  std::pair<std::vector<int>, Action> search_result = Search(state);
  return search_result.second;
}

std::pair<ActionsAndProbs, Action> PIMCBot::StepWithPolicy(const State& state) {
  std::pair<std::vector<int>, Action> search_result = Search(state);
  return {PolicyFromBestAction(state, search_result.second),
          search_result.second};
}

ActionsAndProbs PIMCBot::GetPolicy(const State& state) {
  std::pair<std::vector<int>, Action> search_result = Search(state);
  return PolicyFromBestAction(state, search_result.second);
}

ActionsAndProbs PIMCBot::PolicyFromBestAction(const State& state,
                                              Action best_action) const {
  ActionsAndProbs actions_and_probs;
  for (Action action : state.LegalActions()) {
    if (action == best_action) {
      actions_and_probs.push_back({action, 1.0});
    } else {
      actions_and_probs.push_back({action, 0.0});
    }
  }
  return actions_and_probs;
}

std::pair<std::vector<int>, Action> PIMCBot::Search(const State& root_state) {
  int num_determinizations = num_determinizations_;

  GameType type = root_state.GetGame()->GetType();
  if (type.information == GameType::Information::kPerfectInformation) {
    num_determinizations = 1;
    // TODO(author5): drop down to expectimax or alpha-beta if 2-player
  }

  Player player = root_state.CurrentPlayer();
  std::vector<Action> legal_actions = root_state.LegalActions();
  const int num_legal_actions = legal_actions.size();
  std::vector<int> counts(num_legal_actions, 0);
  absl::flat_hash_map<Action, int> action_counts;
  for (Action action : legal_actions) {
    action_counts[action] = 0;
  }

  auto rng_func = [this]() {
    return absl::Uniform<double>(this->rng_, 0.0, 1.0);
  };

  for (int i = 0; i < num_determinizations; ++i) {
    std::unique_ptr<State> state = nullptr;

    if (num_determinizations == 1) {
      state = root_state.Clone();
    } else {
      state = root_state.ResampleFromInfostate(player, rng_func);
    }

    if (type.utility == GameType::Utility::kZeroSum &&
        type.chance_mode == GameType::ChanceMode::kDeterministic &&
        root_state.NumPlayers() == 2) {
      // Special case for two-player zero-sum deterministic games: use
      // alpha-beta.
      std::pair<double, Action> search_result = algorithms::AlphaBetaSearch(
          *state->GetGame(), state.get(),
          [this, player](const State& state) {
            return this->value_function_(state, player);
          },
          depth_limit_, player, /*use_undo*/ false);
      action_counts[search_result.second] += 1;
    } else {
      std::pair<std::vector<double>, Action> search_result =
          algorithms::MaxNSearch(*state->GetGame(), state.get(),
                                 value_function_, depth_limit_);
      action_counts[search_result.second] += 1;
    }
  }

  Action best_action = kInvalidAction;
  int highest_count = -1;
  for (int aidx = 0; aidx < num_legal_actions; ++aidx) {
    Action action = legal_actions[aidx];
    counts[aidx] = action_counts[action];
    if (counts[aidx] > highest_count) {
      highest_count = counts[aidx];
      best_action = action;
    }
  }

  return {counts, best_action};
}
}  // namespace open_spiel
