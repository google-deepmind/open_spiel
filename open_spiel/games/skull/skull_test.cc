// Copyright 2019 DeepMind Technologies Limited
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

#include "games/skull/skull.h"
#include "open_spiel/spiel.h"
#include "open_spiel/tests/basic_tests.h"
#include "policy.h"
#include "spiel_utils.h"
#include <memory>
#include <random>
#include <vector>

namespace open_spiel {
namespace skull {
namespace {

inline bool legal(const std::vector<Action> &LegalActions, Action a) {
  return std::binary_search(LegalActions.begin(), LegalActions.end(), a);
}
std::vector<Action> make_legal(const State &state, Player p,
                               const std::vector<Action> &potential) {
  const std::vector<Action> legal_actions = state.LegalActions(p);
  std::vector<Action> filteredActions;

  std::copy_if(potential.begin(), potential.end(),
               std::back_inserter(filteredActions),
               [&legal_actions](Action a) { return legal(legal_actions, a); });

  if (filteredActions.empty())
    return legal_actions; // falback to uniform
  return filteredActions;
}

ActionsAndProbs make_uniform(const std::vector<Action> &actions) {
  ActionsAndProbs actions_and_probs;
  absl::c_for_each(actions, [&actions_and_probs, &actions](Action a) {
    actions_and_probs.push_back({a, 1. / static_cast<double>(actions.size())});
  });
  return actions_and_probs;
}

std::vector<Action> safe_flip_targets(const SkullState &state, Player p) {
  const SkullGame &game =
      open_spiel::down_cast<const SkullGame &>(*state.GetGame());
  std::vector<Action> targets;
  for (Player target = 0; target < state.NumPlayers(); ++target) {
    if (state.is_active(target) && state.known_has_only_roses(target)) {
      targets.push_back(game.flip_base() + target);
    }
  }
  return targets;
}
class MetaPolicy : public Policy {
public:
  MetaPolicy(std::vector<std::unique_ptr<Policy>> policies, int num_players,
             std::mt19937 &rng)
      : policies_(std::move(policies)), seat_mapping_(num_players) {
    if (policies_.empty()) {
      open_spiel::SpielFatalError("Policy pool cannot be empty.");
    }
    AssignRandomPoliciesToSeats(rng);
  }
  void AssignRandomPoliciesToSeats(std::mt19937 &rng) {
    std::uniform_int_distribution<size_t> dist(0, policies_.size() - 1);

    for (int player = 0; player < seat_mapping_.size(); ++player) {
      size_t random_idx = dist(rng);
      seat_mapping_[player] = random_idx;
    }
  }
  ActionsAndProbs GetStatePolicy(const State &state, Player player) const {
    SPIEL_CHECK_GE(player, 0);
    auto &assigned_policy = policies_[seat_mapping_[player]];
    return assigned_policy->GetStatePolicy(state, player);
  }

private:
  std::vector<std::unique_ptr<Policy>> policies_;
  std::vector<int> seat_mapping_;
};

PreferredActionPolicy GetInstantFlipPolicy(const SkullGame &game) {
  std::vector<Action> preference_order;
  Action highest_possible_bid = kActionBidBase + game.MaxTotalCards();
  for (Action a = highest_possible_bid; a > kActionBidBase; a--) {
    preference_order.push_back(a);
  }
  preference_order.push_back(kActionPass);
  preference_order.push_back(kActionPlaceRose);
  preference_order.push_back(kActionPlaceSkull);
  preference_order.push_back(kActionDiscardSkull);
  preference_order.push_back(kActionDiscardRose);
  int n =  game.NumPlayers();
  for (Player p = 0; p <n; p++) {
    preference_order.push_back(game.flip_base() + p);
    // should maybe be random but PreferredActionPolicy is deterministic.
  }

  return PreferredActionPolicy(preference_order);
}

class UniformNonTabularPolicy : public Policy {
  ActionsAndProbs GetStatePolicy(const State &state, Player player) const {
    return UniformStatePolicy(state, player);
  }
};

class ImpatientPolicy : public Policy {
  ActionsAndProbs GetStatePolicy(const State &state, Player player) const {
    const SkullState &skull_state =
        open_spiel::down_cast<const SkullState &>(state);
    const std::vector<Action> legal_actions = state.LegalActions(player);
    switch (skull_state.current_phase()) {
    case GamePhase::kPlacement:
      if (legal(legal_actions, skull_state.highest_safe_bid_or_pass(player))) {
        return GetDeterministicPolicy(
            legal_actions, skull_state.highest_safe_bid_or_pass(player));
      } else {
        std::vector<Action> safe_actions =
            make_legal(state, player, {kActionPlaceRose, kActionPlaceSkull});
        return make_uniform(safe_actions);
      }
    case GamePhase::kBidding: {
      Action bid_one_higher = skull_state.current_bid() + kActionBidBase + 1;
      std::vector<Action> safe_actions =
          make_legal(state, player, {bid_one_higher, kActionPass});
      return make_uniform(safe_actions);
    }
    case GamePhase::kFlipping: {
      return make_uniform(
          make_legal(state, player, safe_flip_targets(skull_state, player)));
    }
    case GamePhase::kCardLoss: {
      return PreferredActionPolicy({kActionDiscardRose, kActionDiscardSkull})
          .GetStatePolicy(state, player);
    }
    }
  }
};

class SafePolicy : public Policy {
  ActionsAndProbs GetStatePolicy(const State &state, Player player) const {
    const SkullState &skull_state =
        open_spiel::down_cast<const SkullState &>(state);
    const std::vector<Action> legal_actions = state.LegalActions(player);
    switch (skull_state.current_phase()) {
    case GamePhase::kPlacement: {
      std::vector<Action> safe_actions = make_legal(
          state, player,
          {kActionPlaceRose, skull_state.highest_safe_bid_or_pass(player)});
      return make_uniform(safe_actions);
    }
    case GamePhase::kBidding: {
      std::vector<Action> safe_actions = make_legal(
          state, player, {skull_state.highest_safe_bid_or_pass(player)});
      return make_uniform(safe_actions);
    }
    case GamePhase::kFlipping: {
      return make_uniform(
          make_legal(state, player, safe_flip_targets(skull_state, player)));
    }
    case GamePhase::kCardLoss: {
      return PreferredActionPolicy({kActionDiscardSkull, kActionDiscardRose})
          .GetStatePolicy(state, player);
    }
    }
  }
};

void BasicSkullTests() {
  open_spiel::testing::LoadGameTest("skull");
  open_spiel::testing::ChanceOutcomesTest(*LoadGame("skull"));
  open_spiel::testing::RandomSimTest(*LoadGame("skull"), 50);
  open_spiel::testing::ResampleInfostateTest(*LoadGame("skull"), 10);
  std::mt19937 gen(1234);
  auto gptr = LoadGame("skull");
  const SkullGame& skull_game =
    open_spiel::down_cast<const SkullGame&>(*gptr);

  std::vector<std::unique_ptr<Policy>> policy_pool;
  policy_pool.push_back(std::make_unique<UniformNonTabularPolicy>());
  policy_pool.push_back(std::make_unique<PreferredActionPolicy>(
      GetInstantFlipPolicy(skull_game)));
  policy_pool.push_back(std::make_unique<ImpatientPolicy>());
  policy_pool.push_back(std::make_unique<SafePolicy>());

  MetaPolicy meta_policy(std::move(policy_pool), skull_game.NumPlayers(), gen);

  for (int i = 0; i < 10; i++) {
    open_spiel::testing::TestPoliciesCanPlay(meta_policy, *LoadGame("skull"),
                                             1);
    meta_policy.AssignRandomPoliciesToSeats(gen);
  }
}

} // namespace
} // namespace skull
} // namespace open_spiel

int main(int argc, char **argv) { open_spiel::skull::BasicSkullTests(); }
