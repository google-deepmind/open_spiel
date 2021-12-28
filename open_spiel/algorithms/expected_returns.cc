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

#include "open_spiel/algorithms/expected_returns.h"

#include <functional>
#include <string>
#include <vector>

#include "open_spiel/simultaneous_move_game.h"
#include "open_spiel/spiel.h"

namespace open_spiel {
namespace algorithms {
namespace {

// Implements the recursive traversal using a general way to access the
// player's policies via a function that takes as arguments the player id and
// information state.
// We have a special case for the case where we can get a policy just from the
// InfostateString as that gives us a 2x speedup.
std::vector<double> ExpectedReturnsImpl(
    const State& state,
    const std::function<ActionsAndProbs(Player, const std::string&)>&
        policy_func,
    int depth_limit,
    float prob_cut_threshold) {
  if (state.IsTerminal() || depth_limit == 0) {
    return state.Rewards();
  }

  int num_players = state.NumPlayers();
  std::vector<double> values(num_players, 0.0);
  if (state.IsChanceNode()) {
    ActionsAndProbs action_and_probs = state.ChanceOutcomes();
    for (const auto& action_and_prob : action_and_probs) {
      if (action_and_prob.second <= prob_cut_threshold) continue;
      std::unique_ptr<State> child = state.Child(action_and_prob.first);
      std::vector<double> child_values =
          ExpectedReturnsImpl(
              *child, policy_func, depth_limit - 1, prob_cut_threshold);
      for (auto p = Player{0}; p < num_players; ++p) {
        values[p] += action_and_prob.second * child_values[p];
      }
    }
  } else if (state.IsSimultaneousNode()) {
    // Walk over all the joint actions, and weight by the product of
    // probabilities to choose them.
    values = state.Rewards();
    auto smstate = dynamic_cast<const SimMoveState*>(&state);
    SPIEL_CHECK_TRUE(smstate != nullptr);
    std::vector<ActionsAndProbs> state_policies(num_players);
    for (auto p = Player{0}; p < num_players; ++p) {
      state_policies[p] = policy_func(p, state.InformationStateString(p));
      if (state_policies[p].empty()) {
        SpielFatalError("Error in ExpectedReturnsImpl; infostate not found.");
      }
    }
    for (const Action flat_action : smstate->LegalActions()) {
      std::vector<Action> actions =
          smstate->FlatJointActionToActions(flat_action);
      double joint_action_prob = 1.0;
      for (auto p = Player{0}; p < num_players; ++p) {
        double player_action_prob = GetProb(state_policies[p], actions[p]);
        SPIEL_CHECK_GE(player_action_prob, 0.0);
        SPIEL_CHECK_LE(player_action_prob, 1.0);
        joint_action_prob *= player_action_prob;
        if (joint_action_prob <= prob_cut_threshold) {
          break;
        }
      }

      if (joint_action_prob > prob_cut_threshold) {
        std::unique_ptr<State> child = state.Clone();
        child->ApplyActions(actions);
        std::vector<double> child_values =
            ExpectedReturnsImpl(
                *child, policy_func, depth_limit - 1, prob_cut_threshold);
        for (auto p = Player{0}; p < num_players; ++p) {
          values[p] += joint_action_prob * child_values[p];
        }
      }
    }
  } else {
    // Turn-based decision node.
    Player player = state.CurrentPlayer();
    ActionsAndProbs state_policy =
        policy_func(player, state.InformationStateString());
    if (state_policy.empty()) {
      SpielFatalError("Error in ExpectedReturnsImpl; infostate not found.");
    }
    values = state.Rewards();
    for (const Action action : state.LegalActions()) {
      std::unique_ptr<State> child = state.Child(action);
      double action_prob = GetProb(state_policy, action);
      SPIEL_CHECK_GE(action_prob, 0.0);
      SPIEL_CHECK_LE(action_prob, 1.0);
      if (action_prob > prob_cut_threshold) {
        std::vector<double> child_values =
            ExpectedReturnsImpl(
                *child, policy_func, depth_limit - 1, prob_cut_threshold);
        for (auto p = Player{0}; p < num_players; ++p) {
          values[p] += action_prob * child_values[p];
        }
      }
    }
  }
  SPIEL_CHECK_EQ(values.size(), state.NumPlayers());
  return values;
}

// Same as above, but the policy_func now takes a State as input in, rather
// than a string.
std::vector<double> ExpectedReturnsImpl(
    const State& state,
    const std::function<ActionsAndProbs(Player, const State&)>& policy_func,
    int depth_limit,
    float prob_cut_threshold) {
  if (state.IsTerminal() || depth_limit == 0) {
    return state.Rewards();
  }

  int num_players = state.NumPlayers();
  std::vector<double> values(num_players, 0.0);
  if (state.IsChanceNode()) {
    ActionsAndProbs action_and_probs = state.ChanceOutcomes();
    for (const auto& action_and_prob : action_and_probs) {
      if (action_and_prob.second <= prob_cut_threshold) continue;
      std::unique_ptr<State> child = state.Child(action_and_prob.first);
      std::vector<double> child_values =
          ExpectedReturnsImpl(
              *child, policy_func, depth_limit - 1, prob_cut_threshold);
      for (auto p = Player{0}; p < num_players; ++p) {
        values[p] += action_and_prob.second * child_values[p];
      }
    }
  } else if (state.IsSimultaneousNode()) {
    // Walk over all the joint actions, and weight by the product of
    // probabilities to choose them.
    values = state.Rewards();
    auto smstate = dynamic_cast<const SimMoveState*>(&state);
    SPIEL_CHECK_TRUE(smstate != nullptr);
    std::vector<ActionsAndProbs> state_policies(num_players);
    for (auto p = Player{0}; p < num_players; ++p) {
      state_policies[p] = policy_func(p, state);
      if (state_policies[p].empty()) {
        SpielFatalError("Error in ExpectedReturnsImpl; infostate not found.");
      }
    }
    for (const Action flat_action : smstate->LegalActions()) {
      std::vector<Action> actions =
          smstate->FlatJointActionToActions(flat_action);
      double joint_action_prob = 1.0;
      for (auto p = Player{0}; p < num_players; ++p) {
        double player_action_prob = GetProb(state_policies[p], actions[p]);
        SPIEL_CHECK_GE(player_action_prob, 0.0);
        SPIEL_CHECK_LE(player_action_prob, 1.0);
        joint_action_prob *= player_action_prob;
        if (joint_action_prob <= prob_cut_threshold) {
          break;
        }
      }

      if (joint_action_prob > prob_cut_threshold) {
        std::unique_ptr<State> child = state.Clone();
        child->ApplyActions(actions);
        std::vector<double> child_values =
            ExpectedReturnsImpl(
                *child, policy_func, depth_limit - 1, prob_cut_threshold);
        for (auto p = Player{0}; p < num_players; ++p) {
          values[p] += joint_action_prob * child_values[p];
        }
      }
    }
  } else {
    // Turn-based decision node.
    Player player = state.CurrentPlayer();
    ActionsAndProbs state_policy = policy_func(player, state);
    if (state_policy.empty()) {
      SpielFatalError("Error in ExpectedReturnsImpl; infostate not found.");
    }
    values = state.Rewards();
    for (const Action action : state.LegalActions()) {
      std::unique_ptr<State> child = state.Child(action);
      double action_prob = GetProb(state_policy, action);
      SPIEL_CHECK_GE(action_prob, 0.0);
      SPIEL_CHECK_LE(action_prob, 1.0);
      if (action_prob > prob_cut_threshold) {
        std::vector<double> child_values =
            ExpectedReturnsImpl(
                *child, policy_func, depth_limit - 1, prob_cut_threshold);
        for (auto p = Player{0}; p < num_players; ++p) {
          values[p] += action_prob * child_values[p];
        }
      }
    }
  }
  SPIEL_CHECK_EQ(values.size(), state.NumPlayers());
  return values;
}
}  // namespace

std::vector<double> ExpectedReturns(const State& state,
                                    const std::vector<const Policy*>& policies,
                                    int depth_limit,
                                    bool use_infostate_get_policy,
                                    float prob_cut_threshold) {
  if (use_infostate_get_policy) {
    return ExpectedReturnsImpl(
        state,
        [&policies](Player player, const std::string& info_state) {
          return policies[player]->GetStatePolicy(info_state);
        },
        depth_limit,
        prob_cut_threshold);
  } else {
    return ExpectedReturnsImpl(
        state,
        [&policies](Player player, const State& state) {
          return policies[player]->GetStatePolicy(state, player);
        },
        depth_limit,
        prob_cut_threshold);
  }
}

std::vector<double> ExpectedReturns(const State& state,
                                    const Policy& joint_policy, int depth_limit,
                                    bool use_infostate_get_policy,
                                    float prob_cut_threshold) {
  if (use_infostate_get_policy) {
    return ExpectedReturnsImpl(
        state,
        [&joint_policy](Player player, const std::string& info_state) {
          return joint_policy.GetStatePolicy(info_state);
        },
        depth_limit,
        prob_cut_threshold);
  } else {
    return ExpectedReturnsImpl(
        state,
        [&joint_policy](Player player, const State& state) {
          return joint_policy.GetStatePolicy(state, player);
        },
        depth_limit,
        prob_cut_threshold);
  }
}

}  // namespace algorithms
}  // namespace open_spiel
