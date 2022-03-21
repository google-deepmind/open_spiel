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

#include "open_spiel/algorithms/is_mcts.h"

#include <algorithm>
#include <numeric>

#include "open_spiel/abseil-cpp/absl/random/discrete_distribution.h"
#include "open_spiel/abseil-cpp/absl/random/distributions.h"
#include "open_spiel/abseil-cpp/absl/time/time.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace algorithms {

constexpr double kTieTolerance = 0.00001;
constexpr int kUnexpandedVisitCount = -1;

ISMCTSBot::ISMCTSBot(int seed, std::shared_ptr<Evaluator> evaluator,
                     double uct_c, int max_simulations, int max_world_samples,
                     ISMCTSFinalPolicyType final_policy_type,
                     bool use_observation_string,
                     bool allow_inconsistent_action_sets)
    : rng_(seed),
      evaluator_(evaluator),
      uct_c_(uct_c),
      max_simulations_(max_simulations),
      max_world_samples_(max_world_samples),
      final_policy_type_(final_policy_type),
      use_observation_string_(use_observation_string),
      allow_inconsistent_action_sets_(allow_inconsistent_action_sets) {}

double ISMCTSBot::RandomNumber() { return absl::Uniform(rng_, 0.0, 1.0); }

void ISMCTSBot::Reset() {
  nodes_.clear();
  node_pool_.clear();
  root_samples_.clear();
}

ISMCTSStateKey ISMCTSBot::GetStateKey(const State& state) const {
  if (use_observation_string_) {
    return {state.CurrentPlayer(), state.ObservationString()};
  } else {
    return {state.CurrentPlayer(), state.InformationStateString()};
  }
}

ActionsAndProbs ISMCTSBot::RunSearch(const State& state) {
  Reset();
  SPIEL_CHECK_EQ(state.GetGame()->GetType().dynamics,
                 GameType::Dynamics::kSequential);
  SPIEL_CHECK_EQ(state.GetGame()->GetType().information,
                 GameType::Information::kImperfectInformation);

  // Optimization in case of single legal action, and support for games which
  // do not support ResampleFromInfostate in certain specific (single action)
  // states.
  std::vector<Action> legal_actions = state.LegalActions();
  if (legal_actions.size() == 1) return {{legal_actions[0], 1.0}};

  root_node_ = CreateNewNode(state);
  SPIEL_CHECK_TRUE(root_node_ != nullptr);

  auto root_infostate_key = GetStateKey(state);

  for (int sim = 0; sim < max_simulations_; ++sim) {
    std::unique_ptr<State> sampled_root_state = SampleRootState(state);
    SPIEL_CHECK_TRUE(root_infostate_key == GetStateKey(*sampled_root_state));
    SPIEL_CHECK_TRUE(sampled_root_state != nullptr);
    RunSimulation(sampled_root_state.get());
  }

  if (allow_inconsistent_action_sets_) {
    // Filter illegals for this state.
    std::vector<Action> legal_actions = state.LegalActions();
    ISMCTSNode temp_node = FilterIllegals(root_node_, legal_actions);
    SPIEL_CHECK_GT(temp_node.total_visits, 0);
    return GetFinalPolicy(state, &temp_node);
  } else {
    return GetFinalPolicy(state, root_node_);
  }
}

Action ISMCTSBot::Step(const State& state) {
  ActionsAndProbs policy = RunSearch(state);
  return SampleAction(policy, RandomNumber()).first;
}

ActionsAndProbs ISMCTSBot::GetPolicy(const State& state) {
  return RunSearch(state);
}

std::pair<ActionsAndProbs, Action> ISMCTSBot::StepWithPolicy(
    const State& state) {
  ActionsAndProbs policy = GetPolicy(state);
  Action sampled_action = SampleAction(policy, RandomNumber()).first;
  return {policy, sampled_action};
}

ActionsAndProbs ISMCTSBot::GetFinalPolicy(const State& state,
                                          ISMCTSNode* node) const {
  ActionsAndProbs policy;
  SPIEL_CHECK_FALSE(node == nullptr);

  switch (final_policy_type_) {
    case ISMCTSFinalPolicyType::kNormalizedVisitCount: {
      SPIEL_CHECK_GT(node->total_visits, 0);
      policy.reserve(node->child_info.size());
      double total_visits = static_cast<double>(node->total_visits);
      for (const auto& action_and_child : node->child_info) {
        policy.push_back({action_and_child.first,
                          action_and_child.second.visits / total_visits});
      }
    } break;

    case ISMCTSFinalPolicyType::kMaxVisitCount: {
      SPIEL_CHECK_GT(node->total_visits, 0);
      policy.reserve(node->child_info.size());
      Action max_action = kInvalidAction;
      int max_visits = -std::numeric_limits<int>::infinity();
      for (const auto& action_and_child : node->child_info) {
        if (action_and_child.second.visits > max_visits) {
          max_visits = action_and_child.second.visits;
          max_action = action_and_child.first;
        }
      }
      SPIEL_CHECK_NE(max_action, kInvalidAction);
      for (const auto& action_and_child : node->child_info) {
        policy.push_back({action_and_child.first,
                          action_and_child.first == max_action ? 1.0 : 0.0});
      }
    } break;

    case ISMCTSFinalPolicyType::kMaxValue: {
      SPIEL_CHECK_GT(node->total_visits, 0);
      policy.reserve(node->child_info.size());
      Action max_action = kInvalidAction;
      double max_value = -std::numeric_limits<double>::infinity();
      for (const auto& action_and_child : node->child_info) {
        double value = action_and_child.second.value();
        if (value > max_value) {
          max_value = value;
          max_action = action_and_child.first;
        }
      }
      SPIEL_CHECK_NE(max_action, kInvalidAction);
      for (const auto& action_and_child : node->child_info) {
        policy.push_back({action_and_child.first,
                          action_and_child.first == max_action ? 1.0 : 0.0});
      }
    }
  }

  // In case the search didn't cover all the legal moves, at zero probability
  // for all the remaining actions.
  int policy_size = policy.size();
  std::vector<Action> legal_actions = state.LegalActions();
  if (policy_size < legal_actions.size()) {
    for (Action action : legal_actions) {
      if (node->child_info.find(action) == node->child_info.end()) {
        // Legal action not found in the node's actions: assign probability 0.
        policy.push_back({action, 0.0});
      }
    }
  }
  return policy;
}

std::unique_ptr<State> ISMCTSBot::SampleRootState(const State& state) {
  if (max_world_samples_ == kUnlimitedNumWorldSamples) {
    return ResampleFromInfostate(state);
  } else if (root_samples_.size() < max_world_samples_) {
    root_samples_.push_back(ResampleFromInfostate(state));
    return root_samples_.back()->Clone();
  } else if (root_samples_.size() == max_world_samples_) {
    int idx = absl::Uniform(rng_, 0u, root_samples_.size());
    return root_samples_[idx]->Clone();
  } else {
    SpielFatalError("Case not handled (badly set max_world_samples..?)");
  }
}

std::unique_ptr<State> ISMCTSBot::ResampleFromInfostate(const State& state) {
  if (resampler_cb_) {
    return resampler_cb_(state, state.CurrentPlayer(),
                         [this]() { return RandomNumber(); });
  } else {
    // Try domain-specific implementation
    // (could be not implemented in some games).
    return state.ResampleFromInfostate(state.CurrentPlayer(),
                                       [this]() { return RandomNumber(); });
  }
}

ISMCTSNode* ISMCTSBot::CreateNewNode(const State& state) {
  auto infostate_key = GetStateKey(state);
  node_pool_.push_back(std::unique_ptr<ISMCTSNode>(new ISMCTSNode));
  ISMCTSNode* node = node_pool_.back().get();
  nodes_[infostate_key] = node;
  node->total_visits = kUnexpandedVisitCount;
  return node;
}

ISMCTSNode* ISMCTSBot::LookupNode(const State& state) {
  auto iter = nodes_.find(GetStateKey(state));
  if (iter == nodes_.end()) {
    return nullptr;
  } else {
    return iter->second;
  }
}

ISMCTSNode* ISMCTSBot::LookupOrCreateNode(const State& state) {
  ISMCTSNode* node = LookupNode(state);
  if (node != nullptr) {
    return node;
  } else {
    return CreateNewNode(state);
  }
}

ISMCTSNode ISMCTSBot::FilterIllegals(
    ISMCTSNode* node, const std::vector<Action>& legal_actions) const {
  ISMCTSNode new_node = *node;
  std::vector<Action> to_delete;
  for (const auto& action_and_child : node->child_info) {
    if (std::find(legal_actions.begin(), legal_actions.end(),
                  action_and_child.first) == legal_actions.end()) {
      // Illegal action: mark for deletion.
      new_node.total_visits -= action_and_child.second.visits;
      new_node.child_info.erase(action_and_child.first);
    }
  }

  return new_node;
}

void ISMCTSBot::ExpandIfNecessary(ISMCTSNode* node, Action action) const {
  if (node->child_info.find(action) == node->child_info.end()) {
    node->child_info[action] = ChildInfo{0, 0.0};
  }
}

Action ISMCTSBot::SelectActionTreePolicy(
    ISMCTSNode* node, const std::vector<Action>& legal_actions) {
  // Check to see if we are allowing inconsistent action sets.
  if (allow_inconsistent_action_sets_) {
    // If so, it could mean that the node has actions with child info that are
    // not legal in this state, so we have to remove them.
    ISMCTSNode temp_node = FilterIllegals(node, legal_actions);
    if (temp_node.total_visits == 0) {
      // If we've filtered everything, return a random action.
      Action action =
          legal_actions[absl::Uniform(rng_, 0u, legal_actions.size())];
      ExpandIfNecessary(node, action);
      return action;
    } else {
      return SelectActionUCB(&temp_node);
    }
  } else {
    return SelectActionUCB(node);
  }
}

Action ISMCTSBot::SelectActionUCB(ISMCTSNode* node) {
  std::vector<Action> candidates;
  double max_value = -std::numeric_limits<double>::infinity();

  for (const auto& action_and_child : node->child_info) {
    // Every child should have at least one visit because the child is only
    // created when the action took it in a simulation, which then increases
    // its visit count immediately.
    SPIEL_CHECK_GT(action_and_child.second.visits, 0);

    Action action = action_and_child.first;
    double uct_val = action_and_child.second.value() +
                     uct_c_ * std::sqrt(std::log(node->total_visits) /
                                        action_and_child.second.visits);

    if (uct_val > max_value + kTieTolerance) {
      candidates.clear();
      candidates.push_back(action);
      max_value = uct_val;
    } else if (uct_val > max_value - kTieTolerance &&
               uct_val < max_value + kTieTolerance) {
      candidates.push_back(action);
      max_value = uct_val;
    }
  }

  SPIEL_CHECK_GE(candidates.size(), 1);

  if (candidates.size() == 1) {
    return candidates[0];
  } else {
    return candidates[absl::Uniform(rng_, 0u, candidates.size())];
  }
}

Action ISMCTSBot::CheckExpand(ISMCTSNode* node,
                              const std::vector<Action>& legal_actions) {
  // Fast check in the common/default case.
  if (!allow_inconsistent_action_sets_ &&
      node->child_info.size() == legal_actions.size()) {
    return kInvalidAction;
  }

  // Shuffle the legal actions to remove the bias from the move order.
  std::vector<Action> legal_actions_copy = legal_actions;
  std::shuffle(legal_actions_copy.begin(), legal_actions_copy.end(), rng_);
  for (Action action : legal_actions_copy) {
    if (node->child_info.find(action) == node->child_info.end()) {
      return action;
    }
  }
  return kInvalidAction;
}

std::vector<double> ISMCTSBot::RunSimulation(State* state) {
  if (state->IsTerminal()) {
    return state->Returns();
  } else if (state->IsChanceNode()) {
    Action chance_action =
        SampleAction(state->ChanceOutcomes(), RandomNumber()).first;
    state->ApplyAction(chance_action);
    return RunSimulation(state);
  }

  std::vector<Action> legal_actions = state->LegalActions();
  Player cur_player = state->CurrentPlayer();
  ISMCTSNode* node = LookupOrCreateNode(*state);
  SPIEL_CHECK_TRUE(node != nullptr);

  if (node->total_visits == kUnexpandedVisitCount) {
    // Newly created node, so we've just stepped out of the tree.
    node->total_visits = 0;  // Expand the node.
    return evaluator_->Evaluate(*state);
  } else {
    // Apply tree policy.
    Action chosen_action = CheckExpand(node, legal_actions);
    if (chosen_action != kInvalidAction) {
      // Expand.
      ExpandIfNecessary(node, chosen_action);
    } else {
      // No expansion, so use the tree policy to select.
      chosen_action = SelectActionTreePolicy(node, legal_actions);
    }

    SPIEL_CHECK_NE(chosen_action, kInvalidAction);

    // Need to updates the visits before the recursive call. In games with
    // imperfect recall, a node could be expanded with zero visit counts, and
    // you might encounter the same (node, action) pair in the same simulation
    // and the denominator for the UCT formula would be 0.
    node->total_visits++;
    node->child_info[chosen_action].visits++;

    state->ApplyAction(chosen_action);
    std::vector<double> returns = RunSimulation(state);
    node->child_info[chosen_action].return_sum += returns[cur_player];
    return returns;
  }
}

}  // namespace algorithms
}  // namespace open_spiel
