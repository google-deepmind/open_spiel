// Copyright 2019 DeepMind Technologies Ltd. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "open_spiel/algorithms/is_mcts.h"

#include "open_spiel/abseil-cpp/absl/random/discrete_distribution.h"
#include "open_spiel/abseil-cpp/absl/random/distributions.h"
#include "open_spiel/abseil-cpp/absl/time/time.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace algorithms {

constexpr const double kTieTolerance = 0.00001;

ISMCTSBot::ISMCTSBot(int seed, std::shared_ptr<Evaluator> evaluator,
                     double uct_c, int max_simulations, int max_world_samples)
    : rng_(seed),
      evaluator_(evaluator),
      root_sample_index_(0),
      uct_c_(uct_c),
      max_simualtions_(max_simulations),
      max_world_samples_(max_world_samples) {}

double ISMCTSBot::RandomNumber() { return absl::Uniform(rng_, 0.0, 1.0); }

void ISMCTSBot::Reset() {
  nodes_.clear();
  node_pool_.clear();
  root_samples_.clear();
  root_sample_index_ = 0;
}

Action ISMCTSBot::Step(const State& state) {
  Reset();
  SPIEL_CHECK_EQ(state.GetGame()->GetType().dynamics,
                 GameType::Dynamics::kSequential);

  root_node_ = CreateNewNode(state);
  SPIEL_CHECK_TRUE(root_node_ != nullptr);

  std::string root_infostate_key = state.InformationStateString();

  for (int sim = 0; sim < max_simualtions_; ++sim) {
    std::unique_ptr<State> sampled_root_state = SampleRootState(state);
    SPIEL_CHECK_EQ(root_infostate_key,
                   sampled_root_state->InformationStateString());
    SPIEL_CHECK_TRUE(sampled_root_state != nullptr);
    RunSimulation(sampled_root_state.get());
  }

  absl::discrete_distribution<> visit_dist(root_node_->visits.begin(),
                                           root_node_->visits.end());
  int aidx = visit_dist(rng_);
  return root_node_->actions[aidx];
}

std::unique_ptr<State> ISMCTSBot::SampleRootState(const State& state) {
  if (max_world_samples_ < 0) {
    return state.ResampleFromInfostate(state.CurrentPlayer(),
                                       [this]() { return RandomNumber(); });
  } else if (root_samples_.size() < max_world_samples_) {
    root_samples_.push_back(state.ResampleFromInfostate(
        state.CurrentPlayer(), [this]() { return RandomNumber(); }));
    return root_samples_.back()->Clone();
  } else if (root_samples_.size() == max_world_samples_) {
    std::unique_ptr<State> sampled_state =
        root_samples_[root_sample_index_]->Clone();
    root_sample_index_ = (root_sample_index_ + 1) % root_samples_.size();
    return sampled_state;
  } else {
    SpielFatalError("Case not handled.");
  }
}

ISMCTSNode* ISMCTSBot::CreateNewNode(const State& state) {
  std::string infostate_key = state.InformationStateString();
  node_pool_.push_back(std::unique_ptr<ISMCTSNode>(new ISMCTSNode));
  ISMCTSNode* node = node_pool_.back().get();
  nodes_[infostate_key] = node;
  node->legal_actions = state.LegalActions();
  node->total_visits = -1;  // Means not expanded.
  return node;
}

ISMCTSNode* ISMCTSBot::LookupNode(const State& state) {
  auto iter = nodes_.find(state.InformationStateString());
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

std::pair<Action, int> ISMCTSBot::SelectAction(ISMCTSNode* node) {
  std::vector<std::pair<Action, int>> candidates;
  double max_value = -std::numeric_limits<double>::infinity();

  for (int aidx = 0; aidx < node->actions.size(); ++aidx) {
    Action action = node->actions[aidx];
    double mc_val = node->return_sums[aidx] / node->visits[aidx];
    double uct_val = mc_val + uct_c_ * std::sqrt(std::log(node->total_visits) /
                                                 node->visits[aidx]);

    if (uct_val > max_value + kTieTolerance) {
      candidates.clear();
      candidates.push_back({action, aidx});
      max_value = uct_val;
    } else if (uct_val > max_value - kTieTolerance &&
               uct_val < max_value + kTieTolerance) {
      candidates.push_back({action, aidx});
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

std::vector<double> ISMCTSBot::RunSimulation(State* state) {
  if (state->IsTerminal()) {
    return state->Returns();
  } else if (state->IsChanceNode()) {
    Action chance_action =
        SampleAction(state->ChanceOutcomes(), RandomNumber()).first;
    state->ApplyAction(chance_action);
    return RunSimulation(state);
  }

  Player cur_player = state->CurrentPlayer();
  ISMCTSNode* node = LookupOrCreateNode(*state);
  SPIEL_CHECK_TRUE(node != nullptr);

  if (node->total_visits == -1) {
    // Newly created node, so we've just stepped out of the tree.
    node->total_visits = 0;  // Expand the node.
    return evaluator_->Evaluate(*state);
  } else {
    // Apply tree policy.
    Action chosen_action = kInvalidAction;
    int chosen_aidx = -1;
    if (node->actions.size() < node->legal_actions.size()) {
      // Add the first action not in the tree.
      chosen_aidx = node->actions.size();
      chosen_action = node->legal_actions[chosen_aidx];
      node->actions.push_back(chosen_action);
      node->visits.push_back(0);
      node->return_sums.push_back(0);
    } else {
      // Use UCT to select.
      std::pair<Action, int> uct_selection = SelectAction(node);
      chosen_action = uct_selection.first;
      chosen_aidx = uct_selection.second;
    }

    state->ApplyAction(chosen_action);
    std::vector<double> returns = RunSimulation(state);
    node->total_visits++;
    node->visits[chosen_aidx]++;
    node->return_sums[chosen_aidx] += returns[cur_player];
    return returns;
  }
}

}  // namespace algorithms
}  // namespace open_spiel
