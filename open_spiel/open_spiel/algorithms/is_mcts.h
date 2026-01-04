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

#ifndef OPEN_SPIEL_ALGORITHMS_IS_MCTS_H_
#define OPEN_SPIEL_ALGORITHMS_IS_MCTS_H_

#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "open_spiel/abseil-cpp/absl/container/flat_hash_map.h"
#include "open_spiel/algorithms/mcts.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_bots.h"

// A basic implementation of Information Set Monte Carlo Tree Search (IS-MCTS)
// by Cowling et al. https://ieeexplore.ieee.org/abstract/document/6203567.

namespace open_spiel {
namespace algorithms {

// Use this constant to use an unlimited number of world samples.
inline constexpr int kUnlimitedNumWorldSamples = -1;

// The key identifying a node contains the InformationStateString or
// ObservationString, as well as the player id, because in some games the
// observation string can be the same for different players.
using ISMCTSStateKey = std::pair<Player, std::string>;

enum class ISMCTSFinalPolicyType {
  kNormalizedVisitCount,
  kMaxVisitCount,
  kMaxValue,
};

struct ChildInfo {
  int visits;
  double return_sum;
  double value() const { return return_sum / visits; }
};

struct ISMCTSNode {
  absl::flat_hash_map<Action, ChildInfo> child_info;
  int total_visits;
};

using InfostateResampler = std::function<std::unique_ptr<State>(
    const State& state, Player pl, std::function<double()> rng)>;

class ISMCTSBot : public Bot {
 public:
  // Construct an IS-MCTS bot. The parameter max_world_samples controls how many
  // states are sampled (with replacement!) at the root of the search; use
  // kUnlimitedWorldStates to have no restriction, and a number larger than
  // zero to restrict the number). If use_observation_string is true, then
  // will use ObservationString as a key instead of InformationStateString.
  // If allow_inconsistent_action_sets is true, then the algorithm handles the
  // case of differing legal action sets across states with the same state key
  // (information state string or observation string) which can happen when
  // using observations or with game that have imperfect recall.
  //
  // Important note: this bot requires that State::ResampleFromInfostate is
  // implemented.
  ISMCTSBot(int seed, std::shared_ptr<Evaluator> evaluator, double uct_c,
            int max_simulations, int max_world_samples,
            ISMCTSFinalPolicyType final_policy_type,
            bool use_observation_string, bool allow_inconsistent_action_sets);

  // An IS-MCTS with sensible defaults.
  ISMCTSBot(int seed, std::shared_ptr<Evaluator> evaluator, double uct_c,
            int max_simulations)
      : ISMCTSBot(seed, evaluator, uct_c, max_simulations,
                  kUnlimitedNumWorldSamples,
                  ISMCTSFinalPolicyType::kNormalizedVisitCount, false, false) {}

  Action Step(const State& state) override;

  bool ProvidesPolicy() override { return true; }
  ActionsAndProbs GetPolicy(const State& state) override;
  std::pair<ActionsAndProbs, Action> StepWithPolicy(
      const State& state) override;

  ActionsAndProbs RunSearch(const State& state);

  // Bot maintains no history, so these are empty.
  void Restart() override {}
  void RestartAt(const State& state) override {}
  // Set a custom resampling function.
  void SetResampler(InfostateResampler cb) { resampler_cb_ = cb; }

 private:
  void Reset();
  double RandomNumber();

  ISMCTSStateKey GetStateKey(const State& state) const;
  std::unique_ptr<State> SampleRootState(const State& state);
  // Dispatch to either domain-specific implementation,
  // or a specially supplied one via SetResampler()
  std::unique_ptr<State> ResampleFromInfostate(const State& state);
  ISMCTSNode* CreateNewNode(const State& state);
  ISMCTSNode* LookupNode(const State& state);
  ISMCTSNode* LookupOrCreateNode(const State& state);
  Action SelectActionTreePolicy(ISMCTSNode* node,
                                const std::vector<Action>& legal_actions);
  Action SelectActionUCB(ISMCTSNode* node);
  ActionsAndProbs GetFinalPolicy(const State& state, ISMCTSNode* node) const;
  void ExpandIfNecessary(ISMCTSNode* node, Action action) const;

  // Check if an expansion is possible (i.e. node does not contain all the
  // actions). If so, returns an action not yet in the children. Otherwise,
  // returns kInvalidAction.
  Action CheckExpand(ISMCTSNode* node,
                     const std::vector<Action>& legal_actions);

  // Returns a copy of the node with any actions not in specified legal actions
  // removed.
  ISMCTSNode FilterIllegals(ISMCTSNode* node,
                            const std::vector<Action>& legal_actions) const;

  // Run a simulation, returning the player returns.
  std::vector<double> RunSimulation(State* state);

  std::mt19937 rng_;
  std::shared_ptr<Evaluator> evaluator_;
  absl::flat_hash_map<ISMCTSStateKey, ISMCTSNode*> nodes_;
  std::vector<std::unique_ptr<ISMCTSNode>> node_pool_;

  // If the number of sampled world state is restricted, this list is used to
  // store the sampled states.
  std::vector<std::unique_ptr<State>> root_samples_;

  const double uct_c_;
  const int max_simulations_;
  const int max_world_samples_;
  const ISMCTSFinalPolicyType final_policy_type_;
  const bool use_observation_string_;
  const bool allow_inconsistent_action_sets_;
  ISMCTSNode* root_node_;
  InfostateResampler resampler_cb_;
};

}  // namespace algorithms
}  // namespace open_spiel

#endif  // OPEN_SPIEL_ALGORITHMS_IS_MCTS_H_
