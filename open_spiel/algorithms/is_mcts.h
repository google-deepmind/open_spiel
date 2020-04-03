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

#ifndef OPEN_SPIEL_ALGORITHMS_IS_MCTS_H_
#define OPEN_SPIEL_ALGORITHMS_IS_MCTS_H_

#include <memory>
#include <random>
#include <string>

#include "open_spiel/abseil-cpp/absl/container/flat_hash_map.h"
#include "open_spiel/algorithms/mcts.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_bots.h"

// A basic implementation of Information Set Monte Carlo Tree Search (IS-MCTS)
// by Cowling et al. https://ieeexplore.ieee.org/abstract/document/6203567.

namespace open_spiel {
namespace algorithms {

struct ISMCTSNode {
  std::vector<Action> legal_actions;
  std::vector<Action> actions;
  std::vector<double> return_sums;
  std::vector<int> visits;
  int total_visits;
};

class ISMCTSBot : public Bot {
 public:
  // Construct an IS-MCTS bot. The parameter max_world_samples controls how many
  // states are sampled (with replacement!) at the root of the search; use -1 to
  // have no restriction, and a number larger than zero to restrict the number).
  //
  // Important note: this bot requires that State::ResampleFromInfostate is
  // implementede.
  ISMCTSBot(int seed, std::shared_ptr<Evaluator> evaluator, double uct_c,
            int max_simulations, int max_world_samples);

  Action Step(const State& state) override;

 private:
  void Reset();
  double RandomNumber();
  std::unique_ptr<State> SampleRootState(const State& state);
  ISMCTSNode* CreateNewNode(const State& state);
  ISMCTSNode* LookupNode(const State& state);
  ISMCTSNode* LookupOrCreateNode(const State& state);
  std::pair<Action, int> SelectAction(ISMCTSNode* node);
  std::vector<double> RunSimulation(State* state);

  std::mt19937 rng_;
  std::shared_ptr<Evaluator> evaluator_;
  absl::flat_hash_map<std::string, ISMCTSNode*> nodes_;
  std::vector<std::unique_ptr<ISMCTSNode>> node_pool_;

  // If the number of sampled world state is restricted, these are used to keep
  // track of which were sampled.
  std::vector<std::unique_ptr<State>> root_samples_;
  int root_sample_index_;

  double uct_c_;
  int max_simualtions_;
  int max_world_samples_;
  ISMCTSNode* root_node_;
};

}  // namespace algorithms
}  // namespace open_spiel

#endif  // OPEN_SPIEL_ALGORITHMS_IS_MCTS_H_
