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

#ifndef OPEN_SPIEL_ALGORITHMS_FSICFR_H_
#define OPEN_SPIEL_ALGORITHMS_FSICFR_H_

#include <random>
#include <vector>

#include "open_spiel/abseil-cpp/absl/container/flat_hash_map.h"
#include "open_spiel/policy.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_globals.h"

namespace open_spiel {
namespace algorithms {

// An basic implementation of Neller and Hnath 2011, "Approximating Optimal Dudo
// Play with Fixed-Strategy Iteration Counterfactual Regret Minimization"
// https://cupola.gettysburg.edu/csfac/2/.
//
// This implementation currently assumes the following:
//   - All chance events occur at the start of the game (before any decisions)
//   - There exists a perfect ranking between a player's chance event outcomes
//     and their outcome, encoded as a chance_id integer (one per player)
//
// This implementation was built for and only tested on Liar's dice. For a usage
// example, see examples/fsicfr_liars_dice.cc.

struct FSICFRNode {
  // Maximum number of predecessor nodes (used for top sort order).
  int max_predecessors = 0;

  int id = -1;

  // Chance id corresponding to the player to play at this node.
  int chance_id = -1;

  bool terminal = false;
  double p0_utility = 0;

  std::string string_key = "";
  Player player = kInvalidPlayer;
  int T = 0;
  int visits = 0;
  double v = 0;

  // This is an (Action, other player chance id) -> node id map.
  absl::flat_hash_map<std::pair<Action, int>, int> children;

  std::vector<int> parent_ids;
  std::vector<Action> legal_actions;

  std::vector<double> ssum;
  std::vector<double> psum;
  std::vector<double> strategy;
  std::vector<double> regrets;

  FSICFRNode();
  void AddChild(Action action, int chance_id, FSICFRNode* child);
  std::string ToString();
  void ApplyRegretMatching();
};

class FSICFRGraph {
 public:
  FSICFRGraph() {}
  FSICFRNode* GetOrCreateDecisionNode(const std::vector<Action>& legal_actions,
                                      const std::string& info_state_string,
                                      Player player, int max_predecessors,
                                      int chance_id);
  FSICFRNode* GetOrCreateTerminalNode(const std::string& terminal_string_key,
                                      double p0_utility, int max_predecessors);
  FSICFRNode* GetNode(int id) const {
    if (id < 0 || id >= nodes_.size()) {
      return nullptr;
    } else {
      FSICFRGraph* this_graph = const_cast<FSICFRGraph*>(this);
      return &this_graph->nodes_[id];
    }
  }

  int size() const { return nodes_.size(); }

  // Topologically sort the graph (in order of non-decreasing max_predecessors).
  void TopSort();

  int ordered_node_id(int idx) const { return ordered_ids_[idx]; }

 private:
  // Infostate/terminal string key to node id map
  absl::flat_hash_map<std::string, int> string_key_to_node_id_map_;

  // Nodes. Ids correspond to indices.
  std::vector<FSICFRNode> nodes_;

  // Topologically sorted nodes ids. A more space-efficient implementation could
  // remove this vector and simply build the node list in a such a way that
  // nodes_ is already topologically-ordered.
  std::vector<int> ordered_ids_;
};

class FSICFRSolver {
 public:
  FSICFRSolver(const Game& game, int seed,
               const std::vector<int>& chance_outcome_ranges,
               const FSICFRGraph* graph);
  void RunIteration();
  void RunIterations(int n);

  TabularPolicy GetAveragePolicy() const;

 private:
  void ForwardPass();
  void BackwardPass();

  const Game& game_;
  std::mt19937 rng_;

  int total_iterations_;

  // The maximum value of unique chance outcomes for each player.
  std::vector<int> chance_outcome_ranges_;

  // These are the predetermined chance outcomes for the iteration.
  std::vector<int> sampled_chance_outcomes_;

  // The FSICFR graph.
  const FSICFRGraph* graph_;
};

}  // namespace algorithms
}  // namespace open_spiel

#endif  // OPEN_SPIEL_ALGORITHMS_FSICFR_H_
