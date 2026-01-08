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

#ifndef OPEN_SPIEL_ALGORITHMS_TABULAR_BEST_RESPONSE_MDP_H_
#define OPEN_SPIEL_ALGORITHMS_TABULAR_BEST_RESPONSE_MDP_H_

#include <memory>
#include <string>
#include <vector>

#include "open_spiel/abseil-cpp/absl/container/flat_hash_map.h"
#include "open_spiel/policy.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_globals.h"

// A tabular best response algorithm based on building an information set Markov
// decision process (IS-MDP), and then solving it using value iteration. In
// computing a best response, there is a maximizing player and a number of fixed
// players (who used fixed policies, normally represented as one fixed joint
// policy). An IS-MDP is an MDP whose nodes have a one-to-one correspondence
// with the maximizing player's information states. Transitions from ISMDP
// states to other ISMDP states are functions of the reach probabilities given
// the possible histories in the information states, chance node distributions,
// and policies of the other players.
//
// The keys used to uniquely identify states is in the MDP are
// State::InformationStateString for one-shot games and imperfect information
// games, and State::ObservationString for perfect information games. In the
// case of perfect information games (including simultaneous move games), this
// implementation requires that ObservationString is a sufficient Markovian
// description of the state; it does not need to be perfect recall, but it must
// not merge states that might have different expected values under the
// policies using these keys as state descriptions. As an example: in Goofspiel,
// it is insufficient for the observation to only include the current point card
// because which point cards remain in the point card deck is important for
// determining the expected value of the state (but the particular order they
// were played is not).
//
// This implementation has several advantages over best_response.* and
// tabular_exploitability.*:
//   - It supports perfect information games using State::ObservationString as
//     MDP node keys (rather than artificial blow-ups using info state strings)
//   - It supports abstract games that have imperfect recall due to actions
//   - It supports simultaneous move games without have to transform them
//     via TurnBasedSimultaneousGame
//   - The constraint on the game's State::ToString is lighter; it is only used
//     as a key for terminal states in the MDP.
//   - The memory requirement is linear in the information states (or
//     observations) plus the number of unique terminal states, rather than
//     in the total number of histories.
//
// There are some disadvantages:
//   - It is not nearly as thoroughly-tested
//   - History-level expected values are not stored nor retrievable
//
// Currently no performance comparisons have been done to compare the
// implementations.
//
// This implementation is exposed to Python. See
// test_cfr_plus_solver_best_response_mdp in cfr_test.py for an example use.
namespace open_spiel {
namespace algorithms {

class MDPNode {
 public:
  explicit MDPNode(const std::string& node_key);

  bool terminal() const { return terminal_; }
  double total_weight() const { return total_weight_; }
  double value() const { return value_; }
  std::string node_key() const { return node_key_; }
  void set_terminal(bool terminal) { terminal_ = terminal; }
  void add_weight(double weight) { total_weight_ += weight; }
  void set_value(double value) { value_ = value; }

  absl::flat_hash_map<Action, absl::flat_hash_map<MDPNode*, double>> &
  children() {
    return children_;
  }

  void IncTransitionWeight(Action a, MDPNode* child, double weight);

 private:
  bool terminal_;
  double total_weight_;
  // Children nodes {s'} from (s,a). The double value is the weight
  // (probability) assigned to the transition (s,a,s').
  absl::flat_hash_map<Action, absl::flat_hash_map<MDPNode*, double>> children_;
  double value_;
  std::string node_key_;
};

class MDP {
 public:
  MDP();
  MDPNode* CreateTerminalNode(const std::string& node_key);
  MDPNode* LookupOrCreateNode(const std::string& node_key,
                              bool terminal = false);
  MDPNode* RootNode() { return node_map_[kRootKey].get(); }

  double Solve(double tolerance, TabularPolicy* br_policy);
  int NumNonTerminalNodes() const { return num_nonterminal_nodes_; }
  int TotalSize() const { return node_map_.size(); }

 private:
  constexpr static const char* kRootKey = "**&!@ INFOSET_MDP ROOT KEY";
  constexpr static const char* kTerminalKeyPrefix = "**&!@ ISMDP TERMINAL KEY";
  absl::flat_hash_map<std::string, std::unique_ptr<MDPNode>> node_map_;
  int terminal_node_uid_;
  int num_nonterminal_nodes_;
  int num_terminal_nodes_;
};

// Information returned by the best response computation.
struct TabularBestResponseMDPInfo {
  std::vector<double> br_values;
  std::vector<TabularPolicy> br_policies;
  std::vector<double> on_policy_values;
  std::vector<double> deviation_incentives;
  double nash_conv;
  double exploitability;

  TabularBestResponseMDPInfo(int num_players)
      : br_values(num_players, 0), br_policies(num_players),
        on_policy_values(num_players, 0), deviation_incentives(num_players, 0),
        nash_conv(0), exploitability(0) {}
};


class TabularBestResponseMDP {
 public:
  TabularBestResponseMDP(const Game& game, const Policy& fixed_policy);

  // Compute best responses for all players.
  TabularBestResponseMDPInfo ComputeBestResponses();

  // Compute best responses for all players, and compute the specified metric
  // based on those choses. In the case of exploitability (only supported for
  // constant-sum games), the on-policy-values are not necessary and hence are
  // not returned.
  TabularBestResponseMDPInfo NashConv();
  TabularBestResponseMDPInfo Exploitability();

  // Build only one MDP and compute only the response for the specific player.
  TabularBestResponseMDPInfo ComputeBestResponse(Player max_player);

  int TotalNumNonterminals() const;
  int TotalSize() const;

 private:
  // This function builds all the players' Information Set MDPs in a single tree
  // traversal. There is a distribution of world states h for each s determined
  // by the opponents' policies. The transition probabilities are obtained by
  // summing the weights (h, a, h') that satisfy (s, a, s') and normalizing by
  // the weight obtained by the condition of having reached s.
  void BuildMDPs(const State &state, const std::vector<double> &reach_probs,
                 const std::vector<MDPNode*> &parent_nodes,
                 const std::vector<Action> &parent_actions,
                 Player only_for_player = kInvalidPlayer);

  std::string GetNodeKey(const State &state, Player player) const;

  double OpponentReach(const std::vector<double> &reach_probs, Player p) const;

  std::vector<std::unique_ptr<MDP>> mdps_;
  const Game &game_;
  const Policy &fixed_policy_;
  const int num_players_;
};

}  // namespace algorithms
}  // namespace open_spiel

#endif  // OPEN_SPIEL_ALGORITHMS_TABULAR_BEST_RESPONSE_MDP_H_
