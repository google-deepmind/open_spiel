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

#include "open_spiel/algorithms/mcts.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <random>
#include <vector>

#include "open_spiel/abseil-cpp/absl/algorithm/container.h"
#include "open_spiel/abseil-cpp/absl/random/uniform_int_distribution.h"
#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/abseil-cpp/absl/strings/str_format.h"
#include "open_spiel/abseil-cpp/absl/time/clock.h"
#include "open_spiel/abseil-cpp/absl/time/time.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace algorithms {

// Return the memory use of a vector. Useful to track and limit memory use when
// running for a long time and build a big tree (eg to solve a game).
template <class T>
constexpr inline int VectorMemory(const std::vector<T>& vec) {
  return sizeof(T) * vec.capacity();
}

std::vector<double> RandomRolloutEvaluator::Evaluate(const State& state) {
  std::vector<double> result;
  for (int i = 0; i < n_rollouts_; ++i) {
    std::unique_ptr<State> working_state = state.Clone();
    while (!working_state->IsTerminal()) {
      if (working_state->IsChanceNode()) {
        ActionsAndProbs outcomes = working_state->ChanceOutcomes();
        Action action =
            SampleAction(outcomes,
                         std::uniform_real_distribution<double>(0.0, 1.0)(rng_))
                .first;
        working_state->ApplyAction(action);
      } else {
        std::vector<Action> actions = working_state->LegalActions();
        absl::uniform_int_distribution<int> dist(0, actions.size() - 1);
        int index = dist(rng_);
        working_state->ApplyAction(actions[index]);
      }
    }

    std::vector<double> returns = working_state->Returns();
    if (result.empty()) {
      result.swap(returns);
    } else {
      SPIEL_CHECK_EQ(returns.size(), result.size());
      for (int i = 0; i < result.size(); ++i) {
        result[i] += returns[i];
      }
    }
  }
  for (int i = 0; i < result.size(); ++i) {
    result[i] /= n_rollouts_;
  }
  return result;
}

ActionsAndProbs RandomRolloutEvaluator::Prior(const State& state) {
  // Returns equal probability for all actions.
  if (state.IsChanceNode()) {
    return state.ChanceOutcomes();
  } else {
    std::vector<Action> legal_actions = state.LegalActions();
    ActionsAndProbs prior;
    prior.reserve(legal_actions.size());
    for (const Action& action : legal_actions) {
      prior.emplace_back(action, 1.0 / legal_actions.size());
    }
    return prior;
  }
}

// UCT value of given child
double SearchNode::UCTValue(int parent_explore_count, double uct_c) const {
  if (!outcome.empty()) {
    return outcome[player];
  }

  if (explore_count == 0) return std::numeric_limits<double>::infinity();

  // The "greedy-value" of choosing a given child is always with respect to
  // the current player for this node.
  return total_reward / explore_count +
         uct_c * std::sqrt(std::log(parent_explore_count) / explore_count);
}

double SearchNode::PUCTValue(int parent_explore_count, double uct_c) const {
  // Returns the PUCT value of this node.
  if (!outcome.empty()) {
    return outcome[player];
  }

  return ((explore_count != 0 ? total_reward / explore_count : 0) +
          uct_c * prior * std::sqrt(parent_explore_count) /
              (explore_count + 1));
}

bool SearchNode::CompareFinal(const SearchNode& b) const {
  double out = (outcome.empty() ? 0 : outcome[player]);
  double out_b = (b.outcome.empty() ? 0 : b.outcome[b.player]);
  if (out != out_b) {
    return out < out_b;
  }
  if (explore_count != b.explore_count) {
    return explore_count < b.explore_count;
  }
  return total_reward < b.total_reward;
}

const SearchNode& SearchNode::BestChild() const {
  // Returns the best action from this node, either proven or most visited.
  //
  // This ordering leads to choosing:
  // - Highest proven score > 0 over anything else, including a promising but
  //   unproven action.
  // - A proven draw only if it has higher exploration than others that are
  //   uncertain, or the others are losses.
  // - Uncertain action with most exploration over loss of any difficulty
  // - Hardest loss if everything is a loss
  // - Highest expected reward if explore counts are equal (unlikely).
  // - Longest win, if multiple are proven (unlikely due to early stopping).
  return *std::max_element(children.begin(), children.end(),
                           [](const SearchNode& a, const SearchNode& b) {
                             return a.CompareFinal(b);
                           });
}

std::string SearchNode::ChildrenStr(const State& state) const {
  std::string out;
  if (!children.empty()) {
    std::vector<const SearchNode*> refs;  // Sort a list of refs, not a copy.
    refs.reserve(children.size());
    for (const SearchNode& child : children) {
      refs.push_back(&child);
    }
    std::sort(refs.begin(), refs.end(),
              [](const SearchNode* a, const SearchNode* b) {
                return b->CompareFinal(*a);
              });
    for (const SearchNode* child : refs) {
      absl::StrAppend(&out, child->ToString(state), "\n");
    }
  }
  return out;
}

std::string SearchNode::ToString(const State& state) const {
  return absl::StrFormat(
      "%6s: player: %d, prior: %5.3f, value: %6.3f, sims: %5d, outcome: %s, "
      "%3d children",
      (action != kInvalidAction ? state.ActionToString(player, action)
                                : "none"),
      player, prior, (explore_count ? total_reward / explore_count : 0.),
      explore_count,
      (outcome.empty()
           ? "none"
           : absl::StrFormat("%4.1f",
                             outcome[player == kChancePlayerId ? 0 : player])),
      children.size());
}

std::vector<double> dirichlet_noise(int count, double alpha,
                                    std::mt19937* rng) {
  auto noise = std::vector<double>{};
  noise.reserve(count);

  std::gamma_distribution<double> gamma(alpha, 1.0);
  for (int i = 0; i < count; ++i) {
    noise.emplace_back(gamma(*rng));
  }

  double sum = absl::c_accumulate(noise, 0.0);
  for (double& v : noise) {
    v /= sum;
  }
  return noise;
}

MCTSBot::MCTSBot(const Game& game, Player player, Evaluator* evaluator,
                 double uct_c, int max_simulations, int64_t max_memory_mb,
                 bool solve, int seed, bool verbose,
                 ChildSelectionPolicy child_selection_policy,
                 double dirichlet_alpha, double dirichlet_epsilon)
    : player_id_(player),
      uct_c_{uct_c},
      max_simulations_{max_simulations},
      max_memory_(max_memory_mb << 20),  // megabytes -> bytes
      verbose_(verbose),
      solve_(solve),
      max_utility_(game.MaxUtility()),
      dirichlet_alpha_(dirichlet_alpha),
      dirichlet_epsilon_(dirichlet_epsilon),
      rng_(seed),
      child_selection_policy_(child_selection_policy),
      evaluator_{evaluator} {
  GameType game_type = game.GetType();
  if (game_type.reward_model != GameType::RewardModel::kTerminal)
    SpielFatalError("Game must have terminal rewards.");
  if (game_type.dynamics != GameType::Dynamics::kSequential)
    SpielFatalError("Game must have sequential turns.");
  if (player < 0 || player >= game.NumPlayers())
    SpielFatalError(absl::StrFormat(
        "Game doesn't support that many players. Max: %d, player: %d",
        game.NumPlayers(), player));
}

Action MCTSBot::Step(const State& state) {
  absl::Time start = absl::Now();
  std::unique_ptr<SearchNode> root = MCTSearch(state);
  const SearchNode& best = root->BestChild();

  if (verbose_) {
    double seconds = absl::ToDoubleSeconds(absl::Now() - start);
    std::cerr
        << absl::StrFormat(
               "Finished %d sims in %.3f secs, %.1f sims/s, tree size: %d mb.",
               root->explore_count, seconds, (root->explore_count / seconds),
               memory_used_ / (1 << 20))
        << std::endl;
    std::cerr << "Root:" << std::endl;
    std::cerr << root->ToString(state) << std::endl;
    std::cerr << "Children:" << std::endl;
    std::cerr << root->ChildrenStr(state) << std::endl;
    if (!best.children.empty()) {
      std::unique_ptr<State> chosen_state = state.Clone();
      chosen_state->ApplyAction(best.action);
      std::cerr << "Children of chosen:" << std::endl;
      std::cerr << best.ChildrenStr(*chosen_state) << std::endl;
    }
  }

  return best.action;
}

std::pair<ActionsAndProbs, Action> MCTSBot::StepWithPolicy(const State& state) {
  Action action = Step(state);
  return {{{action, 1.}}, action};
}

std::unique_ptr<State> MCTSBot::ApplyTreePolicy(
    SearchNode* root, const State& state,
    std::vector<SearchNode*>* visit_path) {
  visit_path->push_back(root);
  std::unique_ptr<State> working_state = state.Clone();
  SearchNode* current_node = root;
  while (!working_state->IsTerminal() && current_node->explore_count > 0) {
    if (current_node->children.empty()) {
      // For a new node, initialize its state, then choose a child as normal.
      ActionsAndProbs legal_actions = evaluator_->Prior(*working_state);
      if (current_node == root && dirichlet_alpha_ > 0) {
        std::vector<double> noise =
            dirichlet_noise(legal_actions.size(), dirichlet_alpha_, &rng_);
        for (int i = 0; i < legal_actions.size(); i++) {
          legal_actions[i].second =
              (1 - dirichlet_epsilon_) * legal_actions[i].second +
              dirichlet_epsilon_ * noise[i];
        }
      }
      // Reduce bias from move generation order.
      std::shuffle(legal_actions.begin(), legal_actions.end(), rng_);
      Player player = working_state->CurrentPlayer();
      current_node->children.reserve(legal_actions.size());
      for (auto [action, prior] : legal_actions) {
        current_node->children.emplace_back(action, player, prior);
      }
      memory_used_ += VectorMemory(legal_actions);
    }

    SearchNode* chosen_child = nullptr;
    if (working_state->IsChanceNode()) {
      // For chance nodes, rollout according to chance node's probability
      // distribution
      Action chosen_action =
          SampleAction(working_state->ChanceOutcomes(),
                       std::uniform_real_distribution<double>(0.0, 1.0)(rng_))
              .first;

      for (SearchNode& child : current_node->children) {
        if (child.action == chosen_action) {
          chosen_child = &child;
          break;
        }
      }
    } else {
      // Otherwise choose node with largest UCT value.
      double max_value = -std::numeric_limits<double>::infinity();
      for (SearchNode& child : current_node->children) {
        double val;
        switch (child_selection_policy_) {
          case ChildSelectionPolicy::UCT:
            val = child.UCTValue(current_node->explore_count, uct_c_);
            break;
          case ChildSelectionPolicy::PUCT:
            val = child.PUCTValue(current_node->explore_count, uct_c_);
            break;
        }
        if (val > max_value) {
          max_value = val;
          chosen_child = &child;
        }
      }
    }

    working_state->ApplyAction(chosen_child->action);
    current_node = chosen_child;
    visit_path->push_back(current_node);
  }

  return working_state;
}

std::unique_ptr<SearchNode> MCTSBot::MCTSearch(const State& state) {
  SPIEL_CHECK_EQ(player_id_, state.CurrentPlayer());
  memory_used_ = 0;
  auto root = std::make_unique<SearchNode>(kInvalidAction, player_id_, 1);
  std::vector<SearchNode*> visit_path;
  std::vector<double> returns;
  visit_path.reserve(64);
  for (int i = 0; i < max_simulations_; ++i) {
    visit_path.clear();
    returns.clear();

    std::unique_ptr<State> working_state =
        ApplyTreePolicy(root.get(), state, &visit_path);

    bool solved;
    if (working_state->IsTerminal()) {
      returns = working_state->Returns();
      visit_path[visit_path.size() - 1]->outcome = returns;
      memory_used_ += VectorMemory(returns);
      solved = solve_;
    } else {
      returns = evaluator_->Evaluate(*working_state);
      solved = false;
    }

    // Propagate values back.
    for (auto it = visit_path.rbegin(); it != visit_path.rend(); ++it) {
      SearchNode* node = *it;

      node->total_reward +=
          returns[node->player == kChancePlayerId ? player_id_ : node->player];
      node->explore_count += 1;

      // Back up solved results as well.
      if (solved && !node->children.empty()) {
        Player player = node->children[0].player;
        if (player == kChancePlayerId) {
          // Only back up chance nodes if all have the same outcome.
          // An alternative would be to back up the weighted average of
          // outcomes if all children are solved, but that is less clear.
          const std::vector<double>& outcome = node->children[0].outcome;
          if (!outcome.empty() &&
              std::all_of(node->children.begin() + 1, node->children.end(),
                          [&outcome](const SearchNode& c) {
                            return c.outcome == outcome;
                          })) {
            node->outcome = outcome;
            memory_used_ += VectorMemory(node->outcome);
          } else {
            solved = false;
          }
        } else {
          // If any have max utility (won?), or all children are solved,
          // choose the one best for the player choosing.
          const SearchNode* best = nullptr;
          bool all_solved = true;
          for (const SearchNode& child : node->children) {
            if (child.outcome.empty()) {
              all_solved = false;
            } else if (best == nullptr ||
                       child.outcome[player] > best->outcome[player]) {
              best = &child;
            }
          }
          if (best != nullptr &&
              (all_solved || best->outcome[player] == max_utility_)) {
            node->outcome = best->outcome;
            memory_used_ += VectorMemory(node->outcome);
          } else {
            solved = false;
          }
        }
      }
    }

    if (!root->outcome.empty() ||  // Full game tree is solved.
        (max_memory_ && memory_used_ >= max_memory_) ||
        root->children.size() == 1) {
      break;
    }
  }

  return root;
}

}  // namespace algorithms
}  // namespace open_spiel
