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

#include "open_spiel/algorithms/state_distribution.h"

#include <algorithm>
#include <functional>
#include <map>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include "open_spiel/abseil-cpp/absl/algorithm/container.h"
#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/abseil-cpp/absl/strings/str_format.h"
#include "open_spiel/simultaneous_move_game.h"
#include "open_spiel/spiel.h"

namespace open_spiel::algorithms {
namespace {

std::vector<double> Normalize(const std::vector<double>& weights) {
  std::vector<double> probs(weights);
  const double normalizer = absl::c_accumulate(weights, 0.);
  absl::c_for_each(probs, [&probs, normalizer](double& w) {
    w = (normalizer == 0.0 ? 1.0 / probs.size() : w / normalizer);
  });
  return probs;
}

void AdvanceBeliefHistoryOneAction(HistoryDistribution* previous, Action action,
                                   Player player_id,
                                   const Policy* opponent_policy) {
  for (int i = 0; i < previous->first.size(); ++i) {
    std::unique_ptr<State>& parent = previous->first[i];
    double& prob = previous->second[i];
    if (Near(prob, 0.)) continue;
    switch (parent->GetType()) {
      case StateType::kChance: {
        open_spiel::ActionsAndProbs outcomes = parent->ChanceOutcomes();
        double action_prob = GetProb(outcomes, action);

        // If we don't find the chance outcome, then the state we're in is
        // impossible, so we set it to zero.
        if (action_prob == -1) {
          prob = 0;
          continue;
        }
        SPIEL_CHECK_PROB(action_prob);
        prob *= action_prob;
        break;
      }
      case StateType::kDecision: {
        if (parent->CurrentPlayer() == player_id) break;
        open_spiel::ActionsAndProbs policy =
            opponent_policy->GetStatePolicy(*parent);
        double action_prob = GetProb(policy, action);
        SPIEL_CHECK_PROB(action_prob);
        prob *= action_prob;
        break;
      }
      case StateType::kTerminal:
        ABSL_FALLTHROUGH_INTENDED;
      default:
        SpielFatalError("Unknown state type.");
    }
    if (prob == 0) continue;
    parent->ApplyAction(action);
  }
  previous->second = Normalize(previous->second);
}

int GetBeliefHistorySize(HistoryDistribution* beliefs) {
  int belief_history_size = 0;
  for (int i = 0; i < beliefs->first.size(); ++i) {
    belief_history_size =
        std::max(belief_history_size,
                 static_cast<int>(beliefs->first[i]->History().size()));
  }
  return belief_history_size;
}

}  // namespace

std::unique_ptr<open_spiel::HistoryDistribution> CloneBeliefs(
    const open_spiel::HistoryDistribution& beliefs) {
  auto beliefs_copy = absl::make_unique<open_spiel::HistoryDistribution>();
  for (int i = 0; i < beliefs.first.size(); ++i) {
    beliefs_copy->first.push_back(beliefs.first[i]->Clone());
    beliefs_copy->second.push_back(beliefs.second[i]);
  }
  return beliefs_copy;
}


HistoryDistribution GetStateDistribution(const State& state,
                                         const Policy* opponent_policy) {
  std::shared_ptr<const Game> game = state.GetGame();
  GameType game_type = game->GetType();
  if (game_type.information == GameType::Information::kPerfectInformation) {
    HistoryDistribution dist;
    // We can't use brace initialization here as it triggers the copy ctor.
    dist.first.push_back(state.Clone());
    dist.second.push_back(1.);
    return dist;
  }
  SPIEL_CHECK_EQ(game_type.information,
                 GameType::Information::kImperfectInformation);
  SPIEL_CHECK_EQ(game_type.dynamics, GameType::Dynamics::kSequential);
  SPIEL_CHECK_NE(game_type.chance_mode,
                 GameType::ChanceMode::kSampledStochastic);
  SPIEL_CHECK_FALSE(state.IsChanceNode());
  SPIEL_CHECK_FALSE(state.IsTerminal());

  Player player = state.CurrentPlayer();
  std::string info_state_string = state.InformationStateString();

  // Generate the (info state, action) map for the current player using
  // the state's history.
  std::map<std::string, Action> infostate_action_map;
  std::vector<Action> history = state.History();
  std::unique_ptr<State> tmp_state = game->NewInitialState();
  for (Action action : history) {
    if (tmp_state->CurrentPlayer() == player) {
      infostate_action_map[tmp_state->InformationStateString()] = action;
    }
    tmp_state->ApplyAction(action);
  }
  // Add the current one to this list with an invalid action so that the
  // information state is included.
  infostate_action_map[info_state_string] = kInvalidAction;

  // Should get to the exact same state by re-applying the history.
  SPIEL_CHECK_EQ(tmp_state->ToString(), state.ToString());

  // Now, do a breadth-first search of all the candidate histories, removing
  // them whenever their (infostate, action) is not contained in the map above.
  // The search finishes when all the information states of the states in
  // the list have been found. We use two lists: final_states contains the ones
  // that have been found, while states are the current candidates.
  std::vector<std::unique_ptr<State>> final_states;
  std::vector<double> final_probs;
  std::vector<std::unique_ptr<State>> states;
  std::vector<double> probs;
  states.push_back(game->NewInitialState());
  probs.push_back(1.0);

  while (!states.empty()) {
    for (int idx = 0; idx < states.size();) {
      if (states[idx]->IsTerminal()) {
        // Terminal cannot be a valid history in an information state, so stop
        // considering this line.
      } else if (states[idx]->IsChanceNode()) {
        // At chance nodes, just add all the children and delete the state.
        for (std::pair<Action, double> action_and_prob :
             states[idx]->ChanceOutcomes()) {
          states.push_back(states[idx]->Child(action_and_prob.first));
          probs.push_back(probs[idx] * action_and_prob.second);
        }
      } else if (states[idx]->CurrentPlayer() != player) {
        // At opponent nodes, similar to chance nodes but get the probability
        // from the policy instead.
        std::string opp_infostate_str = states[idx]->InformationStateString();
        ActionsAndProbs state_policy =
            opponent_policy->GetStatePolicy(*states[idx]);
        for (Action action : states[idx]->LegalActions()) {
          double action_prob = GetProb(state_policy, action);
          states.push_back(states[idx]->Child(action));
          probs.push_back(probs[idx] * action_prob);
        }
      } else if (states[idx]->CurrentPlayer() == player) {
        std::string my_infostate_str = states[idx]->InformationStateString();
        // First check if this state is in the target information state. If
        // add it to the final set and don't check for expansion.
        if (my_infostate_str == info_state_string) {
          final_states.push_back(states[idx]->Clone());
          final_probs.push_back(probs[idx]);
        } else {
          // Check for expansion of this candidate. To expand this candidate,
          // the (infostate, action) pair must be contained in the map.
          for (Action action : states[idx]->LegalActions()) {
            auto iter = infostate_action_map.find(my_infostate_str);
            if (iter != infostate_action_map.end() && action == iter->second) {
              states.push_back(states[idx]->Child(action));
              probs.push_back(probs[idx]);
            }
          }
        }
      } else {
        SpielFatalError(
            absl::StrCat("Unknown player: ", states[idx]->CurrentPlayer()));
      }

      // Delete entries at the index i. Rather than call erase, which would
      // shift everything, simply swap with the last element and call
      // pop_back(), which can be done in constant time.
      std::swap(states[idx], states.back());
      std::swap(probs[idx], probs.back());
      states.pop_back();
      probs.pop_back();

      // Do not increment the counter index here because the current one points
      // to a valid state that was just expanded.
    }
  }

  // Now normalize the probs

  return {std::move(final_states), Normalize(final_probs)};
}

std::unique_ptr<HistoryDistribution> UpdateIncrementalStateDistribution(
    const State& state, const Policy* opponent_policy, int player_id,
    std::unique_ptr<HistoryDistribution> previous) {
  if (previous == nullptr) previous = std::make_unique<HistoryDistribution>();
  if (previous->first.empty()) {
    // If the previous pair is empty, then we have to do a BFS to find all
    // relevant nodes:
    return std::make_unique<HistoryDistribution>(
        GetStateDistribution(state, opponent_policy));
  }
  // The current state must be one action ahead of the dist ones.
  const std::vector<Action> history = state.History();
  int belief_history_size = GetBeliefHistorySize(previous.get());
  while (belief_history_size < history.size()) {
    AdvanceBeliefHistoryOneAction(previous.get(), history[belief_history_size],
                                  player_id, opponent_policy);
    belief_history_size = GetBeliefHistorySize(previous.get());
  }
  return previous;
}

std::string PrintBeliefs(const HistoryDistribution& beliefs) {
  const int num_states = beliefs.first.size();
  SPIEL_CHECK_EQ(num_states, beliefs.second.size());
  std::string str;
  for (int i = 0; i < num_states; ++i) {
    absl::StrAppend(
        &str, absl::StrFormat("(%s, %f)", beliefs.first[i]->HistoryString(),
                              beliefs.second[i]));
    if (i < num_states - 1) absl::StrAppend(&str, ", ");
  }
  return str;
}

}  // namespace open_spiel::algorithms
