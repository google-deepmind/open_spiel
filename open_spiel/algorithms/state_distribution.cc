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
#include "open_spiel/spiel_utils.h"

namespace open_spiel::algorithms {
namespace {

int GetBeliefHistorySize(const HistoryDistribution& beliefs) {
  int belief_history_size = 0;
  for (int i = 0; i < beliefs.first.size(); ++i) {
    belief_history_size =
        std::max(belief_history_size,
                 static_cast<int>(beliefs.first[i]->FullHistory().size()));
  }
  return belief_history_size;
}

std::unique_ptr<HistoryDistribution> AdvanceBeliefHistoryOneAction(
    std::unique_ptr<HistoryDistribution> previous, Action action,
    Player player_id, const Policy& opponent_policy) {
  auto dist = absl::make_unique<HistoryDistribution>();
  for (int i = 0; i < previous->first.size(); ++i) {
    std::unique_ptr<State>& state = previous->first[i];
    const double& prob = previous->second[i];
    if (Near(prob, 0.)) continue;
    switch (state->GetType()) {
      case StateType::kChance: {
        // If we can't find the action in the policy, then set it to 0.
        const double action_prob = GetProb(state->ChanceOutcomes(), action);

        // Then, skip all actions with 0 probability, as they don't matter
        // moving forward.
        if (Near(std::max(action_prob, 0.0), 0.0)) continue;
        SPIEL_CHECK_PROB(action_prob);

        // If we don't find the chance outcome, then the state we're in is
        // impossible, so we set it to zero.
        state->ApplyAction(action);

        dist->first.push_back(std::move(state));
        dist->second.push_back(prob * std::max(0.0, action_prob));
        break;
      }
      case StateType::kDecision: {
        if (state->CurrentPlayer() == player_id) {
          state->ApplyAction(action);
          dist->first.push_back(std::move(state));
          dist->second.push_back(prob);
        } else {
          // We have to add all actions as we don't know if the opponent is
          // taking a private or public action.
          // TODO(author1): Add method to open_spiel::State that lets us
          // only loop over the actions that are consistent with a given private
          // action.
          for (const auto& [candidate, action_prob] :
               opponent_policy.GetStatePolicy(*state)) {
            if (Near(std::max(0.0, action_prob), 0.0)) continue;
            SPIEL_CHECK_PROB(action_prob);
            std::unique_ptr<State> child = state->Child(candidate);
            if (child->IsTerminal()) continue;
            dist->first.push_back(std::move(child));
            dist->second.push_back(prob * action_prob);
          }
        }
        break;
      }
      case StateType::kTerminal:
        // If the state is terminal, and we have to advance by an action, we
        // discard the terminal histories from our beliefs.
        continue;
        // SpielFatalError("State is terminal, should not call
        // AdvanceBeliefs.");
      default:
        SpielFatalError(absl::StrCat("Unknown state type: ", state->GetType(),
                                     ", state: ", state->ToString()));
    }
  }
  return dist;
}

// Filters out all beliefs that do not belong to infostate.
std::unique_ptr<HistoryDistribution> FilterOutBeliefs(
    const State& state, std::unique_ptr<HistoryDistribution> dist,
    int player_id) {
  const std::string infostate = state.InformationStateString(player_id);
  auto new_dist = absl::make_unique<HistoryDistribution>();
  std::vector<int> good_indices;
  for (int i = 0; i < dist->first.size(); ++i) {
    if (dist->first[i]->InformationStateString(player_id) == infostate) {
      good_indices.push_back(i);
    }
  }
  new_dist->first.reserve(good_indices.size());
  new_dist->second.reserve(good_indices.size());
  for (int i : good_indices) {
    new_dist->first.push_back(std::move(dist->first[i]));
    new_dist->second.push_back(dist->second[i]);
  }
  return new_dist;
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
                                         const Policy& opponent_policy) {
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
            opponent_policy.GetStatePolicy(*states[idx]);
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
  Normalize(absl::MakeSpan(final_probs));
  HistoryDistribution dist = {std::move(final_states), std::move(final_probs)};

  // Note: We do not call CheckBeliefs here as the beliefs are _wrong_ until we
  // perform the filter step.
  return dist;
}

std::unique_ptr<HistoryDistribution> UpdateIncrementalStateDistribution(
    const State& state, const Policy& opponent_policy, int player_id,
    std::unique_ptr<HistoryDistribution> previous) {
  std::unique_ptr<HistoryDistribution> dist;
  if (previous) {
    dist = std::move(previous);
  }
  // If we don't have a previous set of beliefs, create it.
  if (!dist || dist->first.empty()) {
    // This allows for games to special case this scenario. It only works if
    // this is only called at the first decision node after chance nodes. We
    // leave it to the caller to verify this is the case.
    dist = state.GetHistoriesConsistentWithInfostate();

    // If the game didn't implement GetHistoriesConsistentWithInfostate, then
    // this is empty, otherwise, we're good.
    if (!dist || dist->first.empty()) {
      // If the previous pair is empty, then we have to do a BFS to find all
      // relevant nodes:
      dist = absl::make_unique<HistoryDistribution>(
          GetStateDistribution(state, opponent_policy));
    }
  }
  // Now, we verify that the beliefs match the current infostate.
  const std::vector<State::PlayerAction>& history = state.FullHistory();
  int belief_history_size = GetBeliefHistorySize(*dist);
  std::unique_ptr<State> new_state = state.GetGame()->NewInitialState();
  for (int i = 0; i < belief_history_size; ++i) {
    new_state->ApplyAction(history[i].action);
  }
  SPIEL_DCHECK_TRUE(CheckBeliefs(*new_state, *dist, player_id));
  while (belief_history_size < history.size()) {
    dist = AdvanceBeliefHistoryOneAction(std::move(dist),
                                         history[belief_history_size].action,
                                         player_id, opponent_policy);
    new_state->ApplyAction(history[belief_history_size].action);
    dist = FilterOutBeliefs(*new_state, std::move(dist), player_id);
    SPIEL_CHECK_FALSE(dist->first.empty());
    if (!new_state->IsChanceNode()) {
      SPIEL_DCHECK_TRUE(CheckBeliefs(*new_state, *dist, player_id));
    }
    const int new_belief_history_size = GetBeliefHistorySize(*dist);
    SPIEL_CHECK_LT(belief_history_size, new_belief_history_size);
    belief_history_size = new_belief_history_size;
  }
  SPIEL_CHECK_EQ(belief_history_size, history.size());
  SPIEL_CHECK_EQ(new_state->FullHistory(), state.FullHistory());
  dist = FilterOutBeliefs(state, std::move(dist), player_id);
  SPIEL_CHECK_FALSE(dist->first.empty());

  // We only normalize after filtering out invalid infostates.
  Normalize(absl::MakeSpan(dist->second));

  SPIEL_DCHECK_TRUE(CheckBeliefs(state, *dist, player_id));
  return dist;
}

std::string PrintBeliefs(const HistoryDistribution& beliefs, int player_id) {
  const int num_states = beliefs.first.size();
  SPIEL_CHECK_EQ(num_states, beliefs.second.size());
  std::string str;
  for (int i = 0; i < num_states; ++i) {
    absl::StrAppend(
        &str,
        absl::StrFormat("(%s, %f)",
                        beliefs.first[i]->InformationStateString(player_id),
                        beliefs.second[i]));
    if (i < num_states - 1) absl::StrAppend(&str, "\n");
  }
  return str;
}

bool CheckBeliefs(const State& ground_truth_state,
                  const HistoryDistribution& beliefs, int player_id) {
  const std::string infostate =
      ground_truth_state.InformationStateString(player_id);
  for (int i = 0; i < beliefs.first.size(); ++i) {
    if (Near(beliefs.second[i], 0.0, 1e-5)) {
      continue;
    }
    SPIEL_CHECK_EQ(ground_truth_state.FullHistory().size(),
                   beliefs.first[i]->FullHistory().size());
    SPIEL_CHECK_EQ(infostate,
                   beliefs.first[i]->InformationStateString(player_id));
    SPIEL_CHECK_EQ(ground_truth_state.FullHistory().size(),
                   beliefs.first[i]->FullHistory().size());
    SPIEL_CHECK_EQ(ground_truth_state.IsTerminal(),
                   beliefs.first[i]->IsTerminal());
  }
  return true;
}

}  // namespace open_spiel::algorithms
