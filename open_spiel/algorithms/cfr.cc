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

#include "open_spiel/algorithms/cfr.h"

#include <algorithm>
#include <array>
#include <random>

#include "open_spiel/abseil-cpp/absl/algorithm/container.h"
#include "open_spiel/abseil-cpp/absl/strings/charconv.h"
#include "open_spiel/abseil-cpp/absl/strings/numbers.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/utils/serialization.h"

namespace open_spiel {
namespace algorithms {
namespace {
inline constexpr double kRandomInitialRegretsMagnitude = 0.001;
}  // namespace

constexpr const int kSerializationVersion = 1;

// All CFR solvers support serialization. CFRSolver, CFRPlusSolver and
// CFRBRSolver all rely on CFRSolverBase::Serialize() to provide the
// functionality (each subtype only implements the SerializeThisType() method
// which returns the name of the serialized subclass).
// ExternalSamplingMCCFRSolver and OutcomeSamplingMCCFRSolver implement
// their own versions of Serialize().
//
// During the serialization we store multiple different sections which are
// described below:
//   - [SolverType] is the name of the serialized class.
//   - [SolverSpecificState] is a section to which different solvers write state
//       that is specific to their own type. Note that serialization and
//       deserialization of this section is left entirely to each of the
//       subtypes.
//   - [SolverValuesTable] is the section to which CFRInfoStateValuesTable is
//       written.
//
// During deserialization all solvers rely on the
// PartiallyDeserializeCFRSolver() method which deserializes common properties
// but leaves deserialization of [SolverSpecificState] section to the caller.
//
// Note that there are some specifics that need to be taken into account when
// reading the de/serialization code:
//   - Each solver's Serialize() method has two parameters which are not exposed
//       in Python, i.e. int double_precision and std::string delimiter.
//   - We try to avoid copying/moving of CFRInfoStateValuesTable where possible
//       due to its potentially large size (mainly due to memory concerns). For
//       that reason the PartiallyDeserializeCFRSolver() method also returns a
//       string_view of the table and leaves deserialization to the callers
//       which then in turn call the efficient
//       DeserializeCFRInfoStateValuesTable().

CFRAveragePolicy::CFRAveragePolicy(const CFRInfoStateValuesTable& info_states,
                                   std::shared_ptr<Policy> default_policy)
    : info_states_(info_states), default_policy_(default_policy) {}

ActionsAndProbs CFRAveragePolicy::GetStatePolicy(
    const State& state, Player player) const {
  auto entry = info_states_.find(state.InformationStateString(player));
  if (entry == info_states_.end()) {
    if (default_policy_) {
      return default_policy_->GetStatePolicy(state, player);
    } else {
      // This should never get called.
      SpielFatalError("No policy found, and no default policy.");
    }
  }
  ActionsAndProbs actions_and_probs;
  GetStatePolicyFromInformationStateValues(entry->second, &actions_and_probs);
  return actions_and_probs;
}

ActionsAndProbs CFRAveragePolicy::GetStatePolicy(
    const std::string& info_state) const {
  auto entry = info_states_.find(info_state);
  if (entry == info_states_.end()) {
    if (default_policy_) {
      return default_policy_->GetStatePolicy(info_state);
    } else {
      // This should never get called.
      SpielFatalError("No policy found, and no default policy.");
    }
  }
  ActionsAndProbs actions_and_probs;
  GetStatePolicyFromInformationStateValues(entry->second, &actions_and_probs);
  return actions_and_probs;
}

void CFRAveragePolicy::GetStatePolicyFromInformationStateValues(
    const CFRInfoStateValues& is_vals,
    ActionsAndProbs* actions_and_probs) const {
  double sum_prob = 0.0;
  for (int aidx = 0; aidx < is_vals.num_actions(); ++aidx) {
    sum_prob += is_vals.cumulative_policy[aidx];
  }

  if (sum_prob == 0.0) {
    // Return a uniform policy at this node
    double prob = 1. / is_vals.num_actions();
    for (Action action : is_vals.legal_actions) {
      actions_and_probs->push_back({action, prob});
    }
    return;
  }

  for (int aidx = 0; aidx < is_vals.num_actions(); ++aidx) {
    actions_and_probs->push_back({is_vals.legal_actions[aidx],
                                  is_vals.cumulative_policy[aidx] / sum_prob});
  }
}

TabularPolicy CFRAveragePolicy::AsTabular() const {
  TabularPolicy policy;
  for (const auto& infoset_and_entry : info_states_) {
    ActionsAndProbs state_policy;
    GetStatePolicyFromInformationStateValues(infoset_and_entry.second,
                                             &state_policy);
    policy.SetStatePolicy(infoset_and_entry.first, state_policy);
  }
  return policy;
}

CFRCurrentPolicy::CFRCurrentPolicy(const CFRInfoStateValuesTable& info_states,
                                   std::shared_ptr<Policy> default_policy)
    : info_states_(info_states), default_policy_(default_policy) {}

ActionsAndProbs CFRCurrentPolicy::GetStatePolicy(
    const State& state, Player player) const {
  auto entry = info_states_.find(state.InformationStateString(player));
  if (entry == info_states_.end()) {
    if (default_policy_) {
      return default_policy_->GetStatePolicy(state, player);
    } else {
      SpielFatalError("No policy found, and no default policy.");
    }
  }
  ActionsAndProbs actions_and_probs;
  return GetStatePolicyFromInformationStateValues(entry->second,
                                                  actions_and_probs);
}

ActionsAndProbs CFRCurrentPolicy::GetStatePolicy(
    const std::string& info_state) const {
  auto entry = info_states_.find(info_state);
  if (entry == info_states_.end()) {
    if (default_policy_) {
      return default_policy_->GetStatePolicy(info_state);
    } else {
      SpielFatalError("No policy found, and no default policy.");
    }
  }
  ActionsAndProbs actions_and_probs;
  GetStatePolicyFromInformationStateValues(entry->second, actions_and_probs);
  return actions_and_probs;
}

ActionsAndProbs CFRCurrentPolicy::GetStatePolicyFromInformationStateValues(
    const CFRInfoStateValues& is_vals,
    ActionsAndProbs& actions_and_probs) const {
  for (int aidx = 0; aidx < is_vals.num_actions(); ++aidx) {
    actions_and_probs.push_back(
        {is_vals.legal_actions[aidx], is_vals.current_policy[aidx]});
  }
  return actions_and_probs;
}

TabularPolicy CFRCurrentPolicy::AsTabular() const {
  TabularPolicy policy;
  for (const auto& infoset_and_entry : info_states_) {
    policy.SetStatePolicy(infoset_and_entry.first,
                          infoset_and_entry.second.GetCurrentPolicy());
  }
  return policy;
}

CFRSolverBase::CFRSolverBase(const Game& game, bool alternating_updates,
                             bool linear_averaging, bool regret_matching_plus,
                             bool random_initial_regrets, int seed)
    : game_(game.shared_from_this()),
      root_state_(game.NewInitialState()),
      root_reach_probs_(game_->NumPlayers() + 1, 1.0),
      regret_matching_plus_(regret_matching_plus),
      alternating_updates_(alternating_updates),
      linear_averaging_(linear_averaging),
      random_initial_regrets_(random_initial_regrets),
      chance_player_(game.NumPlayers()),
      rng_(seed) {
  if (game_->GetType().dynamics != GameType::Dynamics::kSequential) {
    SpielFatalError(
        "CFR requires sequential games. If you're trying to run it "
        "on a simultaneous (or normal-form) game, please first transform it "
        "using turn_based_simultaneous_game.");
  }
  InitializeInfostateNodes(*root_state_);
}

CFRSolverBase::CFRSolverBase(std::shared_ptr<const Game> game,
                             bool alternating_updates, bool linear_averaging,
                             bool regret_matching_plus, int iteration,
                             bool random_initial_regrets, int seed)
    : game_(game),
      iteration_(iteration),
      root_state_(game->NewInitialState()),
      root_reach_probs_(game_->NumPlayers() + 1, 1.0),
      regret_matching_plus_(regret_matching_plus),
      alternating_updates_(alternating_updates),
      linear_averaging_(linear_averaging),
      random_initial_regrets_(random_initial_regrets),
      chance_player_(game->NumPlayers()),
      rng_(seed) {
  if (game_->GetType().dynamics != GameType::Dynamics::kSequential) {
    SpielFatalError(
        "CFR requires sequential games. If you're trying to run it "
        "on a simultaneous (or normal-form) game, please first transform it "
        "using turn_based_simultaneous_game.");
  }
}

void CFRSolverBase::InitializeInfostateNodes(const State& state) {
  if (state.IsTerminal()) {
    return;
  }
  if (state.IsChanceNode()) {
    for (const auto& action_prob : state.ChanceOutcomes()) {
      InitializeInfostateNodes(*state.Child(action_prob.first));
    }
    return;
  }

  int current_player = state.CurrentPlayer();
  std::string info_state = state.InformationStateString(current_player);
  std::vector<Action> legal_actions = state.LegalActions();

  if (random_initial_regrets_) {
    CFRInfoStateValues is_vals(legal_actions, &rng_,
                               kRandomInitialRegretsMagnitude);
    info_states_[info_state] = is_vals;
  } else {
    CFRInfoStateValues is_vals(legal_actions);
    info_states_[info_state] = is_vals;
  }

  for (const Action& action : legal_actions) {
    InitializeInfostateNodes(*state.Child(action));
  }
}

void CFRSolverBase::EvaluateAndUpdatePolicy() {
  ++iteration_;
  if (alternating_updates_) {
    for (int player = 0; player < game_->NumPlayers(); player++) {
      ComputeCounterFactualRegret(*root_state_, player, root_reach_probs_,
                                  nullptr);
      if (regret_matching_plus_) {
        ApplyRegretMatchingPlusReset();
      }
      ApplyRegretMatching();
    }
  } else {
    ComputeCounterFactualRegret(*root_state_, absl::nullopt, root_reach_probs_,
                                nullptr);
    if (regret_matching_plus_) {
      ApplyRegretMatchingPlusReset();
    }
    ApplyRegretMatching();
  }
}

std::string CFRSolverBase::Serialize(int double_precision,
                                     std::string delimiter) const {
  SPIEL_CHECK_GE(double_precision, -1);
  std::string str = "";
  // Meta section
  absl::StrAppend(&str,
                  "# Automatically generated by OpenSpiel "
                  "CFRSolverBase::Serialize\n");
  absl::StrAppend(&str, kSerializeMetaSectionHeader, "\n");
  absl::StrAppend(&str, "Version: ", kSerializationVersion, "\n");
  absl::StrAppend(&str, "\n");
  // Game section
  absl::StrAppend(&str, kSerializeGameSectionHeader, "\n");
  absl::StrAppend(&str, game_->Serialize(), "\n");
  // Internal solver state section
  absl::StrAppend(&str, kSerializeSolverTypeSectionHeader, "\n");
  absl::StrAppend(&str, SerializeThisType(), "\n");
  absl::StrAppend(&str, kSerializeSolverSpecificStateSectionHeader, "\n");
  absl::StrAppend(&str, iteration_, "\n");
  absl::StrAppend(&str, kSerializeSolverValuesTableSectionHeader, "\n");
  SerializeCFRInfoStateValuesTable(info_states_, &str, double_precision,
                                   delimiter);
  return str;
}

static double CounterFactualReachProb(
    const std::vector<double>& reach_probabilities, const int player) {
  double cfr_reach_prob = 1.0;
  for (int i = 0; i < reach_probabilities.size(); i++) {
    if (i != player) {
      cfr_reach_prob *= reach_probabilities[i];
    }
  }
  return cfr_reach_prob;
}

// Compute counterfactual regrets. Alternates recursively with
// ComputeCounterFactualRegretForActionProbs.
//
// Args:
// - state: The state to start the recursion.
// - alternating_player: Optionally only update this player.
// - reach_probabilities: The reach probabilities of this state for each
//      player, ending with the chance player.
//
// Returns:
//   The value of the state for each player (excluding the chance player).
std::vector<double> CFRSolverBase::ComputeCounterFactualRegret(
    const State& state, const absl::optional<int>& alternating_player,
    const std::vector<double>& reach_probabilities,
    const std::vector<const Policy*>* policy_overrides) {
  if (state.IsTerminal()) {
    return state.Returns();
  }
  if (state.IsChanceNode()) {
    ActionsAndProbs actions_and_probs = state.ChanceOutcomes();
    std::vector<double> dist(actions_and_probs.size(), 0);
    std::vector<Action> outcomes(actions_and_probs.size(), 0);
    for (int oidx = 0; oidx < actions_and_probs.size(); ++oidx) {
      outcomes[oidx] = actions_and_probs[oidx].first;
      dist[oidx] = actions_and_probs[oidx].second;
    }
    return ComputeCounterFactualRegretForActionProbs(
        state, alternating_player, reach_probabilities, chance_player_, dist,
        outcomes, nullptr, policy_overrides);
  }
  if (AllPlayersHaveZeroReachProb(reach_probabilities)) {
    // The value returned is not used: if the reach probability for all players
    // is 0, then the last taken action has probability 0, so the
    // returned value is not impacting the parent node value.
    return std::vector<double>(game_->NumPlayers(), 0.0);
  }

  int current_player = state.CurrentPlayer();
  std::string info_state = state.InformationStateString();
  std::vector<Action> legal_actions = state.LegalActions(current_player);

  // Load current policy.
  std::vector<double> info_state_policy;
  if (policy_overrides && policy_overrides->at(current_player)) {
    GetInfoStatePolicyFromPolicy(&info_state_policy, legal_actions,
                                 policy_overrides->at(current_player),
                                 info_state);
  } else {
    info_state_policy = GetPolicy(info_state, legal_actions);
  }

  std::vector<double> child_utilities;
  child_utilities.reserve(legal_actions.size());
  const std::vector<double> state_value =
      ComputeCounterFactualRegretForActionProbs(
          state, alternating_player, reach_probabilities, current_player,
          info_state_policy, legal_actions, &child_utilities, policy_overrides);

  // Perform regret and average strategy updates.
  if (!alternating_player || *alternating_player == current_player) {
    CFRInfoStateValues is_vals = info_states_[info_state];
    SPIEL_CHECK_FALSE(is_vals.empty());

    const double self_reach_prob = reach_probabilities[current_player];
    const double cfr_reach_prob =
        CounterFactualReachProb(reach_probabilities, current_player);

    for (int aidx = 0; aidx < legal_actions.size(); ++aidx) {
      // Update regrets.
      double cfr_regret = cfr_reach_prob *
                          (child_utilities[aidx] - state_value[current_player]);

      is_vals.cumulative_regrets[aidx] += cfr_regret;

      // Update average policy.
      if (linear_averaging_) {
        is_vals.cumulative_policy[aidx] +=
            iteration_ * self_reach_prob * info_state_policy[aidx];
      } else {
        is_vals.cumulative_policy[aidx] +=
            self_reach_prob * info_state_policy[aidx];
      }
    }

    info_states_[info_state] = is_vals;
  }

  return state_value;
}

void CFRSolverBase::GetInfoStatePolicyFromPolicy(
    std::vector<double>* info_state_policy,
    const std::vector<Action>& legal_actions, const Policy* policy,
    const std::string& info_state) const {
  ActionsAndProbs actions_and_probs = policy->GetStatePolicy(info_state);
  info_state_policy->reserve(legal_actions.size());

  // The policy may have extra ones not at this infostate
  for (Action action : legal_actions) {
    const auto& iter =
        std::find_if(actions_and_probs.begin(), actions_and_probs.end(),
                     [action](const std::pair<Action, double>& ap) {
                       return ap.first == action;
                     });
    info_state_policy->push_back(iter->second);
  }

  SPIEL_CHECK_EQ(info_state_policy->size(), legal_actions.size());
}

// Compute counterfactual regrets given certain action probabilities.
// Alternates recursively with ComputeCounterFactualRegret.
//
// Args:
// - state: The state to start the recursion.
// - alternating_player: Optionally only update this player.
// - reach_probabilities: The reach probabilities of this state.
// - current_player: Either a player or chance_player_.
// - action_probs: The action probabilities to use for this state.
// - child_values_out: optional output parameter which is filled with the child
//           utilities for each action, for current_player.
// Returns:
//   The value of the state for each player (excluding the chance player).
std::vector<double> CFRSolverBase::ComputeCounterFactualRegretForActionProbs(
    const State& state, const absl::optional<int>& alternating_player,
    const std::vector<double>& reach_probabilities, const int current_player,
    const std::vector<double>& info_state_policy,
    const std::vector<Action>& legal_actions,
    std::vector<double>* child_values_out,
    const std::vector<const Policy*>* policy_overrides) {
  std::vector<double> state_value(game_->NumPlayers());

  for (int aidx = 0; aidx < legal_actions.size(); ++aidx) {
    const Action action = legal_actions[aidx];
    const double prob = info_state_policy[aidx];
    const std::unique_ptr<State> new_state = state.Child(action);
    std::vector<double> new_reach_probabilities(reach_probabilities);
    new_reach_probabilities[current_player] *= prob;
    std::vector<double> child_value =
        ComputeCounterFactualRegret(*new_state, alternating_player,
                                    new_reach_probabilities, policy_overrides);
    for (int i = 0; i < state_value.size(); ++i) {
      state_value[i] += prob * child_value[i];
    }
    if (child_values_out != nullptr) {
      child_values_out->push_back(child_value[current_player]);
    }
  }
  return state_value;
}

bool CFRSolverBase::AllPlayersHaveZeroReachProb(
    const std::vector<double>& reach_probabilities) const {
  for (int i = 0; i < game_->NumPlayers(); i++) {
    if (reach_probabilities[i] != 0.0) {
      return false;
    }
  }
  return true;
}

std::vector<double> CFRSolverBase::GetPolicy(
    const std::string& info_state, const std::vector<Action>& legal_actions) {
  auto entry = info_states_.find(info_state);
  if (entry == info_states_.end()) {
    info_states_[info_state] = CFRInfoStateValues(legal_actions);
    entry = info_states_.find(info_state);
  }

  SPIEL_CHECK_FALSE(entry == info_states_.end());
  SPIEL_CHECK_FALSE(entry->second.empty());
  SPIEL_CHECK_FALSE(entry->second.current_policy.empty());
  return entry->second.current_policy;
}

std::string CFRInfoStateValues::ToString() const {
  std::string str = "";
  absl::StrAppend(&str, "Legal actions: ", absl::StrJoin(legal_actions, ", "),
                  "\n");
  absl::StrAppend(&str, "Current policy: ", absl::StrJoin(current_policy, ", "),
                  "\n");
  absl::StrAppend(&str, "Cumulative regrets: ",
                  absl::StrJoin(cumulative_regrets, ", "), "\n");
  absl::StrAppend(&str,
                  "Cumulative policy: ", absl::StrJoin(cumulative_policy, ", "),
                  "\n");
  return str;
}

std::string CFRInfoStateValues::Serialize(int double_precision) const {
  std::string str = "";
  std::string cumulative_regrets_str, cumulative_policy_str, current_policy_str;
  if (double_precision == -1) {
    cumulative_regrets_str =
        absl::StrJoin(cumulative_regrets, ",", HexDoubleFormatter());
    cumulative_policy_str =
        absl::StrJoin(cumulative_policy, ",", HexDoubleFormatter());
    current_policy_str =
        absl::StrJoin(current_policy, ",", HexDoubleFormatter());
  } else {
    cumulative_regrets_str = absl::StrJoin(
        cumulative_regrets, ",", SimpleDoubleFormatter(double_precision));
    cumulative_policy_str = absl::StrJoin(
        cumulative_policy, ",", SimpleDoubleFormatter(double_precision));
    current_policy_str = absl::StrJoin(current_policy, ",",
                                       SimpleDoubleFormatter(double_precision));
  }
  absl::StrAppend(&str, absl::StrJoin(legal_actions, ","), ";");
  absl::StrAppend(&str, cumulative_regrets_str, ";");
  absl::StrAppend(&str, cumulative_policy_str, ";");
  absl::StrAppend(&str, current_policy_str);
  return str;
}

CFRInfoStateValues DeserializeCFRInfoStateValues(absl::string_view serialized) {
  CFRInfoStateValues res = CFRInfoStateValues();
  if (serialized.empty()) return res;

  std::vector<std::vector<absl::string_view>> str_values;
  str_values.reserve(4);
  for (absl::string_view sv : absl::StrSplit(serialized, ';')) {
    str_values.push_back(absl::StrSplit(sv, ','));
  }

  int num_elements = str_values.at(0).size();
  res.legal_actions.reserve(num_elements);
  res.cumulative_regrets.reserve(num_elements);
  res.cumulative_policy.reserve(num_elements);
  res.current_policy.reserve(num_elements);

  // Insert the actual values
  int la_value;
  double cumu_regret_value, cumu_policy_value, curr_policy_value;
  for (int i = 0; i < num_elements; i++) {
    SPIEL_CHECK_TRUE(absl::SimpleAtoi(str_values.at(0).at(i), &la_value));
    absl::from_chars(
        str_values.at(1).at(i).data(),
        str_values.at(1).at(i).data() + str_values.at(1).at(i).size(),
        cumu_regret_value);
    absl::from_chars(
        str_values.at(2).at(i).data(),
        str_values.at(2).at(i).data() + str_values.at(2).at(i).size(),
        cumu_policy_value);
    absl::from_chars(
        str_values.at(3).at(i).data(),
        str_values.at(3).at(i).data() + str_values.at(3).at(i).size(),
        curr_policy_value);

    res.legal_actions.push_back(la_value);
    res.cumulative_regrets.push_back(cumu_regret_value);
    res.cumulative_policy.push_back(cumu_policy_value);
    res.current_policy.push_back(curr_policy_value);
  }
  return res;
}

ActionsAndProbs CFRInfoStateValues::GetCurrentPolicy() const {
  ActionsAndProbs actions_and_probs;
  actions_and_probs.reserve(legal_actions.size());
  for (int i = 0; i < legal_actions.size(); ++i) {
    actions_and_probs.push_back({legal_actions[i], current_policy[i]});
  }
  return actions_and_probs;
}

void CFRInfoStateValues::ApplyRegretMatchingAllPositive(double delta) {
  SPIEL_CHECK_GT(delta, 0);
  double sum = 0;
  for (int aidx = 0; aidx < num_actions(); ++aidx) {
    sum += std::max(cumulative_regrets[aidx], delta);
  }
  for (int aidx = 0; aidx < num_actions(); ++aidx) {
    current_policy[aidx] = std::max(cumulative_regrets[aidx], delta) / sum;
  }
}

void CFRInfoStateValues::ApplyRegretMatching() {
  double sum_positive_regrets = 0.0;

  for (int aidx = 0; aidx < num_actions(); ++aidx) {
    if (cumulative_regrets[aidx] > 0) {
      sum_positive_regrets += cumulative_regrets[aidx];
    }
  }

  for (int aidx = 0; aidx < num_actions(); ++aidx) {
    if (sum_positive_regrets > 0) {
      current_policy[aidx] =
          cumulative_regrets[aidx] > 0
              ? cumulative_regrets[aidx] / sum_positive_regrets
              : 0;
    } else {
      current_policy[aidx] = 1.0 / legal_actions.size();
    }
  }
}

int CFRInfoStateValues::SampleActionIndex(double epsilon, double z) {
  double sum = 0;
  for (int aidx = 0; aidx < current_policy.size(); ++aidx) {
    double prob = epsilon * 1.0 / current_policy.size() +
                  (1.0 - epsilon) * current_policy[aidx];
    if (z >= sum && z < sum + prob) {
      return aidx;
    }
    sum += prob;
  }
  SpielFatalError(absl::StrCat("SampleActionIndex: sum of probs is ", sum));
}

int CFRInfoStateValues::GetActionIndex(Action a) {
  auto it = std::find(legal_actions.begin(), legal_actions.end(), a);
  if (it != legal_actions.end()) {
    return std::distance(legal_actions.begin(), it);
  }
  SpielFatalError(
      absl::StrCat("GetActionIndex: the action was not found: ", a));
}

void SerializeCFRInfoStateValuesTable(
    const CFRInfoStateValuesTable& info_states, std::string* result,
    int double_precision, std::string delimiter) {
  if (delimiter == "," || delimiter == ";") {
    // The two delimiters are used for de/serialization of CFRInfoStateValues
    SpielFatalError(
        "Please select a different delimiter,"
        "invalid values are \",\" and \";\".");
  }
  if (info_states.empty()) return;

  for (auto const& [info_state, values] : info_states) {
    if (info_state.find(delimiter) != std::string::npos) {
      SpielFatalError(absl::StrCat(
          "Info state contains delimiter \"", delimiter,
          "\", please fix the info state or select a different delimiter."));
    }
    absl::StrAppend(result, info_state, delimiter,
                    values.Serialize(double_precision), delimiter);
  }
  // Remove the trailing delimiter
  result->erase(result->length() - delimiter.length());
}

void DeserializeCFRInfoStateValuesTable(absl::string_view serialized,
                                        CFRInfoStateValuesTable* result,
                                        std::string delimiter) {
  if (serialized.empty()) return;

  std::vector<absl::string_view> splits = absl::StrSplit(serialized, delimiter);
  for (int i = 0; i < splits.size(); i += 2) {
    result->insert({std::string(splits.at(i)),
                    DeserializeCFRInfoStateValues(splits.at(i + 1))});
  }
}

//  Resets negative cumulative regrets to 0.
//
//  Regret Matching+ corresponds to the following cumulative regrets update:
//  cumulative_regrets = max(cumulative_regrets + regrets, 0)
//
//  This must be done at the level of the information set, and thus cannot be
//  done during the tree traversal (which is done on histories). It is thus
//  performed as an additional step.
void CFRSolverBase::ApplyRegretMatchingPlusReset() {
  for (auto& entry : info_states_) {
    for (int aidx = 0; aidx < entry.second.num_actions(); ++aidx) {
      if (entry.second.cumulative_regrets[aidx] < 0) {
        entry.second.cumulative_regrets[aidx] = 0;
      }
    }
  }
}

void CFRSolverBase::ApplyRegretMatching() {
  for (auto& entry : info_states_) {
    entry.second.ApplyRegretMatching();
  }
}

std::unique_ptr<CFRSolver> DeserializeCFRSolver(const std::string& serialized,
                                                std::string delimiter) {
  auto partial = PartiallyDeserializeCFRSolver(serialized);
  SPIEL_CHECK_EQ(partial.solver_type, "CFRSolver");
  auto solver = std::make_unique<CFRSolver>(
      partial.game, std::stoi(partial.solver_specific_state));
  DeserializeCFRInfoStateValuesTable(partial.serialized_cfr_values_table,
                                     &solver->InfoStateValuesTable(),
                                     delimiter);
  return solver;
}

std::unique_ptr<CFRPlusSolver> DeserializeCFRPlusSolver(
    const std::string& serialized, std::string delimiter) {
  auto partial = PartiallyDeserializeCFRSolver(serialized);
  SPIEL_CHECK_EQ(partial.solver_type, "CFRPlusSolver");
  auto solver = std::make_unique<CFRPlusSolver>(
      partial.game, std::stoi(partial.solver_specific_state));
  DeserializeCFRInfoStateValuesTable(partial.serialized_cfr_values_table,
                                     &solver->InfoStateValuesTable(),
                                     delimiter);
  return solver;
}

PartiallyDeserializedCFRSolver PartiallyDeserializeCFRSolver(
    const std::string& serialized) {
  // We don't copy the CFR values table section due to potential large size.
  enum Section {
    kInvalid = -1,
    kMeta = 0,
    kGame = 1,
    kSolverType = 2,
    kSolverSpecificState = 3
  };
  std::array<std::string, 4> section_strings = {"", "", "", ""};
  Section current_section = kInvalid;

  std::vector<absl::string_view> lines = absl::StrSplit(serialized, '\n');
  for (int i = 0; i < lines.size(); i++) {
    if (lines[i].length() == 0 || lines[i].at(0) == '#') {
      // Skip comments and blank lines
    } else if (lines[i] == kSerializeMetaSectionHeader) {
      SPIEL_CHECK_EQ(current_section, kInvalid);
      current_section = kMeta;
    } else if (lines[i] == kSerializeGameSectionHeader) {
      SPIEL_CHECK_EQ(current_section, kMeta);
      current_section = kGame;
    } else if (lines[i] == kSerializeSolverTypeSectionHeader) {
      SPIEL_CHECK_EQ(current_section, kGame);
      current_section = kSolverType;
    } else if (lines[i] == kSerializeSolverSpecificStateSectionHeader) {
      SPIEL_CHECK_EQ(current_section, kSolverType);
      current_section = kSolverSpecificState;
    } else if (lines[i] == kSerializeSolverValuesTableSectionHeader) {
      SPIEL_CHECK_EQ(current_section, kSolverSpecificState);
      break;
    } else {
      SPIEL_CHECK_NE(current_section, kInvalid);
      if (current_section == kSolverSpecificState) {
        absl::StrAppend(&section_strings[current_section], lines[i], "\n");
      } else {
        absl::StrAppend(&section_strings[current_section], lines[i]);
      }
    }
  }

  // We currently just ignore the meta section.
  // In order to avod copying the CFR values table data we rather split it again
  // and obtain a single string_view that can be deserialized later using the
  // DeserializeCFRInfoStateValuesTable method.
  std::pair<absl::string_view, absl::string_view> other_and_values_table_data =
      absl::StrSplit(
          serialized,
          absl::StrCat(kSerializeSolverValuesTableSectionHeader, "\n"));
  return PartiallyDeserializedCFRSolver(DeserializeGame(section_strings[kGame]),
                                        section_strings[kSolverType],
                                        section_strings[kSolverSpecificState],
                                        other_and_values_table_data.second);
}

}  // namespace algorithms
}  // namespace open_spiel

