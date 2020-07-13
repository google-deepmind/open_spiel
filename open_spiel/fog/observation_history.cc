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

#include "open_spiel/fog/observation_history.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {

bool ActionOrObs::operator==(const ActionOrObs& other) const {
  if (tag != other.tag) return false;
  if (tag == Either::kAction) return action == other.action;
  if (tag == Either::kObservation) return observation == other.observation;
  SpielFatalError("Unknown tag.");
  return false;  // This will never return.
}

std::string ActionOrObs::ToString() const {
  if (tag == ActionOrObs::Either::kAction) {
    return absl::StrCat("action='", action, "'");
  }
  if (tag == ActionOrObs::Either::kObservation) {
    return absl::StrCat("observation='", observation, "'");
  }
  SpielFatalError("Unknown tag.");
  return "";  // This will never return.
}

ActionObservationHistory::ActionObservationHistory(
    Player player, const State& target) : player_(player) {
  SPIEL_CHECK_GE(player_, 0);
  SPIEL_CHECK_LT(player_, target.NumPlayers());
  SPIEL_CHECK_TRUE(target.GetGame()->GetType().provides_observation_string);
  const std::vector<State::PlayerAction>& history = target.FullHistory();
  history_.reserve(2 * history.size());

  std::unique_ptr<State> state = target.GetGame()->NewInitialState();
  for (int i = 0; i < history.size(); i++) {
    const auto& [history_player, action] = history[i];
    push_back(state->ObservationString(player));
    if (state->CurrentPlayer() == player) {
      push_back(action);
    }
    state->ApplyAction(action);
  }
  push_back(state->ObservationString(player));
}

ActionObservationHistory::ActionObservationHistory(const State& target)
    : ActionObservationHistory(target.CurrentPlayer(), target) {}

ActionObservationHistory::ActionObservationHistory(
    Player player, std::vector<ActionOrObs> history)
    : history_(std::move(history)), player_(player) {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_FALSE(history_.empty());  // There is always an obs for root node.
  SPIEL_CHECK_TRUE(history_[0].IsObservation());
  for (int i = 1; i < history.size(); i++)
    // Implication  history[i].IsAction() => history[i-1].IsObservation()
    SPIEL_CHECK_TRUE(!history[i].IsAction() || history[i-1].IsObservation());
}

bool ActionObservationHistory::IsPrefix(
    const ActionObservationHistory& other) const {
  const auto& a = history_;
  const auto& b = other.history_;
  if (a.empty()) return true;
  if (b.empty()) return false;  // True only if a is empty, handled before.
  if (a.size() > b.size()) return false;
  if (a.size() == b.size()) return a == b;
  return std::equal(a.begin(), a.end(), b.begin());
}

std::string ActionObservationHistory::ToString() const {
  return absl::StrJoin(history_, ", ", absl::StreamFormatter());
}

void ActionObservationHistory::push_back(const ActionOrObs& aoo) {
  // Action observation history must start with an initial observation.
  const bool starts_with_observation =
      !history_.empty() || aoo.IsObservation();
  SPIEL_CHECK_TRUE(starts_with_observation);

  // Action observation history cannot have two consecutive actions pushed,
  // there should be at least one observation between them.
  const bool not_two_consecutive_actions =
      (history_.empty()) || (aoo.IsAction() && history_.back().IsObservation());
  SPIEL_CHECK_TRUE(aoo.IsObservation() || not_two_consecutive_actions);

  history_.push_back(aoo);
}

bool PublicObservationHistory::IsPrefix(
    const PublicObservationHistory& other) const {
  const auto& a = history_;
  const auto& b = other.history_;
  if (a.empty()) return true;
  if (b.empty()) return false;  // True only if a is empty, handled before.
  if (a.size() > b.size()) return false;
  if (a.size() == b.size()) return a == b;
  return std::equal(a.begin(), a.end(), b.begin());
}

PublicObservationHistory::PublicObservationHistory(const State& target) {
  SPIEL_CHECK_TRUE(
      target.GetGame()->GetType().provides_factored_observation_string);
  history_.reserve(target.FullHistory().size());

  std::unique_ptr<State> state = target.GetGame()->NewInitialState();
  // Use FullHistory even though we don't need the player -- prevent
  // doing a copy.
  for (const auto& [player, action] : target.FullHistory()) {
    history_.push_back(state->PublicObservationString());
    state->ApplyAction(action);
  }
  history_.push_back(state->PublicObservationString());
}

PublicObservationHistory::PublicObservationHistory(
    std::vector<std::string> history) : history_(std::move(history)) {
  SPIEL_CHECK_FALSE(history_.empty());
  SPIEL_CHECK_EQ(history_.at(0), kStartOfGamePublicObservation);
}

std::string PublicObservationHistory::ToString() const {
  return absl::StrJoin(history_, ", ");
}

std::ostream& operator<<(std::ostream& os, const ActionOrObs& aoo) {
  return os << aoo.ToString();
}

std::ostream& operator<<(
    std::ostream& os, const ActionObservationHistory& aoh) {
  return os << aoh.ToString();
}

std::ostream& operator<<(
    std::ostream& os, const PublicObservationHistory& poh) {
  return os << poh.ToString();
}

}  // namespace open_spiel
