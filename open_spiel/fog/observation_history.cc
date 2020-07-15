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

// -----------------------------------------------------------------------------
// ActionOrObs
// -----------------------------------------------------------------------------

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

// -----------------------------------------------------------------------------
// ActionObservationHistory
// -----------------------------------------------------------------------------

ActionObservationHistory::ActionObservationHistory(
    Player player, const State& target) : player_(player), elapsed_time_(0) {
  SPIEL_CHECK_GE(player_, 0);
  SPIEL_CHECK_LT(player_, target.NumPlayers());
  SPIEL_CHECK_TRUE(target.GetGame()->GetType().provides_observation_string);
  const std::vector<State::PlayerAction>& history = target.FullHistory();
  history_.reserve(2 * history.size());

  std::unique_ptr<State> state = target.GetGame()->NewInitialState();
  for (int i = 0; i < history.size(); i++) {
    const auto&[history_player, action] = history[i];
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
    : player_(player), elapsed_time_(0), history_(std::move(history)) {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_FALSE(history_.empty());  // There is always an obs for root node.
  SPIEL_CHECK_TRUE(history_[0].IsObservation());

  for (int i = 1; i < history_.size(); i++) {
    if (history_[i].IsObservation()) elapsed_time_++;
    SPIEL_CHECK_IMPLY(history_[i].IsAction(), history_[i - 1].IsObservation());
  }
}

const std::string& ActionObservationHistory::ObservationAt(int time) const {
  SPIEL_CHECK_GE(time, 0);
  SPIEL_CHECK_LE(time, elapsed_time_);
  // We don't know how many actions happened before the requested time,
  // so we need to traverse the vector.
  int t = 0;
  for (const ActionOrObs& aoo : history_) {
    if (!aoo.IsObservation()) continue;
    if (t == time) return aoo.Observation();
    ++t;
  }

  // Some time inconsistency?
  SpielFatalError("This should never happen");
  return "";
}

Action ActionObservationHistory::ActionAt(int time) const {
  SPIEL_CHECK_GE(time, 0);
  SPIEL_CHECK_LE(time, elapsed_time_);
  // We don't know how many actions happened before the requested time,
  // so we need to traverse the vector.
  int t = 0;
  for (const ActionOrObs& aoo : history_) {
    // Player was not acting at requested time.
    if (t > time) return kInvalidAction;

    if (t == time && aoo.IsAction()) return aoo.Action();
    if (aoo.IsObservation()) ++t;
  }

  // Some time inconsistency?
  SpielFatalError("This should never happen");
  return 0;
}

bool ActionObservationHistory::CorrespondsTo(
    Player pl, const State& state) const {
  if (ClockTime() != state.MoveNumber()) return false;
  bool equal = CheckStateCorrespondenceInSimulation(pl, state, ClockTime());
  SPIEL_CHECK_IMPLY(equal, IsPrefixOf(pl, state));
  SPIEL_CHECK_IMPLY(equal, IsExtensionOf(pl, state));
  return equal;
}

bool ActionObservationHistory::CorrespondsTo(
    const ActionObservationHistory& other) const {
  bool equal = player_ == other.player_ && history_ == other.history_;
  SPIEL_CHECK_IMPLY(equal, elapsed_time_ == other.elapsed_time_);
  SPIEL_CHECK_IMPLY(equal, IsPrefixOf(other));
  SPIEL_CHECK_IMPLY(equal, IsExtensionOf(other));
  return equal;
}

bool ActionObservationHistory::IsPrefixOf(
    const ActionObservationHistory& other) const {
  if (player_ != other.player_) return false;

  if (CorrespondsToInitialState()) return true;
  if (other.CorrespondsToInitialState()) return false;

  const auto& a = history_;
  const auto& b = other.history_;
  if (a.size() > b.size()) return false;
  if (a.size() == b.size()) return a == b;
  return std::equal(a.begin(), a.end(), b.begin());
}

bool ActionObservationHistory::IsPrefixOf(
    Player pl, const State& state) const {
  const std::shared_ptr<const Game> game = state.GetGame();
  SPIEL_CHECK_TRUE(game->GetType().provides_observation_string);

  if (CorrespondsToInitialState()) return true;
  // Cannot be prefix if state is earlier.
  if (ClockTime() > state.MoveNumber()) return false;

  return CheckStateCorrespondenceInSimulation(pl, state, ClockTime());
}

bool ActionObservationHistory::IsExtensionOf(
    const ActionObservationHistory& other) const {
  return other.IsPrefixOf(*this);
}

bool ActionObservationHistory::IsExtensionOf(
    Player pl, const State& state) const {
  const std::shared_ptr<const Game> game = state.GetGame();
  SPIEL_CHECK_TRUE(game->GetType().provides_observation_string);

  if (state.IsInitialState()) return true;
  // Cannot be extension if state is later.
  if (state.MoveNumber() > ClockTime()) return false;

  // Check the latest observation is identical -- most observations
  // will differ only in the last items.
  if (state.ObservationString(pl) != ObservationAt(state.MoveNumber()))
    return false;

  return CheckStateCorrespondenceInSimulation(pl, state, state.MoveNumber());
}

bool ActionObservationHistory::CheckStateCorrespondenceInSimulation(
    Player pl, const State& state, int until_time) const {
  const std::vector<State::PlayerAction>& state_history = state.FullHistory();
  std::unique_ptr<State> simulation = state.GetGame()->NewInitialState();

  int i = 0;  // The index for state_history access.
  int j = 1;  // The index for history_ access.
  while (simulation->MoveNumber() < until_time) {
    SPIEL_CHECK_LT(i, state_history.size());
    SPIEL_CHECK_LT(j, history_.size());
    SPIEL_CHECK_FALSE(simulation->IsTerminal());

    if (simulation->CurrentPlayer() == pl) {
      if (!history_[j].IsAction()) return false;
      if (history_[j].Action() != state_history[i].action) return false;
      j++;
    }

    simulation->ApplyAction(state_history[i].action);
    i++;

    if (!history_[j].IsObservation()) return false;
    if (history_[j].Observation() != simulation->ObservationString(pl))
      return false;
    j++;
  }
  return true;
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

  // Increment time for non-initial observations.
  if (!history_.empty() && aoo.IsObservation()) ++elapsed_time_;
  history_.push_back(aoo);
}

std::string ActionObservationHistory::ToString() const {
  return absl::StrJoin(history_, ", ", absl::StreamFormatter());
}

// -----------------------------------------------------------------------------
// PublicObservationHistory
// -----------------------------------------------------------------------------

PublicObservationHistory::PublicObservationHistory(const State& target) {
  SPIEL_CHECK_TRUE(
      target.GetGame()->GetType().provides_factored_observation_string);
  history_.reserve(target.FullHistory().size());

  std::unique_ptr<State> state = target.GetGame()->NewInitialState();
  // Use FullHistory even though we don't need the player -- prevent
  // doing a copy.
  for (const auto&[player, action] : target.FullHistory()) {
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

int PublicObservationHistory::ClockTime() const {
  SPIEL_CHECK_FALSE(history_.empty());
  SPIEL_CHECK_EQ(history_.at(0), kStartOfGamePublicObservation);
  return history_.size() - 1;
}

const std::string& PublicObservationHistory::ObservationAt(int time) const {
  return history_.at(time);
}

bool PublicObservationHistory::CorrespondsTo(
    const PublicObservationHistory& other) const {
  return history_ == other.history_;
}

bool PublicObservationHistory::CorrespondsTo(const State& state) const {
  if (ClockTime() != state.MoveNumber()) return false;
  bool equal = CheckStateCorrespondenceInSimulation(state, ClockTime());
  SPIEL_CHECK_IMPLY(equal, IsPrefixOf(state));
  SPIEL_CHECK_IMPLY(equal, IsExtensionOf(state));
  return equal;
}

bool PublicObservationHistory::IsPrefixOf(
    const PublicObservationHistory& other) const {
  if (CorrespondsToInitialState()) return true;
  if (other.CorrespondsToInitialState()) return false;

  const auto& a = history_;
  const auto& b = other.history_;
  if (a.size() > b.size()) return false;
  if (a.size() == b.size()) return a == b;
  return std::equal(a.begin(), a.end(), b.begin());
}

bool PublicObservationHistory::IsPrefixOf(const State& state) const {
  const std::shared_ptr<const Game> game = state.GetGame();
  SPIEL_CHECK_TRUE(game->GetType().provides_factored_observation_string);

  if (CorrespondsToInitialState()) return true;
  // Cannot be prefix if state is earlier.
  if (state.MoveNumber() < ClockTime()) return false;

  return CheckStateCorrespondenceInSimulation(state, ClockTime());
}

bool PublicObservationHistory::IsExtensionOf(
    const PublicObservationHistory& other) const {
  return other.IsPrefixOf(*this);
}

bool PublicObservationHistory::IsExtensionOf(const State& state) const {
  const std::shared_ptr<const Game> game = state.GetGame();
  SPIEL_CHECK_TRUE(game->GetType().provides_factored_observation_string);

  if (state.MoveNumber() > ClockTime()) return false;

  // Check the latest observation is identical -- most observations
  // will differ only in the last items.
  if (state.PublicObservationString() != ObservationAt(state.MoveNumber()))
    return false;

  return CheckStateCorrespondenceInSimulation(state, state.MoveNumber());
}

std::string PublicObservationHistory::ToString() const {
  return absl::StrJoin(history_, ", ");
}

void PublicObservationHistory::push_back(const std::string& observation) {
  SPIEL_CHECK_IMPLY(
      history_.empty(), observation == kStartOfGamePublicObservation);
  SPIEL_CHECK_NE(observation, kInvalidPublicObservation);
  history_.push_back(observation);
}

bool PublicObservationHistory::CheckStateCorrespondenceInSimulation(
    const State& state, int until_time) const {
  const std::vector<State::PlayerAction>& state_history = state.FullHistory();
  std::unique_ptr<State> simulation = state.GetGame()->NewInitialState();
  SPIEL_CHECK_EQ(simulation->PublicObservationString(),
                 kStartOfGamePublicObservation);

  int i = 0;  // The index for state_history access.
  int j = 1;  // The index for history_ access.
  while (simulation->MoveNumber() < until_time) {
    SPIEL_CHECK_LT(i, state_history.size());
    SPIEL_CHECK_LT(j, history_.size());
    SPIEL_CHECK_FALSE(simulation->IsTerminal());

    simulation->ApplyAction(state_history[i].action);
    i++;

    if (history_.at(j) != simulation->PublicObservationString()) return false;
    j++;
  }
  return true;
}

// -----------------------------------------------------------------------------
// Streaming.
// -----------------------------------------------------------------------------

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
