// Copyright 2019 DeepMind Technologies Limited
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

#include "open_spiel/algorithms/observation_history.h"

#include <string>

#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/abseil-cpp/absl/types/optional.h"
#include "open_spiel/observer.h"
#include "open_spiel/spiel_globals.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {

// -----------------------------------------------------------------------------
// ActionObservationHistory
// -----------------------------------------------------------------------------

// TODO(author13) Switch to the new Observation API
ActionObservationHistory::ActionObservationHistory(Player player,
                                                   const State& target)
    : player_(player) {
  SPIEL_CHECK_GE(player_, 0);
  SPIEL_CHECK_LT(player_, target.NumPlayers());
  SPIEL_CHECK_TRUE(target.GetGame()->GetType().provides_observation_string);

  const std::vector<State::PlayerAction>& history = target.FullHistory();
  history_.reserve(history.size());

  std::unique_ptr<State> state = target.GetGame()->NewInitialState();
  history_.push_back({absl::nullopt, state->ObservationString(player)});
  for (int i = 0; i < history.size(); i++) {
    const auto& [history_player, action] = history[i];
    const bool is_acting = state->CurrentPlayer() == player;
    state->ApplyAction(action);
    history_.push_back({
      is_acting ? action : static_cast<absl::optional<Action>>(absl::nullopt),
      state->ObservationString(player)
    });
  }
}

ActionObservationHistory::ActionObservationHistory(const State& target)
    : ActionObservationHistory(target.CurrentPlayer(), target) {}

ActionObservationHistory::ActionObservationHistory(
    Player player,
    std::vector<std::pair<absl::optional<Action>, std::string>> history)
    : player_(player), history_(std::move(history)) {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_FALSE(history_.empty());  // There is always an obs for root node.
  SPIEL_CHECK_EQ(history_[0].first, absl::nullopt);  // No action available.
}

int ActionObservationHistory::MoveNumber() const {
  SPIEL_CHECK_FALSE(history_.empty());
  SPIEL_CHECK_EQ(history_.at(0).first, absl::nullopt);
  return history_.size() - 1;
}

const std::string& ActionObservationHistory::ObservationAt(int time) const {
  return history_.at(time).second;
}

absl::optional<Action> ActionObservationHistory::ActionAt(int time) const {
  return history_.at(time).first;
}

bool ActionObservationHistory::CorrespondsTo(Player pl,
                                             const State& state) const {
  if (MoveNumber() != state.MoveNumber()) return false;
  bool equal = CheckStateCorrespondenceInSimulation(pl, state, MoveNumber());
  SPIEL_CHECK_TRUE(!equal || IsPrefixOf(pl, state));
  SPIEL_CHECK_TRUE(!equal || IsExtensionOf(pl, state));
  return equal;
}

bool ActionObservationHistory::CorrespondsTo(
    const ActionObservationHistory& other) const {
  bool equal = player_ == other.player_ && history_ == other.history_;
  SPIEL_CHECK_TRUE(!equal || IsPrefixOf(other));
  SPIEL_CHECK_TRUE(!equal || IsExtensionOf(other));
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

bool ActionObservationHistory::IsPrefixOf(Player pl, const State& state) const {
  const std::shared_ptr<const Game> game = state.GetGame();
  SPIEL_CHECK_TRUE(game->GetType().provides_observation_string);

  if (CorrespondsToInitialState()) return true;
  // Cannot be prefix if state is earlier.
  if (MoveNumber() > state.MoveNumber()) return false;

  return CheckStateCorrespondenceInSimulation(pl, state, MoveNumber());
}

bool ActionObservationHistory::IsExtensionOf(
    const ActionObservationHistory& other) const {
  return other.IsPrefixOf(*this);
}

bool ActionObservationHistory::IsExtensionOf(Player pl,
                                             const State& state) const {
  const std::shared_ptr<const Game> game = state.GetGame();
  SPIEL_CHECK_TRUE(game->GetType().provides_observation_string);

  if (state.IsInitialState()) return true;
  // Cannot be extension if state is later.
  if (state.MoveNumber() > MoveNumber()) return false;

  // Check the latest observation is identical -- most observations
  // will differ only in the last items.
  if (state.ObservationString(pl) != ObservationAt(state.MoveNumber()))
    return false;

  return CheckStateCorrespondenceInSimulation(pl, state, state.MoveNumber());
}

void ActionObservationHistory::Extend(const absl::optional<Action> action,
                                      const std::string& observation_string) {
  history_.push_back({action, observation_string});
}

void ActionObservationHistory::RemoveLast() {
  SPIEL_CHECK_GT(history_.size(), 0);
  history_.pop_back();
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
      if (history_[j].first != state_history[i].action) return false;
    } else {
      if (history_[j].first != absl::nullopt) return false;
    }

    simulation->ApplyAction(state_history[i].action);
    i++;

    if (history_[j].second != simulation->ObservationString(pl)) return false;
    j++;
  }
  return true;
}

std::string ActionObservationHistory::ToString() const {
  std::string s;
  for (int i = 0; i < history_.size(); i++) {
    const auto& action_observation = history_[i];
    if (i > 0) absl::StrAppend(&s, ", ");
    absl::StrAppend(&s, "(action=",
                    (action_observation.first == absl::nullopt
                         ? "None"
                         : std::to_string(*action_observation.first)),
                    ", observation=\"", action_observation.second, "\")");
  }
  return s;
}

// -----------------------------------------------------------------------------
// PublicObservationHistory
// -----------------------------------------------------------------------------

PublicObservationHistory::PublicObservationHistory(const State& target)
    : observer_(target.GetGame()->MakeObserver(
          IIGObservationType{/*public_info*/true,
                             /*perfect_recall*/false,
                             /*private_info*/PrivateInfoType::kNone},
          {})) {
  history_.reserve(target.FullHistory().size());

  std::unique_ptr<State> state = target.GetGame()->NewInitialState();
  // Use FullHistory even though we don't need the player -- prevent
  // doing a copy.
  for (const auto& [_, action] : target.FullHistory()) {
    history_.push_back(observer_->StringFrom(*state, kDefaultPlayerId));
    state->ApplyAction(action);
  }
  history_.push_back(observer_->StringFrom(*state, kDefaultPlayerId));
}

PublicObservationHistory::PublicObservationHistory(
    std::vector<std::string> history)
    : history_(std::move(history)) {
  SPIEL_CHECK_FALSE(history_.empty());
}

int PublicObservationHistory::MoveNumber() const {
  SPIEL_CHECK_FALSE(history_.empty());
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
  if (MoveNumber() != state.MoveNumber()) return false;
  bool equal = CheckStateCorrespondenceInSimulation(state, MoveNumber());
  SPIEL_CHECK_TRUE(!equal || IsPrefixOf(state));
  SPIEL_CHECK_TRUE(!equal || IsExtensionOf(state));
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
  if (CorrespondsToInitialState()) return true;
  // Cannot be prefix if state is earlier.
  if (state.MoveNumber() < MoveNumber()) return false;

  return CheckStateCorrespondenceInSimulation(state, MoveNumber());
}

bool PublicObservationHistory::IsExtensionOf(
    const PublicObservationHistory& other) const {
  return other.IsPrefixOf(*this);
}

bool PublicObservationHistory::IsExtensionOf(const State& state) const {
  if (state.MoveNumber() > MoveNumber()) return false;

  // Check the latest observation is identical -- most observations
  // will differ only in the last items.
  if (observer_->StringFrom(state, kDefaultPlayerId) !=
      ObservationAt(state.MoveNumber()))
    return false;

  return CheckStateCorrespondenceInSimulation(state, state.MoveNumber());
}

std::string PublicObservationHistory::ToString() const {
  return absl::StrJoin(history_, ", ");
}

void PublicObservationHistory::push_back(const std::string& observation) {
  SPIEL_CHECK_FALSE(observation.empty());
  history_.push_back(observation);
}

bool PublicObservationHistory::CheckStateCorrespondenceInSimulation(
    const State& state, int until_time) const {
  const std::vector<State::PlayerAction>& state_history = state.FullHistory();
  std::unique_ptr<State> simulation = state.GetGame()->NewInitialState();

  int i = 0;  // The index for state_history access.
  int j = 1;  // The index for history_ access.
  while (simulation->MoveNumber() < until_time) {
    SPIEL_CHECK_LT(i, state_history.size());
    SPIEL_CHECK_LT(j, history_.size());
    SPIEL_CHECK_FALSE(simulation->IsTerminal());

    simulation->ApplyAction(state_history[i].action);
    i++;

    if (history_.at(j) != observer_->StringFrom(*simulation, kDefaultPlayerId))
      return false;
    j++;
  }
  return true;
}

// -----------------------------------------------------------------------------
// Streaming.
// -----------------------------------------------------------------------------

std::ostream& operator<<(std::ostream& os,
                         const ActionObservationHistory& aoh) {
  return os << aoh.ToString();
}

std::ostream& operator<<(std::ostream& os,
                         const PublicObservationHistory& poh) {
  return os << poh.ToString();
}

}  // namespace open_spiel
