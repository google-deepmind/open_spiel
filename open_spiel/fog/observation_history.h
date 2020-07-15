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

#ifndef OPEN_SPIEL_FOG_OBSERVATION_HISTORY_H_
#define OPEN_SPIEL_FOG_OBSERVATION_HISTORY_H_

#include <string>
#include <utility>
#include <vector>
#include <memory>

#include "open_spiel/fog/fog_constants.h"
#include "open_spiel/spiel_globals.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/spiel.h"

namespace open_spiel {

// Store either action or observation.
// We cannot use C++ unions for this, because observations are strings.
class ActionOrObs {
 private:
  enum class Either { kAction, kObservation };
  const Either tag;
  const Action action = kInvalidAction;
  const std::string observation;

 public:
  explicit ActionOrObs(Action act)
      : tag(Either::kAction), action(act) {
    SPIEL_CHECK_NE(action, kInvalidAction);
  }
  explicit ActionOrObs(std::string obs)
      : tag(Either::kObservation), observation(std::move(obs)) {}

  bool IsAction() const { return tag == Either::kAction; }
  bool IsObservation() const { return tag == Either::kObservation; }
  Action Action() const {
    SPIEL_CHECK_TRUE(tag == Either::kAction);
    return action;
  }
  const std::string& Observation() const {
    SPIEL_CHECK_TRUE(tag == Either::kObservation);
    return observation;
  }
  std::string ToString() const;
  bool operator==(const ActionOrObs& other) const;
};

// Action-Observation histories partition the game tree in the same way
// as information states, but they contain more structured information.
// Some algorithms use this structured information for targeted traversal
// of the imperfect information tree.
//
// Note that in the FOG paper, Action-Observation history $s$ for player $i$
// at world history $h$ is defined precisely as
//
//    s_i(h) := (O_i^0, a_i^0, O_i^1, a_i^1, ... O_i^{t-1}, a_i^{t-1}, O_i^t)
//
// and this can be interpreted for simultaneous-move games as
//
//    Initial Observation  + List of Pair((Action, Observation))
//
// However in OpenSpiel the player is not always acting, as in sequential games,
// but we'd like to support those as well. So we make a compromise to just have
// AOH as a vector of action OR observation, and we make sure that:
//
// 1) The first element is an observation.
// 2) We don't have two consecutive actions (there should be an observation
//    between them).
class ActionObservationHistory {
 private:
  // Player to which this Action-Observation history belongs.
  const Player player_;

  // A redundant field capturing the current time:
  // this can be computed also as accumulation over the history_ field.
  // We keep this so we can retrieve clock time fast.
  int elapsed_time_;

  // Actual Action-Observation history.
  std::vector<ActionOrObs> history_;

 public:
  // Constructs an Action-Observation history for a given player at the target
  // state. This method can be called only if the game provides an
  // implementation of ObservationString().
  //
  // Note that this constructor makes a traversal of the state's history
  // to collect player's observations and this can be expensive.
  ActionObservationHistory(Player player, const State& target);

  // Constructs an Action-Observation history for the current player
  // at the target state.
  ActionObservationHistory(const State& target);

  // Constructs an Action-Observation history "manually" from history vector.
  ActionObservationHistory(
      Player player, std::vector<ActionOrObs> history);

  ActionObservationHistory(const ActionObservationHistory&) = default;
  ~ActionObservationHistory() = default;

  const std::vector<ActionOrObs>& History() const { return history_; }
  Player GetPlayer() const { return player_; }

  // Gets the current time on the clock - this allows to relate the "depth"
  // of Action-Observation history to the "depth" of State, as clock time
  // corresponds exactly to State::MoveNumber().
  int ClockTime() const { return elapsed_time_; }

  // Returns the player's observation (i.e. public+private observation) 
  // at the given time. Root node has time 0.
  const std::string& ObservationAt(int time) const;

  // Returns the action at the given time.
  // If player was not acting at requested time, returns an kInvalidAction.
  Action ActionAt(int time) const;

  // Does the Action-Observation history correspond to the initial state?
  bool CorrespondsToInitialState() const { return ClockTime() == 0; };

  // Does the Action-Observation history correspond to the other
  // Action-Observation history? This is just like an equality operator.
  bool CorrespondsTo(const ActionObservationHistory& other) const;

  // Does the Action-Observation history correspond to the requested state?
  //
  // In other words, if we constructed Action-Observation history for the state, 
  // would that correspond to this Action-Observation history? 
  // 
  // As in the following:
  //
  //   CorrespondsTo(pl, state) == CorrespondsTo(
  //      ActionObservationHistory(pl, state))
  //
  // This method is provided so that you do not need to construct 
  // Action-Observation History explicitly and is more efficient. 
  // This is like an equality operator (for State).
  bool CorrespondsTo(Player pl, const State& state) const;

  // Is the current Action-Observation history prefix (or equal) of the other?
  bool IsPrefixOf(const ActionObservationHistory& other) const;

  // Is the current Action-Observation history prefix (or equal) of the 
  // Action-Observation history that we could construct from the State?
  bool IsPrefixOf(Player pl, const State& state) const;

  // Is the current Action-Observation history extension (or equal) 
  // of the other one?
  bool IsExtensionOf(const ActionObservationHistory& other) const;

  // Is the current Action-Observation history extension (or equal) 
  // of the Action-Observation history that we could construct from the State?
  bool IsExtensionOf(Player pl, const State& state) const;

  std::string ToString() const;

  bool operator==(const ActionObservationHistory& other) const {
    return CorrespondsTo(other);
  }

 private:
  void push_back(const ActionOrObs& aoo);
  void push_back(const Action& action) { push_back(ActionOrObs(action)); }
  void push_back(const std::string& observation) {
    push_back(ActionOrObs(observation));
  }
  bool CheckStateCorrespondenceInSimulation(
      Player pl, const State& state, int until_time) const;
};

// Public-observation histories partition the game tree according to available
// public information into a corresponding public tree. Public observation
// history identifies the current public state (a node in the public tree),
// and is useful for integration with public state API -- you can construct
// a PublicState by using the public observation history.
//
// Some algorithms use this structured information for targeted traversal
// of the (im)perfect information tree.
class PublicObservationHistory {
 private:
  std::vector<std::string> history_;

 public:
  // Construct a history of public observations.
  // This method can be called only if the game provides factored observations
  // strings, mainly State::PublicObservationString() -- private observations
  // are not used.
  //
  // Note that this constructor makes a traversal of the state's history
  // to collect public  observations and this can be expensive.
  PublicObservationHistory(const State& target);

  // Constructs Public-Observation history "manually".
  PublicObservationHistory(std::vector<std::string> history);

  PublicObservationHistory(const PublicObservationHistory&) = default;
  ~PublicObservationHistory() = default;

  const std::vector<std::string>& History() const { return history_; }

  // Gets the current time on the clock - this allows to relate the "depth"
  // of Public-Observation history to the "depth" of State, as clock time
  // corresponds exactly to State::MoveNumber().
  int ClockTime() const;

  // Returns the public observation at the given time. Root node has time 0.
  const std::string& ObservationAt(int time) const;

  // Does the Public-Observation history correspond to the initial state?
  bool CorrespondsToInitialState() const { return ClockTime() == 0; }

  // Does the Public-Observation history correspond to the other
  // Public-Observation history? This is just like an equality operator.
  bool CorrespondsTo(const PublicObservationHistory& other) const;

  // Does the Public-Observation history correspond to the requested state?
  //
  // In other words, if we constructed Public-Observation history for the state,
  // would that correspond to this Public-Observation history?
  // As in the following:
  //
  //   CorrespondsTo(state) == CorrespondsTo(PublicObservationHistory(state))
  //
  // This method is provided so that you do not need to construct
  // Public-Observation history explicitly and is more efficient.
  // This is like an equality operator.
  bool CorrespondsTo(const State& state) const;

  // Is the current Public-Observation history prefix (or equal) of the other?
  bool IsPrefixOf(const PublicObservationHistory& other) const;

  // Is the current Public-Observation history prefix (or equal) of the
  // Public-Observation history that we could construct from the State?
  bool IsPrefixOf(const State& state) const;

  // Is the current Public-Observation history extension (or equal)
  // of the other one?
  bool IsExtensionOf(const PublicObservationHistory& other) const;

  // Is the current Public-Observation history extension (or equal)
  // of the Public-Observation history that we could construct from the State?
  bool IsExtensionOf(const State& state) const;

  std::string ToString() const;

  bool operator==(const PublicObservationHistory& other) const {
    return CorrespondsTo(other);
  }

 private:
  void push_back(const std::string& observation);
  bool CheckStateCorrespondenceInSimulation(
      const State& state, int until_time) const;
};

std::ostream& operator<<(std::ostream& os, const ActionOrObs& aoo);
std::ostream& operator<<(std::ostream& os, const ActionObservationHistory& aoh);
std::ostream& operator<<(std::ostream& os, const PublicObservationHistory& poh);

}  // namespace open_spiel

#endif  // OPEN_SPIEL_FOG_OBSERVATION_HISTORY_H_
