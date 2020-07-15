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
  Action GetAction() const {
    SPIEL_CHECK_TRUE(tag == Either::kAction);
    return action;
  }
  const std::string& GetObservation() const {
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
  std::vector<ActionOrObs> history_;
  const Player player_;

 public:
  // Constructs an Action-Observation history for a given player at the target
  // state. This method can be called only if the game provides
  // ObservationString().
  //
  // Note that this method makes a traversal of the state's history
  // to collect player's observations and this can be expensive.
  ActionObservationHistory(Player player, const State& target);

  // Constructs an Action-Observation history for the current player
  // at the target state.
  ActionObservationHistory(const State& target);

  // Constructs an Action-Observation history "manually".
  ActionObservationHistory(
      Player player, std::vector<ActionOrObs> history);

  ActionObservationHistory(const ActionObservationHistory&) = default;
  ~ActionObservationHistory() = default;

  const std::vector<ActionOrObs>& History() const { return history_; }
  Player GetPlayer() const { return player_; }

  // Is the current history prefix of the other one?
  // Empty AO History is a prefix of all AO histories.
  bool IsPrefix(const ActionObservationHistory& other) const;

  // Does the Action-Observation history correspond to the initial state
  // (root node)?
  bool IsRoot() const {
    // We receive observations also in the initial state!
    return history_.size() == 1;
  }

  std::string ToString() const;

  bool operator==(const ActionObservationHistory& other) const {
    return player_ == other.player_ && history_ == other.history_;
  }

  // A number of helper methods for extending AO history.
 private:
  void push_back(const ActionOrObs& aoo);
  void push_back(const Action& action) { push_back(ActionOrObs(action)); }
  void push_back(const std::string& observation) {
    push_back(ActionOrObs(observation));
  }
};

// Public-observation histories partition the game tree according to public
// available information into a corresponding public tree. Public observation
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
  // strings.
  //
  // Note that this method makes a traversal of the current game trajectory
  // to collect public observations and this can be expensive.
  PublicObservationHistory(const State& target);

  // Construct Public-observation history "manually".
  PublicObservationHistory(std::vector<std::string> history);

  PublicObservationHistory(const PublicObservationHistory&) = default;
  ~PublicObservationHistory() = default;

  const std::vector<std::string>& History() const { return history_; }

  // Is the current history prefix of the other one?
  // Empty PO history is a prefix of all PO histories.
  bool IsPrefix(const PublicObservationHistory& other) const;

  // Does the Public-observation history correspond to the initial state
  // (root node)?
  bool IsRoot() const {
    SPIEL_CHECK_EQ(history_.at(0), kStartOfGamePublicObservation);
    return history_.size() == 1;
  }

  std::string ToString() const;

  bool operator==(const PublicObservationHistory& other) const {
    return history_ == other.history_;
  }

  bool operator==(const std::vector<std::string>& other) const {
    return history_ == other;
  }

 private:
  // A number of helper methods for extending the history.
  void reserve(size_t n) {  }
  void push_back(const std::string& observation) {
    SPIEL_CHECK_TRUE(!history_.empty() ||
                     observation == kStartOfGamePublicObservation);
    SPIEL_CHECK_NE(observation, kInvalidPublicObservation);
    history_.push_back(observation);
  }
};


std::ostream& operator<<(std::ostream& os, const ActionOrObs& aoo);
std::ostream& operator<<(std::ostream& os, const ActionObservationHistory& aoh);
std::ostream& operator<<(std::ostream& os, const PublicObservationHistory& poh);

}  // namespace open_spiel

#endif  // OPEN_SPIEL_FOG_OBSERVATION_HISTORY_H_
