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

#ifndef OPEN_SPIEL_OBSERVATION_HISTORY_H_
#define OPEN_SPIEL_OBSERVATION_HISTORY_H_

#include <utility>
#include <vector>
#include <string>

#include "open_spiel/fog/fog_constants.h"
#include "open_spiel/spiel_constants.h"

namespace open_spiel {

// Store either action or observation.
// We cannot use C++ unions for this, because observations are strings.
class ActionOrObs {
 private:
  enum class Either { kAction, kObservation };
  const Either tag;
  const Action action = kInvalidAction;
  const std::string observation = kInvalidObservation;
 public:
  explicit ActionOrObs(Action act)
      : tag(Either::kAction), action(act) {
    SPIEL_CHECK_NE(action, kInvalidAction);
  }
  explicit ActionOrObs(std::string obs)
      : tag(Either::kObservation), observation(std::move(obs)) {
    SPIEL_CHECK_NE(observation, kInvalidObservation);
  }

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
class ActionObsHistory {
 private:
  std::vector<ActionOrObs> history_;

 public:
  ActionObsHistory() = default;

  ActionObsHistory(std::vector<ActionOrObs> history)
      : history_(std::move(history)) {
    SPIEL_CHECK_FALSE(history_.empty());
    SPIEL_CHECK_EQ(
        history_.at(0), ActionOrObs(kStartOfGameObservation));
  }

  ActionObsHistory(const ActionObsHistory&) = default;
  ~ActionObsHistory() = default;

  const std::vector<ActionOrObs>& History() const { return history_; }

  // Is the current history prefix of the other one?
  // Empty AO History is a prefix of all AO histories.
  bool IsPrefix(const ActionObsHistory& other) const;

  // Does the Action-Observation history correspond to the initial state
  // (root node)?
  bool IsRoot() const {
    SPIEL_CHECK_EQ(
        history_.at(0), ActionOrObs(kStartOfGameObservation));
    return history_.size() == 1;
  }

  std::string ToString() const;

  bool operator==(const ActionObsHistory& other) const {
    return history_ == other.history_;
  }

  // A number of helper methods, so we don't access history directly.
  void reserve(size_t n) {
    history_.reserve(n);
  }
  void push_back(const ActionOrObs& aoo) {
    history_.push_back(aoo);
  }
  void push_back(const Action& action) {
    history_.push_back(ActionOrObs(action));
  }
  void push_back(const std::string& observation) {
    history_.push_back(ActionOrObs(observation));
  }
};

// Public-observation histories partition the game tree in the same way
// as information states, but they contain more structured information.
// Some algorithms use this structured information for targeted traversal
// of the imperfect information tree.
class PubObsHistory {
 private:
  std::vector<std::string> history_;

 public:
  PubObsHistory() = default;

  PubObsHistory(std::vector<std::string> history)
      : history_(std::move(history)) {
    SPIEL_CHECK_FALSE(history_.empty());
    SPIEL_CHECK_EQ(history_.at(0), kStartOfGameObservation);
  }

  PubObsHistory(const PubObsHistory&) = default;
  ~PubObsHistory() = default;

  const std::vector<std::string>& History() const { return history_; }

  // Is the current history prefix of the other one?
  // Empty PO history is a prefix of all PO histories.
  bool IsPrefix(const PubObsHistory& other) const;

  // Does the Public-observation history correspond to the initial state
  // (root node)?
  bool IsRoot() const {
    SPIEL_CHECK_EQ(history_.at(0), kStartOfGameObservation);
    return history_.size() == 1;
  }

  std::string ToString() const;

  bool operator==(const PubObsHistory& other) const {
    return history_ == other.history_;
  }

  bool operator==(const std::vector<std::string>& other) const {
    if (other.empty() || other[0] != kStartOfGameObservation) return false;
    return history_ == other;
  }

  // A number of helper methods, so we don't access history directly.
  void reserve(size_t n) {
    history_.reserve(n);
  }
  void push_back(const std::string& observation) {
    SPIEL_CHECK_TRUE(
        !history_.empty() || observation == kStartOfGameObservation);
    history_.push_back(observation);
  }
};

std::ostream& operator<<(std::ostream& os, const ActionOrObs& aoo);
std::ostream& operator<<(std::ostream& os, const ActionObsHistory& aoh);
std::ostream& operator<<(std::ostream& os, const PubObsHistory& aoh);

}  // namespace open_spiel

#endif  // OPEN_SPIEL_OBSERVATION_HISTORY_H_
