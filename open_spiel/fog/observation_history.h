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

#include "open_spiel/fog/fog_constants.h"
#include "open_spiel/spiel_constants.h"

namespace open_spiel {

// Store either action or observation.
// We cannot use C++ unions for this, because observations are strings.
struct ActionOrObservation {
    enum class Either { kAction, kObservation };
    const Either tag;
    const Action action = kInvalidAction;
    const std::string observation = kInvalidObservation;

    explicit ActionOrObservation(Action act)
        : tag(Either::kAction), action(act) {
      SPIEL_CHECK_NE(action, kInvalidAction);
    }

    explicit ActionOrObservation(std::string obs)
        : tag(Either::kObservation), observation(std::move(obs)) {
      SPIEL_CHECK_NE(observation, kInvalidObservation);
    }

    bool IsAction() const { return tag == Either::kAction; }
    bool IsObservation() const { return tag == Either::kObservation; }
    bool operator==(const ActionOrObservation& other) const;
};

// Action-Observation histories partition the game tree in the same way
// as information states, but they contain more structured information.
// Some algorithms use this structured information for targeted traversal
// of the imperfect information tree.
class AOHistory {
 private:
  std::vector<ActionOrObservation> history_;

 public:
  AOHistory() : history_({ActionOrObservation(kStartOfGameObservation)}) {}

  AOHistory(std::initializer_list<ActionOrObservation> history)
      : history_(std::move(history)) {
    SPIEL_CHECK_FALSE(history_.empty());
    SPIEL_CHECK_EQ(history_.at(0),
                   ActionOrObservation(kStartOfGameObservation));
  }

  AOHistory(const AOHistory&) = default;
  ~AOHistory() = default;

  const std::vector<ActionOrObservation>& History() const { return history_; }

  // Is the current history prefix of the other one?
  // Empty AOHistory is a prefix of all AOHistories.
  bool IsPrefix(const AOHistory& other) const;

  // Does the Action-Observation history correspond to the initial state
  // (root node)?
  bool IsRoot() const {
    SPIEL_CHECK_EQ(
        history_.at(0), ActionOrObservation(kStartOfGameObservation));
    return history_.size() == 1;
  }

  // A number of helper methods, so we don't need to access history directly.
  void reserve(size_t n) {
    history_.reserve(n);
  }
  void push_back(const Action& action) {
    history_.push_back(ActionOrObservation(action));
  }
  void push_back(const std::string& observation) {
    history_.push_back(ActionOrObservation(observation));
  }
  void pop_back() {
    // Do not pop the kStartOfGameObservation.
    SPIEL_CHECK_GE(history_.size(), 2);
    history_.pop_back();
  }

  const ActionOrObservation& back() const { history_.back(); }

  bool operator==(const AOHistory& other) const {
    return history_ == other.history_;
  };
};

// Public-observation histories partition the game tree in the same way
// as information states, but they contain more structured information.
// Some algorithms use this structured information for targeted traversal
// of the imperfect information tree.
class POHistory {
 private:
  std::vector<std::string> history_;

 public:
  POHistory() = default;

  POHistory(std::initializer_list<std::string> history)
      : history_(std::move(history)) {
    SPIEL_CHECK_FALSE(history_.empty());
    SPIEL_CHECK_EQ(history_.at(0), kStartOfGameObservation);
  }

  POHistory(const POHistory&) = default;
  ~POHistory() = default;

  const std::vector<std::string>& History() const { return history_; }

  // Is the current history prefix of the other one?
  // Empty POHistory is a prefix of all POHistories.
  bool IsPrefix(const POHistory& other) const;

  // Does the Public-observation history correspond to the initial state
  // (root node)?
  bool IsRoot() const {
    SPIEL_CHECK_EQ(history_.at(0), kStartOfGameObservation);
    return history_.size() == 1;
  }

  // A number of helper methods, so we don't need to access history directly.
  void reserve(size_t n) {
    history_.reserve(n);
  }
  void push_back(const std::string& observation) {
    SPIEL_CHECK_TRUE(
        !history_.empty() || observation == kStartOfGameObservation);
    history_.push_back(observation);
  }
  void pop_back() {
    // Do not pop the kStartOfGameObservation.
    SPIEL_CHECK_GE(history_.size(), 2);
    history_.pop_back();
  }

  const std::string& back() const { history_.back(); }

  bool operator==(const POHistory& other) const {
    return history_ == other.history_;
  };
};

std::ostream& operator<<(std::ostream& os, const ActionOrObservation& aoo);
std::ostream& operator<<(std::ostream& os, const AOHistory& aoh);
std::ostream& operator<<(std::ostream& os, const POHistory& aoh);

}  // namespace open_spiel

#endif  // OPEN_SPIEL_OBSERVATION_HISTORY_H_
