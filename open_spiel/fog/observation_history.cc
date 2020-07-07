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

namespace open_spiel {

bool ActionOrObservation::operator==(const ActionOrObservation& other) const {
  if (tag != other.tag) return false;
  if (tag == Either::kAction) return action == other.action;
  if (tag == Either::kObservation) return observation == other.observation;
  SpielFatalError("Unknown tag.");
  return "This will never return.";
}

bool AOHistory::IsPrefix(const AOHistory& other) const {
  const auto& a = history_;
  const auto& b = other.history_;
  if (a.empty()) return true;
  if (b.empty()) return false;  // True only if a is empty, handled before.
  if (a.size() > b.size()) return false;
  if (a.size() == b.size()) return a == b;
  return std::equal(a.begin(), a.end(), b.begin());
}

bool POHistory::IsPrefix(const POHistory& other) const {
  const auto& a = history_;
  const auto& b = other.history_;
  if (a.empty()) return true;
  if (b.empty()) return false;  // True only if a is empty, handled before.
  if (a.size() > b.size()) return false;
  if (a.size() == b.size()) return a == b;
  return std::equal(a.begin(), a.end(), b.begin());
}

std::ostream& operator<<(std::ostream& os, const ActionOrObservation& aoo) {
  if (aoo.tag == ActionOrObservation::Either::kAction) {
    return os << "action='" << aoo.action << "'";
  } else if (aoo.tag == ActionOrObservation::Either::kObservation) {
    return os << "observation='" << aoo.observation << "'";
  }
  SpielFatalError("Unrecognized tag.");
  return;  // "This will never return.";
}

std::ostream& operator<<(std::ostream& os, const AOHistory& aoh) {
  return os << aoh.History();
}

std::ostream& operator<<(std::ostream& os, const POHistory& poh) {
  return os << poh.History();
}

}  // namespace open_spiel
