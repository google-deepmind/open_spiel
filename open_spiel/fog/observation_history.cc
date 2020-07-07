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

bool ActionOrObs::operator==(const ActionOrObs& other) const {
  if (tag != other.tag) return false;
  if (tag == Either::kAction) return action == other.action;
  if (tag == Either::kObservation) return observation == other.observation;
  SpielFatalError("Unknown tag.");
  return "This will never return.";
}

std::string ActionOrObs::ToString() const {
  if (tag == ActionOrObs::Either::kAction) {
    return absl::StrCat("action='", action, "'");
  }
  if (tag == ActionOrObs::Either::kObservation) {
    return absl::StrCat("observation='", observation, "'");
  }
  SpielFatalError("Unrecognized tag.");
  return;  // "This will never return.";
}

bool ActionObsHistory::IsPrefix(const ActionObsHistory& other) const {
  const auto& a = history_;
  const auto& b = other.history_;
  if (a.empty()) return true;
  if (b.empty()) return false;  // True only if a is empty, handled before.
  if (a.size() > b.size()) return false;
  if (a.size() == b.size()) return a == b;
  return std::equal(a.begin(), a.end(), b.begin());
}

std::string ActionObsHistory::ToString() const {
  return absl::StrJoin(history_, ", ", absl::StreamFormatter());
}

bool PubObsHistory::IsPrefix(const PubObsHistory& other) const {
  const auto& a = history_;
  const auto& b = other.history_;
  if (a.empty()) return true;
  if (b.empty()) return false;  // True only if a is empty, handled before.
  if (a.size() > b.size()) return false;
  if (a.size() == b.size()) return a == b;
  return std::equal(a.begin(), a.end(), b.begin());
}

std::string PubObsHistory::ToString() const {
  return absl::StrJoin(history_, ", ");
}

std::ostream& operator<<(std::ostream& os, const ActionOrObs& aoo) {
  return os << aoo.ToString();
}

std::ostream& operator<<(std::ostream& os, const ActionObsHistory& aoh) {
  return os << aoh.ToString();
}

std::ostream& operator<<(std::ostream& os, const PubObsHistory& poh) {
  return os << poh.ToString();
}

}  // namespace open_spiel
