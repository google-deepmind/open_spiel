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

#include "open_spiel/utils/status.h"

#include <iostream>
#include <string>

namespace open_spiel {

Status OkStatus() {
  return Status(StatusValue::kOk, "");
}

Status ErrorStatus(const std::string& message) {
  return Status(StatusValue::kError, message);
}

std::string Status::ToString() const {
  switch (status_value_) {
    case StatusValue::kOk:
      return "OkStatus";
    case StatusValue::kError:
      return "ErrorStatus: " + message_;
    default:
      return "UnknownStatus";
  }
}


std::ostream& operator<<(std::ostream& os, const Status& status) {
  os << status.ToString();
  return os;
}

}  // namespace open_spiel
