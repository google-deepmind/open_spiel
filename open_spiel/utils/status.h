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

#ifndef OPEN_SPIEL_UTILS_STATUS_H_
#define OPEN_SPIEL_UTILS_STATUS_H_

#include <iostream>
#include <string>

namespace open_spiel {

enum class StatusValue {
  kOk = 0,
  kError = 1,
};

class Status {
 public:
  explicit Status() : status_value_(StatusValue::kOk) {}
  Status(StatusValue status_value, const std::string& message)
      : status_value_(status_value), message_(message) {}
  bool ok() const { return status_value_ == StatusValue::kOk; }
  std::string message() const { return message_; }
  std::string ToString() const;

 private:
  StatusValue status_value_;
  std::string message_;
};

template <typename T>
class StatusWithValue : public Status {
 public:
  explicit StatusWithValue() : Status() {}

  StatusWithValue(StatusValue status_value,
                  const std::string& message,
                  T value)
      : Status(status_value, message), value_(value) {}
  T value() const { return value_; }

 private:
  T value_;
};

Status OkStatus();
Status ErrorStatus(const std::string& message);

std::ostream& operator<<(std::ostream& os, const Status& status);

}  // namespace open_spiel

#endif  // OPEN_SPIEL_UTILS_INIT_H_
