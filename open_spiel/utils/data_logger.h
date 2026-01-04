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

#ifndef OPEN_SPIEL_UTILS_DATA_LOGGER_H_
#define OPEN_SPIEL_UTILS_DATA_LOGGER_H_

#include <string>
#include <utility>
#include <vector>

#include "open_spiel/abseil-cpp/absl/time/time.h"
#include "open_spiel/utils/file.h"
#include "open_spiel/utils/json.h"

namespace open_spiel {

class DataLogger {
 public:
  using Record = json::Object;

  virtual ~DataLogger() = default;
  virtual void Write(Record record) = 0;
  virtual void Flush() {}
};

// Writes to a file in http://jsonlines.org/ format.
class DataLoggerJsonLines : public DataLogger {
 public:
  explicit DataLoggerJsonLines(const std::string& path, const std::string& name,
                               bool flush = false,
                               const std::string& mode = "w",
                               absl::Time start_time = absl::Now());
  ~DataLoggerJsonLines() override;

  // The json lines logger is move only.
  DataLoggerJsonLines(DataLoggerJsonLines&& other) = default;
  DataLoggerJsonLines& operator=(DataLoggerJsonLines&& other) = default;
  DataLoggerJsonLines(const DataLoggerJsonLines&) = delete;
  DataLoggerJsonLines& operator=(const DataLoggerJsonLines&) = delete;

  void Write(Record record) override;
  void Flush() override;

 private:
  file::File fd_;
  bool flush_;
  absl::Time start_time_;
};

class DataLoggerNoop : public DataLogger {
 public:
  ~DataLoggerNoop() override = default;
  void Write(Record record) override {}
};

}  // namespace open_spiel

#endif  // OPEN_SPIEL_UTILS_DATA_LOGGER_H_
