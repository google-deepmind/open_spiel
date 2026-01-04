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

#include "open_spiel/utils/data_logger.h"

#include "open_spiel/abseil-cpp/absl/strings/str_format.h"
#include "open_spiel/abseil-cpp/absl/strings/str_join.h"
#include "open_spiel/abseil-cpp/absl/time/clock.h"
#include "open_spiel/abseil-cpp/absl/time/time.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/utils/json.h"

namespace open_spiel {

DataLoggerJsonLines::DataLoggerJsonLines(const std::string& path,
                                         const std::string& name, bool flush,
                                         const std::string& mode,
                                         absl::Time start_time)
    : fd_(absl::StrFormat("%s/%s.jsonl", path, name), mode),
      flush_(flush),
      start_time_(start_time) {}

void DataLoggerJsonLines::Write(DataLogger::Record record) {
  static absl::TimeZone utc = absl::UTCTimeZone();
  absl::Time now = absl::Now();
  record.insert({
      {"time_str", absl::FormatTime("%Y-%m-%d %H:%M:%E6S %z", now, utc)},
      {"time_abs", absl::ToUnixMicros(now) / 1000000.},
      {"time_rel", absl::ToDoubleSeconds(now - start_time_)},
  });
  fd_.Write(json::ToString(record));
  fd_.Write("\n");
  if (flush_) {
    Flush();
  }
}

void DataLoggerJsonLines::Flush() { fd_.Flush(); }

DataLoggerJsonLines::~DataLoggerJsonLines() { Flush(); }

}  // namespace open_spiel
