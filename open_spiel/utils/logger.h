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

#ifndef OPEN_SPIEL_UTILS_LOGGER_H_
#define OPEN_SPIEL_UTILS_LOGGER_H_

#include <cstdio>
#include <string>

#include "open_spiel/abseil-cpp/absl/strings/str_format.h"
#include "open_spiel/abseil-cpp/absl/time/clock.h"
#include "open_spiel/abseil-cpp/absl/time/time.h"
#include "open_spiel/utils/file.h"

namespace open_spiel {

class Logger {
 public:
  virtual ~Logger() = default;
  virtual void Print(const std::string& str) = 0;

  // A specialization of Print that passes everything through StrFormat first.
  template <typename Arg1, typename... Args>
  void Print(const absl::FormatSpec<Arg1, Args...>& format, const Arg1& arg1,
             const Args&... args) {
    Print(absl::StrFormat(format, arg1, args...));
  }
};


// A logger to print stuff to a file.
class FileLogger : public Logger {
 public:
  FileLogger(const std::string& path, const std::string& name,
             const std::string& mode = "w")
      : fd_(absl::StrFormat("%s/log-%s.txt", path, name), mode),
        tz_(absl::LocalTimeZone()) {
    Print("%s started", name);
  }

  using Logger::Print;
  void Print(const std::string& str) override {
    std::string time =
        absl::FormatTime("%Y-%m-%d %H:%M:%E3S", absl::Now(), tz_);
    fd_.Write(absl::StrFormat("[%s] %s\n", time, str));
    fd_.Flush();
  }

  ~FileLogger() override { Print("Closing the log."); }

 private:
  open_spiel::file::File fd_;
  absl::TimeZone tz_;
};


class NoopLogger : public Logger {
 public:
  using Logger::Print;
  void Print(const std::string& str) override {}
};

}  // namespace open_spiel

#endif  // OPEN_SPIEL_UTILS_LOGGER_H_
