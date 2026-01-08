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

#ifndef OPEN_SPIEL_UTILS_SERIALIZATION_H_
#define OPEN_SPIEL_UTILS_SERIALIZATION_H_

#include <iomanip>
#include <sstream>

#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/abseil-cpp/absl/strings/str_format.h"

namespace open_spiel {

// Formats doubles with human-readable strings with a specified number of
// decimal places, i.e. results in lossy serialization.
struct SimpleDoubleFormatter {
  SimpleDoubleFormatter(int precision = 6) : precision(precision) {}

  void operator()(std::string* out, const double& d) const {
    std::stringstream stream;
    stream << std::fixed << std::setprecision(precision) << d;
    absl::StrAppend(out, stream.str());
  }

  const int precision;
};

// Formats doubles with non-portable bitwise representation hex strings, i.e.
// results in lossless serialization.
struct HexDoubleFormatter {
  void operator()(std::string* out, const double& d) const {
    absl::StrAppend(out, absl::StrFormat("%a", d));
  }
};

}  // namespace open_spiel

#endif  // OPEN_SPIEL_UTILS_SERIALIZATION_H_
