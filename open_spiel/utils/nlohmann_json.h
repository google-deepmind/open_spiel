// Copyright 2019 DeepMind Technologies Limited
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

#ifndef OPEN_SPIEL_UTILS_NLOHMANN_JSON_H_
#define OPEN_SPIEL_UTILS_NLOHMANN_JSON_H_

#include <optional>
#include "open_spiel/json/include/nlohmann/json.hpp"

namespace nlohmann {

template <typename T>
struct adl_serializer<std::optional<T>> {
  static void to_json(json& j, const std::optional<T>& opt) {
    if (opt.has_value()) {
      j = *opt;
    } else {
      j = nullptr;
    }
  }

  static void from_json(const json& j, std::optional<T>& opt) {
    if (j.is_null()) {
      opt = std::nullopt;
    } else {
      opt = j.get<T>();
    }
  }
};

}  // namespace nlohmann

#endif  // OPEN_SPIEL_UTILS_NLOHMANN_JSON_H_
