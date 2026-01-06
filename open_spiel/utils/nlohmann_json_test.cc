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

#include "open_spiel/utils/nlohmann_json.h"  // IWYU pragma: keep
#include <optional>

#include "open_spiel/json/include/nlohmann/json.hpp"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace {

void TestStdOptionalSerialization() {
  nlohmann::json j;
  std::optional<int> opt_int;

  // Test serialization of nullopt.
  opt_int = std::nullopt;
  j = opt_int;
  SPIEL_CHECK_EQ(j.dump(), "null");

  // Test deserialization of nullopt.
  opt_int = 42;
  j = nullptr;
  opt_int = j.get<std::optional<int>>();
  SPIEL_CHECK_FALSE(opt_int.has_value());

  // Test serialization of a value.
  opt_int = 123;
  j = opt_int;
  SPIEL_CHECK_EQ(j.dump(), "123");

  // Test deserialization of a value.
  opt_int = std::nullopt;
  j = 123;
  opt_int = j.get<std::optional<int>>();
  SPIEL_CHECK_TRUE(opt_int.has_value());
  SPIEL_CHECK_EQ(opt_int.value(), 123);
}

}  // namespace
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::TestStdOptionalSerialization();
}
