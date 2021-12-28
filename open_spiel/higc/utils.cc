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

#include "open_spiel/higc/utils.h"

#include <thread>   // NOLINT (Used only externally.)

namespace open_spiel {
namespace higc {

void sleep_ms(int ms) {
  std::this_thread::sleep_for(std::chrono::milliseconds(ms));
}

int time_elapsed(
    const std::chrono::time_point<std::chrono::system_clock>& start) {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
             std::chrono::system_clock::now() - start)
      .count();
}

}  // namespace higc
}  // namespace open_spiel
