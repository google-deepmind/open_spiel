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

#include "open_spiel/utils/thread.h"

#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace {

void TestThread() {
  int value = 1;
  Thread thread([&](){ value = 2; });
  thread.join();
  SPIEL_CHECK_EQ(value, 2);
}

void TestThreadMove() {
  int value = 1;
  Thread thread([&](){ value = 2; });
  Thread thread2(std::move(thread));
  thread2.join();
  SPIEL_CHECK_EQ(value, 2);
}

void TestThreadMoveAssign() {
  int value = 1;
  Thread thread([&](){ value = 2; });
  Thread thread2 = std::move(thread);
  thread2.join();
  SPIEL_CHECK_EQ(value, 2);
}

}  // namespace
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::TestThread();
  open_spiel::TestThreadMove();
  open_spiel::TestThreadMoveAssign();
}
