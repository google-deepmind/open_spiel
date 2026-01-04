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

#include "open_spiel/utils/threaded_queue.h"

#include "open_spiel/abseil-cpp/absl/time/clock.h"
#include "open_spiel/abseil-cpp/absl/time/time.h"
#include "open_spiel/abseil-cpp/absl/types/optional.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace {

void TestThreadedQueue() {
  ThreadedQueue<int> q(4);

  auto CheckPopEq = [&q](int expected) {
    absl::optional<int> v = q.Pop();
    SPIEL_CHECK_TRUE(v);
    SPIEL_CHECK_EQ(*v, expected);
  };

  SPIEL_CHECK_TRUE(q.Empty());
  SPIEL_CHECK_EQ(q.Size(), 0);

  SPIEL_CHECK_FALSE(q.Pop(absl::Milliseconds(1)));
  SPIEL_CHECK_FALSE(q.Pop(absl::Now() + absl::Milliseconds(1)));

  SPIEL_CHECK_TRUE(q.Push(10, absl::Now() + absl::Milliseconds(1)));
  SPIEL_CHECK_FALSE(q.Empty());
  SPIEL_CHECK_EQ(q.Size(), 1);

  CheckPopEq(10);

  SPIEL_CHECK_TRUE(q.Push(11));
  SPIEL_CHECK_TRUE(q.Push(12));
  SPIEL_CHECK_EQ(q.Size(), 2);
  SPIEL_CHECK_TRUE(q.Push(13));
  SPIEL_CHECK_TRUE(q.Push(14));
  SPIEL_CHECK_EQ(q.Size(), 4);
  SPIEL_CHECK_FALSE(q.Push(15, absl::Milliseconds(1)));

  CheckPopEq(11);

  SPIEL_CHECK_TRUE(q.Push(16, absl::Milliseconds(1)));

  CheckPopEq(12);
  CheckPopEq(13);
  CheckPopEq(14);
  CheckPopEq(16);
  SPIEL_CHECK_EQ(q.Size(), 0);

  SPIEL_CHECK_TRUE(q.Push(17));
  SPIEL_CHECK_TRUE(q.Push(18));
  SPIEL_CHECK_EQ(q.Size(), 2);

  q.Clear();

  SPIEL_CHECK_TRUE(q.Empty());
  SPIEL_CHECK_EQ(q.Size(), 0);

  SPIEL_CHECK_TRUE(q.Push(19));
  SPIEL_CHECK_TRUE(q.Push(20));

  q.BlockNewValues();

  SPIEL_CHECK_EQ(q.Size(), 2);
  SPIEL_CHECK_FALSE(q.Push(21));
  SPIEL_CHECK_EQ(q.Size(), 2);
  CheckPopEq(19);
  CheckPopEq(20);
  SPIEL_CHECK_FALSE(q.Pop());
}

}  // namespace
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::TestThreadedQueue();
}
