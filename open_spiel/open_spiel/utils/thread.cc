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

#include <thread>  // NOLINT

namespace open_spiel {

class Thread::ThreadImpl : public std::thread {
 public:
  using std::thread::thread;  // Inherit the constructors.
};

Thread::Thread(std::function<void()> fn) : thread_(new ThreadImpl(fn)) {}

// defaults required to be here for pimpl to work.
Thread::~Thread() = default;
Thread::Thread(Thread&& other) = default;
Thread& Thread::operator=(Thread&& other) = default;

void Thread::join() { thread_->join(); }

}  // namespace open_spiel
