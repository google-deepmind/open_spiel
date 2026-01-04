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

#ifndef OPEN_SPIEL_UTILS_THREAD_H_
#define OPEN_SPIEL_UTILS_THREAD_H_

#include <atomic>
#include <functional>
#include <memory>

namespace open_spiel {

// A simple thread class similar to std::thread, but only accepting a function
// without args. Wrap your args in a lambda if necessary. Needed for
// compatibility with Google's libraries.
class Thread {
 public:
  explicit Thread(std::function<void()> fn);
  ~Thread();

  // Thread is move only.
  Thread(Thread&& other);
  Thread& operator=(Thread&& other);
  Thread(const Thread&) = delete;
  Thread& operator=(const Thread&) = delete;

  void join();

 private:
  class ThreadImpl;
  std::unique_ptr<ThreadImpl> thread_;
};

// A token for whether a thread has been requested to stop.
class StopToken {
 public:
  StopToken() : token_(false) {}
  void Stop() { token_ = true; }
  bool StopRequested() const { return token_; }
 private:
  std::atomic<bool> token_;
};

}  // namespace open_spiel

#endif  // OPEN_SPIEL_UTILS_THREAD_H_
