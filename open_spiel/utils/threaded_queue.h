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

#ifndef OPEN_SPIEL_UTILS_THREADED_QUEUE_H_
#define OPEN_SPIEL_UTILS_THREADED_QUEUE_H_

#include <queue>

#include "open_spiel/abseil-cpp/absl/synchronization/mutex.h"
#include "open_spiel/abseil-cpp/absl/time/clock.h"
#include "open_spiel/abseil-cpp/absl/time/time.h"
#include "open_spiel/abseil-cpp/absl/types/optional.h"

namespace open_spiel {

// A threadsafe-queue.
template <class T>
class ThreadedQueue {
 public:
  explicit ThreadedQueue(int max_size) : max_size_(max_size) {}

  // Add an element to the queue.
  bool Push(const T& value) { return Push(value, absl::InfiniteDuration()); }
  bool Push(const T& value, absl::Duration wait) {
    return Push(value, absl::Now() + wait);
  }
  bool Push(const T& value, absl::Time deadline) {
    absl::MutexLock lock(&m_);
    if (block_new_values_) {
      return false;
    }
    while (q_.size() >= max_size_) {
      if (absl::Now() > deadline || block_new_values_) {
        return false;
      }
      cv_.WaitWithDeadline(&m_, deadline);
    }
    q_.push(value);
    cv_.Signal();
    return true;
  }

  absl::optional<T> Pop() { return Pop(absl::InfiniteDuration()); }
  absl::optional<T> Pop(absl::Duration wait) { return Pop(absl::Now() + wait); }
  absl::optional<T> Pop(absl::Time deadline) {
    absl::MutexLock lock(&m_);
    while (q_.empty()) {
      if (absl::Now() > deadline || block_new_values_) {
        return absl::nullopt;
      }
      cv_.WaitWithDeadline(&m_, deadline);
    }
    T val = q_.front();
    q_.pop();
    cv_.Signal();
    return val;
  }

  bool Empty() {
    absl::MutexLock lock(&m_);
    return q_.empty();
  }

  void Clear() {
    absl::MutexLock lock(&m_);
    while (!q_.empty()) {
      q_.pop();
    }
  }

  int Size() {
    absl::MutexLock lock(&m_);
    return q_.size();
  }

  // Causes pushing new values to fail. Useful for shutting down the queue.
  void BlockNewValues() {
    absl::MutexLock lock(&m_);
    block_new_values_ = true;
    cv_.SignalAll();
  }

 private:
  bool block_new_values_ = false;
  int max_size_;
  std::queue<T> q_;
  absl::Mutex m_;
  absl::CondVar cv_;
};

}  // namespace open_spiel

#endif  // OPEN_SPIEL_UTILS_THREADED_QUEUE_H_
