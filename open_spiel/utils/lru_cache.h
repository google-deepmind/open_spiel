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

#ifndef OPEN_SPIEL_UTILS_LRU_CACHE_H_
#define OPEN_SPIEL_UTILS_LRU_CACHE_H_

#include <list>

#include "open_spiel/abseil-cpp/absl/container/flat_hash_map.h"
#include "open_spiel/abseil-cpp/absl/synchronization/mutex.h"

namespace open_spiel {

struct LRUCacheInfo {
  int64_t hits = 0;
  int64_t misses = 0;
  int size = 0;
  int max_size = 0;

  double Usage() const {
    return max_size == 0 ? 0 : static_cast<double>(size) / max_size;
  }
  int64_t Total() const { return hits + misses; }
  double HitRate() const {
    return Total() == 0 ? 0 : static_cast<double>(hits) / Total();
  }

  void operator+=(const LRUCacheInfo& o) {
    hits += o.hits;
    misses += o.misses;
    size += o.size;
    max_size += o.max_size;
  }
};

template <typename K, typename V>
class LRUCache {  // Least Recently Used Cache.
  // TODO(author7): Consider the performance implications here. Some ideas:
  // - Shard the cache to avoid lock contention. Can be done by the user.
  // - Use shared pointers to avoid copying data out, and shorten the lock.
  // - Use two generations to avoid order updates on hot items. The mature
  //   generation wouldn't be ordered or evicted so can use a reader/writer lock
  // - Use atomics for hits/misses to shorten lock times.
  // - Embed the list directly into the map value to avoid extra indirection.
 public:
  explicit LRUCache(int max_size) : hits_(0), misses_(0) {
    SetMaxSize(max_size);
  }

  // Move only, not copyable.
  LRUCache(LRUCache&& other) = default;
  LRUCache& operator=(LRUCache&& other) = default;
  LRUCache(const LRUCache&) = delete;
  LRUCache& operator=(const LRUCache&) = delete;

  void SetMaxSize(int max_size) { max_size_ = std::max(max_size, 4); }

  int Size() {
    absl::MutexLock lock(&m_);
    return map_.size();
  }

  void Clear() {
    absl::MutexLock lock(&m_);
    order_.clear();
    map_.clear();
    hits_ = 0;
    misses_ = 0;
  }

  void Set(const K& key, const V& value) {
    absl::MutexLock lock(&m_);
    auto pos = map_.find(key);
    if (pos == map_.end()) {           // Not found, add it.
      if (map_.size() >= max_size_) {  // Make space if needed.
        map_.erase(order_.back());
        order_.pop_back();
      }
      order_.push_front(key);
      map_[key] = Entry{value, order_.begin()};
    } else {  // Found, move it to the front.
      order_.erase(pos->second.order_iterator);
      order_.push_front(key);
      pos->second.order_iterator = order_.begin();
    }
  }

  absl::optional<const V> Get(const K& key) {
    absl::MutexLock lock(&m_);
    auto pos = map_.find(key);
    if (pos == map_.end()) {  // Not found.
      misses_ += 1;
      return absl::nullopt;
    } else {  // Found, move it to the front, and return the value.
      hits_ += 1;
      order_.erase(pos->second.order_iterator);
      order_.push_front(key);
      pos->second.order_iterator = order_.begin();
      return pos->second.value;
    }
  }

  LRUCacheInfo Info() {
    absl::MutexLock lock(&m_);
    return LRUCacheInfo{hits_, misses_, static_cast<int>(map_.size()),
                        max_size_};
  }

 private:
  struct Entry {
    V value;
    typename std::list<K>::iterator order_iterator;
  };

  int64_t hits_;
  int64_t misses_;
  int max_size_;
  std::list<K> order_;
  absl::flat_hash_map<K, Entry> map_;
  absl::Mutex m_;
};

}  // namespace open_spiel

#endif  // OPEN_SPIEL_UTILS_LRU_CACHE_H_
