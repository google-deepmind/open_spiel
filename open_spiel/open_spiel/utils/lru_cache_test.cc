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

#include "open_spiel/utils/lru_cache.h"

#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace {

void TestLRUCache() {
  LRUCache<int, std::string> cache(4);

  SPIEL_CHECK_EQ(cache.Size(), 0);

  LRUCacheInfo info = cache.Info();
  SPIEL_CHECK_EQ(info.hits, 0);
  SPIEL_CHECK_EQ(info.misses, 0);
  SPIEL_CHECK_EQ(info.size, 0);
  SPIEL_CHECK_EQ(info.max_size, 4);
  SPIEL_CHECK_EQ(info.Usage(), 0);
  SPIEL_CHECK_EQ(info.HitRate(), 0);

  SPIEL_CHECK_FALSE(cache.Get(1));

  cache.Set(13, "13");
  SPIEL_CHECK_EQ(cache.Size(), 1);

  SPIEL_CHECK_FALSE(cache.Get(1));

  {
    absl::optional<const std::string> v = cache.Get(13);
    SPIEL_CHECK_TRUE(v);
    SPIEL_CHECK_EQ(*v, "13");
  }

  cache.Set(14, "14");
  cache.Set(15, "15");
  cache.Set(16, "16");

  SPIEL_CHECK_EQ(cache.Size(), 4);

  cache.Set(17, "17");

  SPIEL_CHECK_EQ(cache.Size(), 4);

  SPIEL_CHECK_FALSE(cache.Get(13));  // evicted
  SPIEL_CHECK_TRUE(cache.Get(14));

  SPIEL_CHECK_EQ(cache.Size(), 4);

  cache.Set(18, "18");

  SPIEL_CHECK_FALSE(cache.Get(15));  // evicted
  SPIEL_CHECK_TRUE(cache.Get(14));  // older but more recently used

  info = cache.Info();
  SPIEL_CHECK_EQ(info.Usage(), 1);

  cache.Clear();

  SPIEL_CHECK_FALSE(cache.Get(18));  // evicted
}

}  // namespace
}  // namespace open_spiel

int main(int argc, char** argv) { open_spiel::TestLRUCache(); }
