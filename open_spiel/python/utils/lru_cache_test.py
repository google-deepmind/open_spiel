# Copyright 2019 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Tests for open_spiel.python.utils.lru_cache."""

from absl.testing import absltest

from open_spiel.python.utils import lru_cache


class LruCacheTest(absltest.TestCase):

  def test_lru_cache(self):
    cache = lru_cache.LRUCache(4)

    self.assertEmpty(cache)

    info = cache.info()
    self.assertEqual(info.hits, 0)
    self.assertEqual(info.misses, 0)
    self.assertEqual(info.size, 0)
    self.assertEqual(info.max_size, 4)
    self.assertEqual(info.usage, 0)
    self.assertEqual(info.hit_rate, 0)

    self.assertIsNone(cache.get(1))

    cache.set(13, "13")
    self.assertLen(cache, 1)

    self.assertIsNone(cache.get(1))

    self.assertEqual(cache.get(13), "13")

    cache.set(14, "14")
    cache.set(15, "15")
    cache.set(16, "16")

    self.assertLen(cache, 4)

    cache.set(17, "17")

    self.assertLen(cache, 4)

    self.assertIsNone(cache.get(13))  # evicted
    self.assertTrue(cache.get(14))

    self.assertLen(cache, 4)

    cache.set(18, "18")

    self.assertIsNone(cache.get(15))  # evicted
    self.assertTrue(cache.get(14))  # older but more recently used

    info = cache.info()
    self.assertEqual(info.usage, 1)

    cache.clear()

    self.assertIsNone(cache.get(18))  # evicted

    self.assertEqual(cache.make(19, lambda: "19"), "19")
    self.assertEqual(cache.get(19), "19")
    self.assertEqual(cache.make(19, lambda: "20"), "19")

if __name__ == "__main__":
  absltest.main()
