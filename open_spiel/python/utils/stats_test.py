# Copyright 2026 DeepMind Technologies Limited
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

"""Tests for statistics utilities."""

from absl.testing import absltest
from absl.testing import parameterized

from open_spiel.python.utils import stats


class SlidingWindowAccumulatorTest(parameterized.TestCase):

  @parameterized.parameters(0, -1)
  def test_rejects_non_positive_window_size(self, window_size):
    with self.assertRaisesRegex(ValueError, "max_window_size must be positive"):
      stats.SlidingWindowAccumulator(window_size)

  def test_empty(self):
    accumulator = stats.SlidingWindowAccumulator(3)
    self.assertEqual(accumulator.mean(), 0)

  def test_first_eviction(self):
    accumulator = stats.SlidingWindowAccumulator(3)
    for value in [1, 2, 3, 4]:
      accumulator.add(value)
    self.assertEqual(accumulator.mean(), 3)

  @parameterized.parameters(1, 2, 3, 5)
  def test_mean_matches_latest_values(self, window_size):
    accumulator = stats.SlidingWindowAccumulator(window_size)
    values = [2.5, -1, 7, 0, -3.5, 8, 1, 4, -2, 6, 9, 3, -4, 5, 10, 0]
    for index, value in enumerate(values):
      accumulator.add(value)
      window = values[max(0, index + 1 - window_size) : index + 1]
      self.assertAlmostEqual(accumulator.mean(), sum(window) / len(window))


if __name__ == "__main__":
  absltest.main()
