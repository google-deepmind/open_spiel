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

"""Tests for open_spiel.python.utils.replay_buffer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest

from open_spiel.python.utils.replay_buffer import ReplayBuffer


class ReplayBufferTest(absltest.TestCase):

  def test_replay_buffer_add(self):
    # pylint: disable=g-generic-assert
    replay_buffer = ReplayBuffer(replay_buffer_capacity=10)
    self.assertEqual(len(replay_buffer), 0)
    replay_buffer.add("entry1")
    self.assertEqual(len(replay_buffer), 1)
    replay_buffer.add("entry2")
    self.assertEqual(len(replay_buffer), 2)

    self.assertIn("entry1", replay_buffer)
    self.assertIn("entry2", replay_buffer)

  def test_replay_buffer_max_capacity(self):
    # pylint: disable=g-generic-assert
    replay_buffer = ReplayBuffer(replay_buffer_capacity=2)
    replay_buffer.add("entry1")
    replay_buffer.add("entry2")
    replay_buffer.add("entry3")
    self.assertEqual(len(replay_buffer), 2)

    self.assertIn("entry2", replay_buffer)
    self.assertIn("entry3", replay_buffer)

  def test_replay_buffer_sample(self):
    replay_buffer = ReplayBuffer(replay_buffer_capacity=3)
    replay_buffer.add("entry1")
    replay_buffer.add("entry2")
    replay_buffer.add("entry3")

    samples = replay_buffer.sample(3)

    self.assertIn("entry1", samples)
    self.assertIn("entry2", samples)
    self.assertIn("entry3", samples)

  def test_replay_buffer_reset(self):
    replay_buffer = ReplayBuffer(replay_buffer_capacity=3)
    replay_buffer.add("entry1")
    replay_buffer.add("entry2")

    replay_buffer.reset()
    self.assertEmpty(replay_buffer)


if __name__ == "__main__":
  absltest.main()
