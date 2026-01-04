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

# Lint as python3
"""Tests for open_spiel.python.algorithms.playthrough."""

from absl.testing import absltest
import numpy as np

from open_spiel.python.algorithms import generate_playthrough


class PlaythroughTest(absltest.TestCase):

  def test_runs(self):
    result = generate_playthrough.playthrough(
        "tic_tac_toe", action_sequence=[0, 1, 2, 3, 4, 5, 6, 7, 8])
    self.assertNotEmpty(result)

  def test_format_tensor_1d(self):
    lines = generate_playthrough._format_tensor(np.array((1, 0, 1, 1)), "x")
    self.assertEqual(lines, ["x: ◉◯◉◉"])

  def test_format_tensor_2d(self):
    lines = generate_playthrough._format_tensor(np.array(((1, 0), (1, 1))), "x")
    self.assertEqual(lines, [
        "x: ◉◯",
        "   ◉◉",
    ])

  def test_format_tensor_3d(self):
    lines = []
    tensor = np.array((
        ((1, 0), (1, 1)),
        ((0, 0), (1, 0)),
        ((0, 1), (1, 0)),
    ))
    lines = generate_playthrough._format_tensor(tensor, "x")
    self.assertEqual(lines, [
        "x:",
        "◉◯  ◯◯  ◯◉",
        "◉◉  ◉◯  ◉◯",
    ])

  def test_format_tensor_3d_linewrap(self):
    tensor = np.array((
        ((1, 0), (1, 1)),
        ((0, 0), (1, 0)),
        ((0, 1), (1, 0)),
    ))
    lines = generate_playthrough._format_tensor(tensor, "x", max_cols=9)
    self.assertEqual(lines, [
        "x:",
        "◉◯  ◯◯",
        "◉◉  ◉◯",
        "",
        "◯◉",
        "◉◯",
    ])


if __name__ == "__main__":
  absltest.main()
