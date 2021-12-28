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

"""Test Python bindings for game transforms."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest

import pyspiel


class RepeatedGameTest(absltest.TestCase):

  def test_create_repeated_game(self):
    """Test both create_repeated_game function signatures."""
    repeated_game = pyspiel.create_repeated_game("matrix_rps",
                                                 {"num_repetitions": 10})
    state = repeated_game.new_initial_state()
    for _ in range(10):
      state.apply_actions([0, 0])
    assert state.is_terminal()

    stage_game = pyspiel.load_game("matrix_mp")
    repeated_game = pyspiel.create_repeated_game(stage_game,
                                                 {"num_repetitions": 5})
    state = repeated_game.new_initial_state()
    for _ in range(5):
      state.apply_actions([0, 0])
    assert state.is_terminal()


if __name__ == "__main__":
  absltest.main()
