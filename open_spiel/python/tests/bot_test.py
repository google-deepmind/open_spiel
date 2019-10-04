# Copyright 2019 DeepMind Technologies Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Test that Python and C++ bots can be called by a C++ algorithm."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest

import numpy as np

from open_spiel.python.bots import uniform_random
import pyspiel


class BotTest(absltest.TestCase):

  def test_python_and_cpp_bot(self):
    game = pyspiel.load_game("kuhn_poker")
    bots = [
        pyspiel.make_uniform_random_bot(game, 0, 1234),
        uniform_random.UniformRandomBot(game, 1, np.random.RandomState(4321)),
    ]
    results = np.array([
        pyspiel.evaluate_bots(game.new_initial_state(), bots, iteration)
        for iteration in range(10000)
    ])
    average_results = np.mean(results, axis=0)
    np.testing.assert_allclose(average_results, [0.125, -0.125], atol=0.1)


if __name__ == "__main__":
  absltest.main()
