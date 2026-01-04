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

"""Tests for nash conv."""

from absl.testing import absltest

from open_spiel.python import policy
from open_spiel.python.mfg.algorithms import nash_conv
from open_spiel.python.mfg.games import crowd_modelling
import pyspiel


class BestResponseTest(absltest.TestCase):

  def test_python_game(self):
    """Checks if the NashConv is consistent through time."""
    game = crowd_modelling.MFGCrowdModellingGame()
    uniform_policy = policy.UniformRandomPolicy(game)
    nash_conv_fp = nash_conv.NashConv(game, uniform_policy)

    self.assertAlmostEqual(nash_conv_fp.nash_conv(), 2.8135365543870385)

  def test_cpp_game(self):
    """Checks if the NashConv is consistent through time."""
    game = pyspiel.load_game("mfg_crowd_modelling")
    uniform_policy = policy.UniformRandomPolicy(game)
    nash_conv_fp = nash_conv.NashConv(game, uniform_policy)

    self.assertAlmostEqual(nash_conv_fp.nash_conv(), 2.8135365543870385)


if __name__ == "__main__":
  absltest.main()
