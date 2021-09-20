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

# Lint as python3
"""Tests for Python mean field routing game."""

from absl.testing import absltest

from open_spiel.python import policy
from open_spiel.python.mfg.algorithms import distribution
from open_spiel.python.mfg.games import dynamic_routing
import pyspiel

# pylint: disable=g-bad-todo


class MeanFieldRoutingGameTest(absltest.TestCase):
  """Checks we can create the game and clone states."""

  def test_load(self):
    """Test load and game creation."""
    game = pyspiel.load_game("python_mfg_dynamic_routing")
    game.new_initial_state()

  def test_create(self):
    """Checks we can create the game and clone states."""
    game = dynamic_routing.MeanFieldRoutingGame()
    self.assertEqual(game.get_type().dynamics,
                     pyspiel.GameType.Dynamics.MEAN_FIELD)
    state = game.new_initial_state()
    state.clone()

  # TODO(Theo): is this used? If not, should we remove it?
  # def check_cloning(self, state):
  #   """Test cloning."""
  #   cloned = state.clone()
  #   self.assertEqual(str(cloned), str(state))
  #   self.assertEqual(cloned._distribution, state._distribution)
  #   self.assertEqual(cloned.current_player(), state.current_player())

  def test_trajectory_under_uniform_distribution(self):
    """Test random simulation."""
    game = dynamic_routing.MeanFieldRoutingGame()
    pyspiel.random_sim_test(game, num_sims=10, serialize=False, verbose=True)

  def test_evolving_trajectory_with_uniform_policy(self):
    """Test evolving distribution."""
    game = dynamic_routing.MeanFieldRoutingGame()
    distribution.DistributionPolicy(game, policy.UniformRandomPolicy(game))

  # TODO: Add test with bigger network. See dynamic routing game class.
  # TODO: add test for FP iteration.
  # TODO: test departure time enabled
  # TODO: test evolution of the game as expected (test value of the state).


if __name__ == "__main__":
  absltest.main()
