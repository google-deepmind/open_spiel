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
"""Tests for Python Crowd Modelling game."""

from absl.testing import absltest
import numpy as np
from open_spiel.python.mfg.games import crowd_modelling
import pyspiel

MFG_STR_CONST = "_a"


class MFGCrowdModellingGameTest(absltest.TestCase):

  def test_load(self):
    game = pyspiel.load_game("python_mfg_crowd_modelling")
    game.new_initial_state()

  def test_create(self):
    """Checks we can create the game and clone states."""
    game = crowd_modelling.MFGCrowdModellingGame()
    self.assertEqual(game.size, crowd_modelling._SIZE)
    self.assertEqual(game.horizon, crowd_modelling._HORIZON)
    self.assertEqual(game.get_type().dynamics,
                     pyspiel.GameType.Dynamics.MEAN_FIELD)
    print("Num distinct actions:", game.num_distinct_actions())
    state = game.new_initial_state()
    clone = state.clone()
    print("Initial state:", state)
    print("Cloned initial state:", clone)

  def test_create_with_params(self):
    game = pyspiel.load_game("python_mfg_crowd_modelling(horizon=100,size=20)")
    self.assertEqual(game.size, 20)
    self.assertEqual(game.horizon, 100)

  def test_random_game(self):
    """Tests basic API functions."""
    horizon = 20
    size = 50
    game = crowd_modelling.MFGCrowdModellingGame(params={
        "horizon": horizon,
        "size": size
    })
    pyspiel.random_sim_test(
        game, num_sims=10, serialize=False, verbose=True)

  def test_reward(self):
    game = crowd_modelling.MFGCrowdModellingGame()
    state = game.new_initial_state()
    self.assertEqual(state.current_player(), pyspiel.PlayerId.CHANCE)
    state.apply_action(game.size // 2)
    self.assertEqual(state.current_player(), 0)
    # This expected reward assumes that the game is initialized with
    # uniform state distribution.
    self.assertAlmostEqual(state.rewards()[0], 1. + np.log(game.size))
    self.assertAlmostEqual(state.returns()[0], 1. + np.log(game.size))
    state.apply_action(1)
    self.assertEqual(state.current_player(), pyspiel.PlayerId.CHANCE)
    self.assertAlmostEqual(state.returns()[0], 1. + np.log(game.size))

  def test_distribution(self):
    """Checks that distribution-related functions work."""
    game = crowd_modelling.MFGCrowdModellingGame()
    state = game.new_initial_state()
    self.assertEqual(state.current_player(), pyspiel.PlayerId.CHANCE)
    state.apply_action(game.size // 2)
    self.assertEqual(state.current_player(), 0)
    # This expected reward assumes that the game is initialized with
    # uniform state distribution.
    self.assertAlmostEqual(state.rewards()[0], 1. + np.log(game.size))
    state.apply_action(crowd_modelling.MFGCrowdModellingState._NEUTRAL_ACTION)
    # Chance node.
    self.assertEqual(state.current_player(), pyspiel.PlayerId.CHANCE)
    state.apply_action(crowd_modelling.MFGCrowdModellingState._NEUTRAL_ACTION)
    self.assertEqual(state.distribution_support(), [
        "(0, 1)_a", "(1, 1)_a", "(2, 1)_a", "(3, 1)_a", "(4, 1)_a", "(5, 1)_a",
        "(6, 1)_a", "(7, 1)_a", "(8, 1)_a", "(9, 1)_a"
    ])
    new_distrib = [0.01] * 9 + [1. - 0.01 * 9]
    state.update_distribution(new_distrib)
    self.assertAlmostEqual(state._distribution, new_distrib)

    # Check that the distribution is taken into account for the reward
    # computation.
    self.assertAlmostEqual(state.rewards()[0], 1. - np.log(0.01))

  def test_compare_py_cpp(self):
    """Compares py and cpp implementations of this game."""
    py_game = pyspiel.load_game("python_mfg_crowd_modelling")
    cpp_game = pyspiel.load_game("mfg_crowd_modelling")
    np.random.seed(7)
    py_state = py_game.new_initial_state()
    cpp_state = cpp_game.new_initial_state()
    t = 0
    while not cpp_state.is_terminal():
      self.assertFalse(py_state.is_terminal())
      self.assertEqual(str(cpp_state), str(py_state))
      self.assertAlmostEqual(cpp_state.returns()[0], py_state.returns()[0])
      if cpp_state.current_player() == pyspiel.PlayerId.CHANCE:
        actions, probs = zip(*cpp_state.chance_outcomes())
        action = np.random.choice(actions, p=probs)
        self.assertEqual(
            cpp_state.action_to_string(action),
            py_state.action_to_string(action))
        cpp_state.apply_action(action)
        py_state.apply_action(action)
      elif cpp_state.current_player() == pyspiel.PlayerId.MEAN_FIELD:
        num_cpp_states = len(cpp_state.distribution_support())
        distribution = [1 / num_cpp_states] * num_cpp_states
        cpp_state.update_distribution(distribution)
        py_state.update_distribution(distribution)
      else:
        self.assertEqual(cpp_state.current_player(), 0)
        legal_actions = cpp_state.legal_actions()
        action = np.random.choice(legal_actions)
        self.assertEqual(
            cpp_state.action_to_string(action),
            py_state.action_to_string(action))
        cpp_state.apply_action(action)
        py_state.apply_action(action)
        t += 1


if __name__ == "__main__":
  absltest.main()
