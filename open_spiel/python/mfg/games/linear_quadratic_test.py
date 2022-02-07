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
from open_spiel.python.mfg.games import linear_quadratic
import pyspiel

MFG_STR_CONST = "_a"


class MFGLinearQuadraticGameTest(absltest.TestCase):

  def test_load(self):
    game = pyspiel.load_game("mean_field_lin_quad")
    game.new_initial_state()

  def test_create(self):
    """Checks we can create the game and clone states."""
    game = linear_quadratic.MFGLinearQuadraticGame()
    self.assertEqual(game.size, linear_quadratic._SIZE)
    self.assertEqual(game.horizon, linear_quadratic._HORIZON)
    self.assertEqual(game.get_type().dynamics,
                     pyspiel.GameType.Dynamics.MEAN_FIELD)
    print("Num distinct actions:", game.num_distinct_actions())
    state = game.new_initial_state()
    clone = state.clone()
    print("Initial state:", state)
    print("Cloned initial state:", clone)

  def test_create_with_params(self):
    game = pyspiel.load_game("mean_field_lin_quad(horizon=30,size=100)")
    self.assertEqual(game.size, 100)
    self.assertEqual(game.horizon, 30)

  def check_cloning(self, state):
    cloned = state.clone()
    self.assertEqual(str(cloned), str(state))
    self.assertEqual(cloned._distribution, state._distribution)
    self.assertEqual(cloned._returns(), state._returns())
    self.assertEqual(cloned.current_player(), state.current_player())
    self.assertEqual(cloned.size, state.size)
    self.assertEqual(cloned.horizon, state.horizon)
    self.assertEqual(cloned._last_action, state._last_action)

  def test_random_game(self):
    """Tests basic API functions."""
    np.random.seed(7)
    horizon = 30
    size = 100
    game = linear_quadratic.MFGLinearQuadraticGame(params={
        "horizon": horizon,
        "size": size
    })
    state = game.new_initial_state()
    t = 0
    while not state.is_terminal():
      if state.current_player() == pyspiel.PlayerId.CHANCE:
        actions, probs = zip(*state.chance_outcomes())
        action = np.random.choice(actions, p=probs)
        self.check_cloning(state)
        self.assertEqual(len(state.legal_actions()),
                         len(state.chance_outcomes()))
        state.apply_action(action)
      elif state.current_player() == pyspiel.PlayerId.MEAN_FIELD:
        self.assertEqual(state.legal_actions(), [])
        self.check_cloning(state)
        num_states = len(state.distribution_support())
        state.update_distribution([1 / num_states] * num_states)
      else:
        self.assertEqual(state.current_player(), 0)
        self.check_cloning(state)
        state.observation_string()
        state.information_state_string()
        legal_actions = state.legal_actions()
        action = np.random.choice(legal_actions)
        state.apply_action(action)
        t += 1

    self.assertEqual(t, horizon)


if __name__ == "__main__":
  absltest.main()
