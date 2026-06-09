# Copyright 2024 DeepMind Technologies Limited
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

"""Tests for Python Pursuit-Evasion."""

from absl.testing import absltest
import numpy as np

from open_spiel.python.games import pursuit_evasion
import pyspiel


def _play_game(game, evader_bot=None):
  """Play a full game, returning (state, steps).

  If evader_bot is None a uniform-random evader is used.
  """
  if evader_bot is None:
    evader_bot = pursuit_evasion.RandomEvaderBot(rng=np.random.RandomState())
  state = game.new_initial_state()
  steps = 0
  while not state.is_terminal():
    current = state.current_player()
    if current == 0:
      legal = state.legal_actions(0)
      action = np.random.choice(legal)
    else:
      action = evader_bot.step(state)
    state.apply_action(action)
    steps += 1
  return state, steps


class PursuitEvasionTest(absltest.TestCase):

  def test_can_create_game_and_state(self):
    game = pursuit_evasion.PursuitEvasionGame()
    state = game.new_initial_state()
    self.assertIn("Pursuer:", str(state))
    self.assertIn("Evader:", str(state))

  def test_game_from_cc(self):
    game = pyspiel.load_game("python_pursuit_evasion")
    for _ in range(10):
      _play_game(game)

  def test_random_simulation(self):
    game = pursuit_evasion.PursuitEvasionGame()
    state, steps = _play_game(game)
    self.assertLessEqual(steps, game.max_steps * 2)
    returns = state.returns()
    self.assertLen(returns, 2)
    self.assertIn(returns[0], [1.0, -1.0])
    self.assertEqual(returns[0], -returns[1])

  def test_two_player_returns(self):
    game = pursuit_evasion.PursuitEvasionGame()
    state = game.new_initial_state()
    state.apply_action(0)
    self.assertEqual(state.current_player(), 1)
    r = state.returns()
    self.assertLen(r, 2)
    self.assertEqual(r[0], -r[1])

  def test_terminal_conditions(self):
    game = pursuit_evasion.PursuitEvasionGame(params={"max_steps": 5})
    state, _ = _play_game(game)
    self.assertTrue(state.is_terminal())
    self.assertIn(state.pursuer_reward, [1.0, -1.0])

  def test_pursuer_wins_on_capture(self):
    game = pursuit_evasion.PursuitEvasionGame(
        params={"capture_radius": 100.0, "max_steps": 10})
    state, _ = _play_game(game)
    self.assertTrue(state.is_terminal())
    self.assertEqual(state.pursuer_reward, 1.0)

  def test_evader_wins_on_timeout(self):
    game = pursuit_evasion.PursuitEvasionGame(
        params={"capture_radius": 0.001, "max_steps": 3, "space_size": 100.0})
    state, _ = _play_game(game)
    self.assertTrue(state.is_terminal())
    self.assertEqual(state.pursuer_reward, -1.0)

  def test_all_evader_bots_run(self):
    bots = [
        pursuit_evasion.RandomEvaderBot(rng=np.random.RandomState(42)),
        pursuit_evasion.ConstantVelocityEvaderBot(),
        pursuit_evasion.ZigzagEvaderBot(),
        pursuit_evasion.AdaptiveEvaderBot(),
    ]
    for bot in bots:
      game = pursuit_evasion.PursuitEvasionGame(params={"max_steps": 10})
      state, _ = _play_game(game, evader_bot=bot)
      self.assertLessEqual(state.step, game.max_steps)
      self.assertIn(state.pursuer_reward, [1.0, -1.0])

  def test_legal_actions(self):
    game = pursuit_evasion.PursuitEvasionGame()
    state = game.new_initial_state()
    self.assertEqual(state.current_player(), 0)
    self.assertEqual(state.legal_actions(0), list(range(9)))
    self.assertEmpty(state.legal_actions(1))
    state.apply_action(0)
    self.assertEqual(state.current_player(), 1)
    self.assertEmpty(state.legal_actions(0))
    self.assertEqual(state.legal_actions(1), list(range(9)))

  def test_observation_tensor(self):
    game = pursuit_evasion.PursuitEvasionGame()
    state = game.new_initial_state()
    obs = state.observation_tensor()
    self.assertEqual(obs.shape, (5,))
    self.assertTrue(np.all(obs >= 0.0))
    self.assertTrue(np.all(obs <= 1.0))

  def test_information_state_tensor(self):
    game = pursuit_evasion.PursuitEvasionGame()
    state = game.new_initial_state()
    info = state.information_state_tensor()
    self.assertEqual(info.shape, (5,))
    obs = state.observation_tensor()
    np.testing.assert_array_equal(info, obs)

  def test_deterministic_fixed_strategy(self):
    game = pursuit_evasion.PursuitEvasionGame(params={"max_steps": 10})
    state = game.new_initial_state()
    bot = pursuit_evasion.ConstantVelocityEvaderBot()
    actions = []
    while not state.is_terminal():
      current = state.current_player()
      if current == 0:
        legal = state.legal_actions(0)
        action = legal[0]
      else:
        action = bot.step(state)
      actions.append(action)
      state.apply_action(action)
    state2 = game.new_initial_state()
    bot2 = pursuit_evasion.ConstantVelocityEvaderBot()
    for i, a in enumerate(actions):
      if state2.current_player() == 1 and i > 0:
        bot2.restart_at(state2)
      state2.apply_action(a)
    np.testing.assert_array_equal(state.returns(), state2.returns())
    np.testing.assert_array_equal(state.observation_tensor(),
                                  state2.observation_tensor())

  def test_evader_bots_step_with_policy(self):
    game = pursuit_evasion.PursuitEvasionGame()
    state = game.new_initial_state()
    state.apply_action(0)
    self.assertEqual(state.current_player(), 1)

    bots = [
        pursuit_evasion.RandomEvaderBot(rng=np.random.RandomState(42)),
        pursuit_evasion.ConstantVelocityEvaderBot(),
        pursuit_evasion.ZigzagEvaderBot(),
        pursuit_evasion.AdaptiveEvaderBot(),
    ]
    for bot in bots:
      bot.restart_at(state)
      policy, action = bot.step_with_policy(state)
      self.assertIn(action, range(1, 9))
      self.assertGreater(len(policy), 0)

  def test_num_players(self):
    game = pursuit_evasion.PursuitEvasionGame()
    self.assertEqual(game.num_players(), 2)

  def test_legal_actions_empty_on_terminal(self):
    game = pursuit_evasion.PursuitEvasionGame(
        params={"capture_radius": 100.0, "max_steps": 10})
    state, _ = _play_game(game)
    self.assertTrue(state.is_terminal())
    self.assertEmpty(state.legal_actions(0))
    self.assertEmpty(state.legal_actions(1))


if __name__ == "__main__":
  absltest.main()
