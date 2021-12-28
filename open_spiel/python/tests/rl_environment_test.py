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

"""Tests for open_spiel.python.pybind11.pyspiel."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest

from open_spiel.python import rl_environment
import pyspiel


class RLEnvironmentTest(absltest.TestCase):

  def test_create_game(self):
    env = rl_environment.Environment("tic_tac_toe")
    self.assertEqual(env.is_turn_based, True)
    self.assertEqual(env.num_players, 2)

  def test_create_game_with_args(self):
    env = rl_environment.Environment("kuhn_poker", **{"players": 3})
    self.assertEqual(env.is_turn_based, True)
    self.assertEqual(env.num_players, 3)

  def test_create_env_from_game_instance(self):
    game = pyspiel.load_game("tic_tac_toe")
    env = rl_environment.Environment(game)
    self.assertEqual(env.is_turn_based, True)
    self.assertEqual(env.num_players, 2)

  def test_reset(self):
    env = rl_environment.Environment("kuhn_poker", **{"players": 3})
    time_step = env.reset()
    self.assertEqual(time_step.observations["current_player"], 0)
    self.assertEmpty(time_step.observations["serialized_state"], 0)
    self.assertLen(time_step.observations["info_state"], 3)
    self.assertLen(time_step.observations["legal_actions"], 3)
    self.assertIsNone(time_step.rewards)
    self.assertIsNone(time_step.discounts)
    self.assertEqual(time_step.step_type.first(), True)

  def test_initial_info_state_is_decision_node(self):
    env = rl_environment.Environment("kuhn_poker")
    time_step = env.reset()
    self.assertEqual(time_step.step_type.first(), True)
    self.assertEqual(env.is_chance_node, False)

  def test_full_game(self):
    env = rl_environment.Environment("tic_tac_toe", include_full_state=True)
    _ = env.reset()
    time_step = env.step([0])
    self.assertEqual(time_step.observations["current_player"], 1)
    self.assertLen(time_step.observations["info_state"], 2)
    self.assertLen(time_step.observations["legal_actions"], 2)
    self.assertLen(time_step.rewards, 2)
    self.assertLen(time_step.discounts, 2)
    self.assertLen(time_step.observations, 4)

    # O X O   # Moves 0, 1, 2
    # X O X   # Moves 3, 4, 5
    # O . .   # Move 6, game over (player 0 wins).

    for i in range(1, 7):
      self.assertEqual(time_step.step_type.mid(), True)
      time_step = env.step([i])
    self.assertEqual(time_step.step_type.last(), True)

  def test_spec_fields(self):
    env = rl_environment.Environment("tic_tac_toe")
    env_spec = env.observation_spec()
    action_spec = env.action_spec()

    ttt_max_actions = 9
    ttt_normalized_info_set_shape = (27,)

    self.assertEqual(action_spec["num_actions"], ttt_max_actions)
    self.assertEqual(env_spec["info_state"], ttt_normalized_info_set_shape)
    self.assertCountEqual(
        env_spec.keys(),
        ["current_player", "info_state", "serialized_state", "legal_actions"])
    self.assertCountEqual(action_spec.keys(),
                          ["dtype", "max", "min", "num_actions"])

  def test_full_game_simultaneous_move(self):
    env = rl_environment.Environment("goofspiel")
    _ = env.reset()
    time_step = env.step([0, 0])
    self.assertEqual(time_step.observations["current_player"],
                     rl_environment.SIMULTANEOUS_PLAYER_ID)
    self.assertLen(time_step.observations["info_state"], 2)
    self.assertLen(time_step.observations["legal_actions"], 2)
    self.assertLen(time_step.rewards, 2)
    self.assertLen(time_step.discounts, 2)
    self.assertLen(time_step.observations, 4)

    actions = [act[0] for act in time_step.observations["legal_actions"]]
    time_step = env.step(actions)
    self.assertEqual(time_step.step_type.mid(), True)

    while not time_step.last():
      actions = [act[0] for act in time_step.observations["legal_actions"]]
      time_step = env.step(actions)

  def test_set_and_get_state(self):
    env_ttt1 = rl_environment.Environment("tic_tac_toe")
    env_ttt2 = rl_environment.Environment("tic_tac_toe")
    env_kuhn1 = rl_environment.Environment("kuhn_poker", players=2)
    env_kuhn2 = rl_environment.Environment("kuhn_poker", players=3)

    env_ttt1.reset()
    env_ttt2.reset()
    env_kuhn1.reset()
    env_kuhn2.reset()

    # Transfering states between identical games should work.
    env_ttt1.set_state(env_ttt2.get_state)
    env_ttt2.set_state(env_ttt1.get_state)

    # Transfering states between different games or games with different
    # parameters should fail.
    with self.assertRaises(AssertionError):
      self.fail(env_ttt1.set_state(env_kuhn1.get_state))
    with self.assertRaises(AssertionError):
      self.fail(env_kuhn1.set_state(env_ttt1.get_state))

    with self.assertRaises(AssertionError):
      self.fail(env_kuhn1.set_state(env_kuhn2.get_state))
    with self.assertRaises(AssertionError):
      self.fail(env_kuhn2.set_state(env_kuhn1.get_state))


if __name__ == "__main__":
  absltest.main()
