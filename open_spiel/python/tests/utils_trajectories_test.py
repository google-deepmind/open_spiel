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

from absl.testing import absltest
import numpy as np

import pyspiel

SIM_GAMES = [
    "backgammon",
    "bargaining",
    "breakthrough(rows=6,columns=6)",
    "chess",
    "checkers",
    "connect_four",
    "goofspiel",
    "kuhn_poker",
    "leduc_poker",
    "liars_dice",
]


EXAMPLE_TRAJECTORY_STRING = """
{
  "header": {
    "game_string": "tic_tac_toe",
    "terminal": true,
    "returns": [1, -1],
    "meta_data": "some_extra_info"
  },
  "transitions": [
    {
      "player": 0,
      "action": 4,
      "legal_actions": [0, 1, 2, 3, 4, 5, 6, 7, 8]
    },
    {
      "player": 1,
      "action": 3,
      "legal_actions": [0, 1, 2, 3, 5, 6, 7, 8]
    },
    {
      "player": 0,
      "action": 6,
      "legal_actions": [0, 1, 2, 5, 6, 7, 8]
    },
    {
      "player": 1,
      "action": 2,
      "legal_actions": [0, 1, 2, 5, 7, 8]
    },
    {
      "player": 0,
      "action": 8,
      "legal_actions": [0, 1, 5, 7, 8]
    },
    {
      "player": 1,
      "action": 0,
      "legal_actions": [0, 1, 5, 7]
    },
    {
      "player": 0,
      "action": 7,
      "legal_actions": [1, 5, 7]
    }
  ]
}
"""


class UtilsTrajectoriesTest(absltest.TestCase):

  def test_example_trajectory(self):
    trajectory = pyspiel.Trajectory(EXAMPLE_TRAJECTORY_STRING)

    header = trajectory.header()
    self.assertEqual(header.game_string, "tic_tac_toe")
    self.assertTrue(header.terminal)
    self.assertEqual(header.returns, [1.0, -1.0])
    self.assertEqual(header.meta_data, "some_extra_info")

    trajectory_str = trajectory.to_string()

    states = trajectory.reconstruct_all_states()
    self.assertLen(states, 8)

    final_state = trajectory.reconstruct_final_state()
    self.assertTrue(final_state.is_terminal())
    self.assertEqual(final_state.to_string(), states[-1].to_string())

    trajectory2 = pyspiel.Trajectory(trajectory_str)
    final_state2 = trajectory2.reconstruct_final_state()
    self.assertIsNotNone(final_state2)
    self.assertTrue(final_state2.is_terminal())
    self.assertEqual(final_state2.to_string(), final_state.to_string())
    self.assertEqual(final_state.history_str(), final_state2.history_str())

  def test_random_simulation_trajectories(self):
    for game_string in SIM_GAMES:
      with self.subTest(game=game_string):
        self._random_simulation_trajectory_test(game_string)

  def _random_simulation_trajectory_test(self, game_string):
    game = pyspiel.load_game(game_string)
    state = game.new_initial_state()
    rng = np.random.default_rng(42)

    expected_steps = 0
    while not state.is_terminal():
      if state.is_chance_node():
        outcomes = state.chance_outcomes()
        actions, probs = zip(*outcomes)
        action = rng.choice(actions, p=probs)
        state.apply_action(action)
        expected_steps += 1
      elif state.is_mean_field_node():
        raise ValueError("Mean field nodes not supported in this test.")
      elif state.is_simultaneous_node():
        joint_action = []
        for player in range(game.num_players()):
          actions = state.legal_actions(player)
          action = 0
          if actions:
            action = rng.choice(actions)
          joint_action.append(action)
        state.apply_actions(joint_action)
        expected_steps += 1
      else:
        actions = state.legal_actions()
        action = rng.choice(actions)
        state.apply_action(action)
        expected_steps += 1

    trajectory = pyspiel.Trajectory(state)
    trajectory_str = trajectory.to_string()

    states = trajectory.reconstruct_all_states()
    self.assertLen(states, expected_steps + 1)

    reconstructed_final_state = trajectory.reconstruct_final_state()
    self.assertIsNotNone(reconstructed_final_state)
    self.assertTrue(reconstructed_final_state.is_terminal())
    self.assertEqual(
        reconstructed_final_state.to_string(), states[-1].to_string()
    )
    self.assertEqual(reconstructed_final_state.to_string(), state.to_string())

    trajectory2 = pyspiel.Trajectory(trajectory_str)
    reconstructed_final_state2 = trajectory2.reconstruct_final_state()
    self.assertIsNotNone(reconstructed_final_state2)
    self.assertTrue(reconstructed_final_state2.is_terminal())
    self.assertEqual(reconstructed_final_state2.to_string(), state.to_string())
    self.assertEqual(
        state.history_str(), reconstructed_final_state2.history_str()
    )


if __name__ == "__main__":
  np.random.seed(87375711)
  absltest.main()
