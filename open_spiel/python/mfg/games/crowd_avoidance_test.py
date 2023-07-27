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
"""Tests for Python Crowd avoidance game."""

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import numpy.testing as npt
from open_spiel.python.mfg.games import crowd_avoidance
import pyspiel


class MFGCrowdAvoidanceGameTest(parameterized.TestCase):

  def test_load(self):
    game = pyspiel.load_game('python_mfg_crowd_avoidance')
    game.new_initial_state_for_population(0)
    game.new_initial_state_for_population(1)

  @parameterized.parameters(
      {
          'geometry': crowd_avoidance.Geometry.SQUARE,
          'expected_pos': np.array([5, 3]),
      },
      {
          'geometry': crowd_avoidance.Geometry.TORUS,
          'expected_pos': np.array([5, 3]),
      },
  )
  def test_dynamics(self, geometry, expected_pos):
    game = pyspiel.load_game(
        'python_mfg_crowd_avoidance',
        {
            'geometry': geometry,
        },
    )
    state = game.new_initial_state_for_population(1)
    # Initial chance node.
    self.assertEqual(state.current_player(), pyspiel.PlayerId.CHANCE)
    self.assertLen(state.chance_outcomes(), 3)
    self.assertEqual(
        state.chance_outcomes()[0][0],
        crowd_avoidance.pos_to_merged(np.array([5, 2]), state.size),
    )
    state.apply_action(state.chance_outcomes()[0][0])
    self.assertEqual(state.current_player(), 1)
    npt.assert_array_equal(state.pos, [5, 2])
    self.assertEqual(state._action_to_string(player=1, action=2), '[0 1]')
    state.apply_action(2)
    npt.assert_array_equal(state.pos, expected_pos)

  def test_create_with_params(self):
    setting = 'python_mfg_crowd_avoidance()'
    game = pyspiel.load_game(setting)
    self.assertEqual(game.size, 7)
    self.assertEqual(game.horizon, 10)

  @parameterized.parameters(
      {'population': 0},
      {'population': 1},
  )
  def test_random_game(self, population):
    """Tests basic API functions."""
    congestion_matrix = np.array([[0, 1], [1, 0]])
    init_distrib = np.array([
        # First population
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.4, 0.4, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        # Second population
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.4, 0.4, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ])
    forbidden_states_grid = [
        '#######',
        '#  #  #',
        '#     #',
        '#  #  #',
        '#     #',
        '#  #  #',
        '#######',
    ]
    forbidden_states = crowd_avoidance.grid_to_forbidden_states(
        forbidden_states_grid
    )
    game = crowd_avoidance.MFGCrowdAvoidanceGame(
        params={
            'horizon': 10,
            'size': 7,
            'players': 2,
            'congestion_matrix': ' '.join(
                str(v) for v in congestion_matrix.flatten()
            ),
            'init_distrib': ' '.join(str(v) for v in init_distrib.flatten()),
            'forbidden_states': forbidden_states,
        }
    )
    pyspiel.random_sim_test(
        game,
        num_sims=10,
        serialize=False,
        verbose=True,
        mean_field_population=population,
    )

  @parameterized.parameters(
      {
          'coef_congestion': 1.5,
          'coef_target': 0.6,
          'congestion_matrix': np.array([[0, 1], [1, 0]]),
          'population': 0,
          'players': 2,
          'initial_pos': np.array([0, 0]),
          'distributions': [
              # First population
              np.array([[0.8, 0.2], [0.0, 0.0]]),
              # Second population
              np.array([[0.3, 0.7], [0.0, 0.0]]),
          ],
          'expected_rewards': np.array([
              -1.5 * 0.3 + 0.0,
              -1.5 * 0.8 + 0.0,
          ]),
          'init_distrib': np.array([
              # First population
              [0.8, 0.2],
              [0.0, 0.0],
              # Second population
              [0.3, 0.7],
              [0.0, 0.0],
          ]),
      },
  )
  def test_rewards(
      self,
      coef_congestion,
      coef_target,
      congestion_matrix,
      players,
      population,
      initial_pos,
      distributions,
      expected_rewards,
      init_distrib,
  ):
    game = pyspiel.load_game(
        'python_mfg_crowd_avoidance',
        {
            'size': 2,
            'coef_congestion': coef_congestion,
            'coef_target': coef_target,
            'congestion_matrix': ' '.join(
                str(v) for v in congestion_matrix.flatten()
            ),
            'players': players,
            'init_distrib': ' '.join(str(v) for v in init_distrib.flatten()),
            'forbidden_states': '[]',
        },
    )
    state = game.new_initial_state_for_population(population)
    # Initial chance node.
    self.assertEqual(state.current_player(), pyspiel.PlayerId.CHANCE)
    state.apply_action(crowd_avoidance.pos_to_merged(initial_pos, state.size))
    self.assertEqual(state.current_player(), population)
    npt.assert_array_equal(state.pos, initial_pos)
    state.apply_action(state._NEUTRAL_ACTION)
    npt.assert_array_equal(state.pos, initial_pos)
    self.assertEqual(state.current_player(), pyspiel.PlayerId.CHANCE)
    state.apply_action(state._NEUTRAL_ACTION)
    self.assertEqual(state.current_player(), pyspiel.PlayerId.MEAN_FIELD)

    # Maps states (in string representation) to their proba.
    dist = {}
    for x in range(state.size):
      for y in range(state.size):
        for pop in range(len(congestion_matrix)):
          state_str = state.state_to_str(
              np.array([x, y]),
              state.t,
              pop,
              player_id=pyspiel.PlayerId.MEAN_FIELD,
          )
          dist[state_str] = distributions[pop][y][x]
    support = state.distribution_support()
    state.update_distribution([dist[s] for s in support])

    # Decision node where we get a reward.
    self.assertEqual(state.current_player(), population)
    npt.assert_array_equal(state.rewards(), expected_rewards)


if __name__ == '__main__':
  absltest.main()
