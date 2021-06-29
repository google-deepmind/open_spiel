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

"""Tests that games are finite horizon.

These tests are intended to help developers to write games that satisfy most of
the unspecified constraints assumed by the following algorithms:
- python/mfg/algorithms/policy_value.py
- python/mfg/algorithms/nash_conv.py
- python/mfg/algorithms/mirror_descent.py
- python/mfg/algorithms/fictitious_play.py
- python/mfg/algorithms/distribution.py
- python/mfg/algorithms/best_response_value.py
These tests are not exhaustive.
"""


from absl.testing import absltest
from absl.testing import parameterized

from open_spiel.python.algorithms import get_all_states
from open_spiel.python.mfg.games import crowd_modelling  # pylint:disable=unused-import
from open_spiel.python.mfg.games import predator_prey  # pylint:disable=unused-import
import pyspiel


def _get_next_states(state, next_states, to_string):
  """Extract non-chance states for a subgame into the all_states dict."""
  is_mean_field = state.current_player() == pyspiel.PlayerId.MEAN_FIELD
  if state.is_chance_node():
    # Add only if not already present

    for action, _ in state.chance_outcomes():
      next_state = state.child(action)
      state_str = to_string(next_state)
      if state_str not in next_states:
        next_states[state_str] = next_state

  if is_mean_field:
    support = state.distribution_support()
    next_state = state.clone()
    support_length = len(support)
    # update with a dummy distribution
    next_state.update_distribution(
        [1.0 / support_length for _ in range(support_length)])
    state_str = to_string(next_state)
    if state_str not in next_states:
      next_states[state_str] = next_state

  if int(state.current_player()) >= 0:
    for action in state.legal_actions():
      next_state = state.child(action)
      state_str = to_string(next_state)
      if state_str not in next_states:
        next_states[state_str] = next_state


def _next_states(states, to_string):
  next_states = {}
  for state in states:
    _get_next_states(state, next_states, to_string)
  return set(next_states.keys()), set(next_states.values())


def type_from_states(states):
  """Get node type of a list of states and assert they are the same."""
  types = [state.get_type() for state in states]
  assert len(set(types)) == 1
  return types[0]


class FiniteHorizonTest(parameterized.TestCase):

  @parameterized.parameters(
      {'game_name': 'python_mfg_crowd_modelling'},
      {'game_name': 'mfg_crowd_modelling'},
      {'game_name': 'mfg_crowd_modelling_2d'},
      {'game_name': 'python_mfg_predator_prey'},
  )
  def test_is_finite_horizon(self, game_name):
    """Check that the game has no loop."""
    game = pyspiel.load_game(game_name)
    states = set(game.new_initial_states())
    to_string = lambda s: s.observation_string(pyspiel.PlayerId.
                                               DEFAULT_PLAYER_ID)
    all_states_key = set(to_string(state) for state in states)
    while type_from_states(states) != pyspiel.StateType.TERMINAL:
      new_states_key, states = _next_states(states, to_string)
      self.assertEmpty(all_states_key.intersection(new_states_key))
      all_states_key.update(new_states_key)

  @parameterized.parameters(
      {'game_name': 'python_mfg_crowd_modelling'},
      {'game_name': 'mfg_crowd_modelling'},
      {'game_name': 'mfg_crowd_modelling_2d'},
      {'game_name': 'python_mfg_predator_prey'},
  )
  def test_has_at_least_an_action(self, game_name):
    """Check that all population's state have at least one action."""
    game = pyspiel.load_game(game_name)
    to_string = lambda s: s.observation_string(pyspiel.PlayerId.
                                               DEFAULT_PLAYER_ID)
    states = get_all_states.get_all_states(
        game,
        depth_limit=-1,
        include_terminals=False,
        include_chance_states=False,
        include_mean_field_states=False,
        to_string=to_string)
    for state in states.values():
      self.assertNotEmpty(state.legal_actions())

if __name__ == '__main__':
  absltest.main()
