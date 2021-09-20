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
"""Tests for Python dynamic routing game."""

from absl.testing import absltest

from open_spiel.python import rl_environment
from open_spiel.python.algorithms import cfr
from open_spiel.python.algorithms import exploitability
from open_spiel.python.algorithms import external_sampling_mccfr as external_mccfr
from open_spiel.python.algorithms import outcome_sampling_mccfr as outcome_mccfr
from open_spiel.python.games import dynamic_routing
from open_spiel.python.games import dynamic_routing_utils as utils
import pyspiel

# pylint: disable=g-bad-todo


class DynamicRoutingGameTest(absltest.TestCase):

  def setUp(self):
    """Setup the test."""
    super().setUp()
    network = utils.Network({
        'A': ['B'],
        'B': ['C', 'D'],
        'C': ['D', 'E'],
        'D': ['E', 'G'],
        'E': ['F'],
        'F': [],
        'G': []
    })
    vehicles = [utils.Vehicle('A->B', 'E->F') for _ in range(3)]
    self.more_complex_game = dynamic_routing.DynamicRoutingGame(
        network=network, vehicles=vehicles, max_num_time_step=100)
    self.num_iteration_cfr_test = 1

  def test_bad_initialization(self):
    """Test bad initializtion."""
    # network = dynamic_routing_game.Network(
    #     {"O": ["A"], "A": ["D"], "D": []})
    # vehicles = [dynamic_routing_game.Vehicle('O->A', 'D->B')]
    # dynamic_routing_game.DynamicRoutingGame(
    #     network=network, vehicles=vehicles)

  # TODO(Theo): test chance_outcomes()
  # TODO(Theo): test legal_actions()
  # TODO(Theo): test apply_action()
  # TODO(Theo): test apply_actions()
  # TODO: test departure time enabled

  def test_random_game(self):
    """Tests basic API functions."""
    game = self.more_complex_game
    pyspiel.random_sim_test(game, num_sims=10, serialize=False, verbose=True)

  def test_game_as_turn_based(self):
    """Check the game can be converted to a turn-based game."""
    game = self.more_complex_game
    turn_based = pyspiel.convert_to_turn_based(game)
    pyspiel.random_sim_test(
        turn_based, num_sims=10, serialize=False, verbose=True)

  def test_game_as_turn_based_via_string(self):
    """Check the game can be created as a turn-based game from a string."""
    game = pyspiel.load_game(
        'turn_based_simultaneous_game(game=python_dynamic_routing())')
    pyspiel.random_sim_test(game, num_sims=10, serialize=False, verbose=True)

  def test_game_from_cc(self):
    """Runs our standard game tests, checking API consistency."""
    game = pyspiel.load_game('python_dynamic_routing')
    pyspiel.random_sim_test(game, num_sims=10, serialize=False, verbose=True)

  def test_action_consistency_convert_to_turn_based(self):
    """Check if the sequential game is consistent with the game."""
    game = pyspiel.load_game('python_dynamic_routing')
    seq_game = pyspiel.convert_to_turn_based(game)
    state = game.new_initial_state()
    seq_state = seq_game.new_initial_state()
    self.assertEqual(
        state.legal_actions(seq_state.current_player()),
        seq_state.legal_actions(),
        msg='The sequential actions are not correct.')

  def test_cfr_on_turn_based_game_with_exploitability(self):
    """Check if CFR can be applied to the sequential game."""
    game = pyspiel.load_game('python_dynamic_routing')
    seq_game = pyspiel.convert_to_turn_based(game)
    cfr_solver = cfr.CFRSolver(seq_game)
    for _ in range(self.num_iteration_cfr_test):
      cfr_solver.evaluate_and_update_policy()
    exploitability.nash_conv(seq_game, cfr_solver.average_policy())

  def test_ext_mccfr_on_turn_based_game_with_exploitability(self):
    """Check if external sampling MCCFR can be applied."""
    game = pyspiel.load_game('python_dynamic_routing')
    seq_game = pyspiel.convert_to_turn_based(game)
    cfr_solver = external_mccfr.ExternalSamplingSolver(
        seq_game, external_mccfr.AverageType.SIMPLE)
    for _ in range(self.num_iteration_cfr_test):
      cfr_solver.iteration()
    exploitability.nash_conv(seq_game, cfr_solver.average_policy())

  def test_int_mccfr_on_turn_based_game_with_exploitability(self):
    """Check if outcome sampling MCCFR can be applied."""
    game = pyspiel.load_game('python_dynamic_routing')
    seq_game = pyspiel.convert_to_turn_based(game)
    cfr_solver = outcome_mccfr.OutcomeSamplingSolver(seq_game)
    for _ in range(self.num_iteration_cfr_test):
      cfr_solver.iteration()
    exploitability.nash_conv(seq_game, cfr_solver.average_policy())

  def test_creation_of_rl_environment(self):
    """Check if RL environment can be created."""
    game = pyspiel.load_game('python_dynamic_routing')
    seq_game = pyspiel.convert_to_turn_based(game)
    rl_environment.Environment(seq_game)

  # TODO: test evolution of the game as expected (test value of the state).


if __name__ == '__main__':
  absltest.main()
