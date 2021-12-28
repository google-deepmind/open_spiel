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
"""Tests for Python dynamic routing game."""

from absl.testing import absltest

from open_spiel.python import games  # pylint:disable=unused-import
from open_spiel.python import policy
from open_spiel.python import rl_environment
from open_spiel.python.algorithms import cfr
from open_spiel.python.algorithms import expected_game_score
from open_spiel.python.algorithms import exploitability
from open_spiel.python.algorithms import external_sampling_mccfr as external_mccfr
from open_spiel.python.algorithms import outcome_sampling_mccfr as outcome_mccfr
from open_spiel.python.games import dynamic_routing
from open_spiel.python.games import dynamic_routing_utils
import pyspiel

_NUM_ITERATION_CFR_TEST = 1


class DynamicRoutingGameTest(absltest.TestCase):

  def test_random_game(self):
    """Tests basic API functions with the standard game tests."""
    game = pyspiel.load_game("python_dynamic_routing")
    pyspiel.random_sim_test(game, num_sims=10, serialize=False, verbose=True)

  def test_game_as_turn_based(self):
    """Check the game can be converted to a turn-based game."""
    game = pyspiel.load_game("python_dynamic_routing")
    turn_based = pyspiel.convert_to_turn_based(game)
    pyspiel.random_sim_test(
        turn_based, num_sims=10, serialize=False, verbose=True)

  def test_game_as_turn_based_via_string(self):
    """Check the game can be created as a turn-based game from a string."""
    game = pyspiel.load_game(
        "turn_based_simultaneous_game(game=python_dynamic_routing())")
    pyspiel.random_sim_test(game, num_sims=10, serialize=False, verbose=True)

  def test_non_default_param_from_string(self):
    """Check params can be given through string loading."""
    game = pyspiel.load_game("python_dynamic_routing(max_num_time_step=5)")
    self.assertEqual(game.max_game_length(), 5)

  def test_non_default_param_from_dict(self):
    """Check params can be given through a dictionary."""
    game = pyspiel.load_game("python_dynamic_routing", {"max_num_time_step": 5})
    self.assertEqual(game.max_game_length(), 5)

  def test_action_consistency_convert_to_turn_based(self):
    """Check if the sequential game is consistent with the game."""
    game = pyspiel.load_game("python_dynamic_routing")
    seq_game = pyspiel.convert_to_turn_based(game)
    state = game.new_initial_state()
    seq_state = seq_game.new_initial_state()
    self.assertEqual(
        state.legal_actions(seq_state.current_player()),
        seq_state.legal_actions(),
        msg="The sequential actions are not correct.")

  def test_cfr_on_turn_based_game_with_exploitability(self):
    """Check if CFR can be applied to the sequential game."""
    game = pyspiel.load_game(
        "python_dynamic_routing(max_num_time_step=5,time_step_length=1.0)")
    seq_game = pyspiel.convert_to_turn_based(game)
    cfr_solver = cfr.CFRSolver(seq_game)
    for _ in range(_NUM_ITERATION_CFR_TEST):
      cfr_solver.evaluate_and_update_policy()
    exploitability.nash_conv(seq_game, cfr_solver.average_policy())

  def test_ext_mccfr_on_turn_based_game_with_exploitability(self):
    """Check if external sampling MCCFR can be applied."""
    game = pyspiel.load_game(
        "python_dynamic_routing(max_num_time_step=5,time_step_length=1.0)")
    seq_game = pyspiel.convert_to_turn_based(game)
    cfr_solver = external_mccfr.ExternalSamplingSolver(
        seq_game, external_mccfr.AverageType.SIMPLE)
    for _ in range(_NUM_ITERATION_CFR_TEST):
      cfr_solver.iteration()
    exploitability.nash_conv(seq_game, cfr_solver.average_policy())

  def test_int_mccfr_on_turn_based_game_with_exploitability(self):
    """Check if outcome sampling MCCFR can be applied."""
    game = pyspiel.load_game(
        "python_dynamic_routing(max_num_time_step=5,time_step_length=1.0)")
    seq_game = pyspiel.convert_to_turn_based(game)
    cfr_solver = outcome_mccfr.OutcomeSamplingSolver(seq_game)
    for _ in range(_NUM_ITERATION_CFR_TEST):
      cfr_solver.iteration()
    exploitability.nash_conv(seq_game, cfr_solver.average_policy())

  def test_creation_of_rl_environment(self):
    """Check if RL environment can be created."""
    game = pyspiel.load_game("python_dynamic_routing")
    seq_game = pyspiel.convert_to_turn_based(game)
    rl_environment.Environment(seq_game)

  def test_vehicle_origin_outside_network(self):
    """Check raise assertion if vehicle's origin is outside the Network."""
    vehicles = [dynamic_routing_utils.Vehicle("I->O", "D->E", 0)]
    with self.assertRaises(ValueError):
      dynamic_routing.DynamicRoutingGame(
          {
              "max_num_time_step": 10,
              "time_step_length": 0.5,
              "players": -1
          },
          vehicles=vehicles)

  def test_vehicle_destination_outside_network(self):
    """Check raise assertion if vehicle's destination is outside the Network."""
    vehicles = [dynamic_routing_utils.Vehicle("O->A", "E->F", 0)]
    with self.assertRaises(ValueError):
      dynamic_routing.DynamicRoutingGame(
          {
              "max_num_time_step": 10,
              "time_step_length": 0.5,
              "players": -1
          },
          vehicles=vehicles)

  def test_multiple_departure_time_vehicle(self):
    """Check that departure time can be define."""
    vehicles = [
        dynamic_routing_utils.Vehicle("O->A", "D->E", 0),
        dynamic_routing_utils.Vehicle("O->A", "D->E", 0.5),
        dynamic_routing_utils.Vehicle("O->A", "D->E", 1.0)
    ]
    game = dynamic_routing.DynamicRoutingGame(
        {
            "max_num_time_step": 10,
            "time_step_length": 0.5,
            "players": -1
        },
        vehicles=vehicles)
    pyspiel.random_sim_test(game, num_sims=10, serialize=False, verbose=True)

  def test_game_evolution_first_action_policy(self):
    """Check game deterministic evolution under first action policy."""
    # TODO(cabannes): test evolution of the game as expected (test value of the
    # state).
    # TODO(cabannes): test legal_actions().

  def test_observer_correct(self):
    """Check that the observer is correclty updated."""
    # TODO(cabannes): add test about observer and tensor being updated.

  def test_apply_actions_error_no_movement_with_negative_waiting_time(self):
    """Check that a vehicle cannot choose to not move if it has to move."""
    # TODO(cabannes): test apply_actions().

  def test_apply_actions_error_wrong_movement_with_negative_waiting_time(self):
    """Check that a vehicle cannot choose to move to a not successor link."""
    # TODO(cabannes): test apply_actions().

  def test_apply_actions_error_movement_with_positive_waiting_time(self):
    """Check that a vehicle cannot choose to move if it cannot move yet."""
    # TODO(cabannes): test apply_actions().

  def test_braess_paradox(self):
    """Test that Braess paradox can be reproduced with the mean field game."""
    num_player = 8
    braess_network = dynamic_routing_utils.Network(
        {
            "O": "A",
            "A": ["B", "C"],
            "B": ["C", "D"],
            "C": ["D"],
            "D": ["E"],
            "E": []
        },
        node_position={
            "O": (0, 0),
            "A": (1, 0),
            "B": (2, 1),
            "C": (2, -1),
            "D": (3, 0),
            "E": (4, 0)
        },
        bpr_a_coefficient={
            "O->A": 0,
            "A->B": 1.0,
            "A->C": 0,
            "B->C": 0,
            "B->D": 0,
            "C->D": 1.0,
            "D->E": 0
        },
        bpr_b_coefficient={
            "O->A": 1.0,
            "A->B": 1.0,
            "A->C": 1.0,
            "B->C": 1.0,
            "B->D": 1.0,
            "C->D": 1.0,
            "D->E": 1.0
        },
        capacity={
            "O->A": num_player,
            "A->B": num_player,
            "A->C": num_player,
            "B->C": num_player,
            "B->D": num_player,
            "C->D": num_player,
            "D->E": num_player
        },
        free_flow_travel_time={
            "O->A": 0,
            "A->B": 1.0,
            "A->C": 2.0,
            "B->C": 0.25,
            "B->D": 2.0,
            "C->D": 1.0,
            "D->E": 0
        })

    demand = [
        dynamic_routing_utils.Vehicle("O->A", "D->E") for _ in range(num_player)
    ]
    game = dynamic_routing.DynamicRoutingGame(
        {"time_step_length": 0.125, "max_num_time_step": 40},
        network=braess_network,
        vehicles=demand)

    class TruePathPolicy(policy.Policy):

      def __init__(self, game):
        super().__init__(game, list(range(num_player)))
        self._path = {}

      def action_probabilities(self, state, player_id=None):
        assert player_id is not None
        legal_actions = state.legal_actions(player_id)
        if not legal_actions:
          return {dynamic_routing_utils.NO_POSSIBLE_ACTION: 1.0}
        elif len(legal_actions) == 1:
          return {legal_actions[0]: 1.0}
        else:
          if legal_actions[0] == 2:
            if self._path[player_id] in ["top", "middle"]:
              return {2: 1.0}
            elif self._path[player_id] == "bottom":
              return {3: 1.0}
            else:
              raise ValueError()
          elif legal_actions[0] == 4:
            if self._path[player_id] == "top":
              return {5: 1.0}
            elif self._path[player_id] == "middle":
              return {4: 1.0}
            else:
              raise ValueError()
        raise ValueError(f"{legal_actions} is not correct.")

    class NashEquilibriumBraess(TruePathPolicy):

      def __init__(self, game):
        super().__init__(game)
        for player_id in range(num_player):
          if player_id % 2 == 0:
            self._path[player_id] = "middle"
          if player_id % 4 == 1:
            self._path[player_id] = "top"
          if player_id % 4 == 3:
            self._path[player_id] = "bottom"

    class SocialOptimumBraess(NashEquilibriumBraess):

      def __init__(self, game):
        super().__init__(game)
        for player_id in range(num_player):
          if player_id % 2 == 0:
            self._path[player_id] = "top"
          if player_id % 2 == 1:
            self._path[player_id] = "bottom"

    ne_policy = NashEquilibriumBraess(game)
    # TODO(cabannes): debug issue with nash conv computation and uncomment the
    # following line.
    # self.assertEqual(exploitability.nash_conv(game, ne_policy), 0.0)
    self.assertSequenceAlmostEqual(
        -expected_game_score.policy_value(game.new_initial_state(), ne_policy),
        [3.75] * num_player)

    so_policy = SocialOptimumBraess(game)
    # TODO(cabannes): debug issue with nash conv computation and uncomment the
    # following line.
    # self.assertEqual(exploitability.nash_conv(game, so_policy), 0.125)
    self.assertSequenceAlmostEqual(
        -expected_game_score.policy_value(game.new_initial_state(), so_policy),
        [3.5] * num_player)


if __name__ == "__main__":
  absltest.main()
