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
"""Tests for Python mean field routing game."""

from absl.testing import absltest
import numpy as np
import numpy.testing as npt

from open_spiel.python import games  # pylint:disable=unused-import
from open_spiel.python import policy
from open_spiel.python.games import dynamic_routing_utils
from open_spiel.python.mfg import games as mfg_games  # pylint:disable=unused-import
from open_spiel.python.mfg.algorithms import distribution
from open_spiel.python.mfg.algorithms import mirror_descent
from open_spiel.python.mfg.algorithms import nash_conv
from open_spiel.python.mfg.algorithms import policy_value
from open_spiel.python.mfg.games import dynamic_routing
from open_spiel.python.mfg.games import factory
from open_spiel.python.observation import make_observation
import pyspiel

_NUMBER_OF_ITERATIONS_TESTS = 1


class MeanFieldRoutingGameTest(absltest.TestCase):
  """Checks we can create the game and clone states."""

  def test_load(self):
    """Test load and game creation."""
    game = pyspiel.load_game("python_mfg_dynamic_routing")
    game.new_initial_state()

  def test_create(self):
    """Checks we can create the game and clone states."""
    game = pyspiel.load_game("python_mfg_dynamic_routing")
    self.assertEqual(game.get_type().dynamics,
                     pyspiel.GameType.Dynamics.MEAN_FIELD)
    state = game.new_initial_state()
    state.clone()

  def test_random_game(self):
    """Test random simulation."""
    game = pyspiel.load_game("python_mfg_dynamic_routing")
    pyspiel.random_sim_test(game, num_sims=10, serialize=False, verbose=True)

  def test_evolving_trajectory_with_uniform_policy(self):
    """Test evolving distribution."""
    game = pyspiel.load_game("python_mfg_dynamic_routing")
    distribution.DistributionPolicy(game, policy.UniformRandomPolicy(game))

  def test_non_default_param_from_string(self):
    """Check params can be given through string loading."""
    game = pyspiel.load_game("python_mfg_dynamic_routing(max_num_time_step=5)")
    self.assertEqual(game.max_game_length(), 5)

  def test_non_default_param_from_dict(self):
    """Check params can be given through a dictionary."""
    game = pyspiel.load_game("python_mfg_dynamic_routing",
                             {"max_num_time_step": 5})
    self.assertEqual(game.max_game_length(), 5)

  # TODO(open_spiel): enable ficticious_play with game where the dynamics depend
  # on the distribution.
  # def test_ficticious_play(self):
  #   """Test that ficticious play can be used on this game."""
  #   mfg_game = pyspiel.load_game("python_mfg_dynamic_routing")
  #   fp = fictitious_play.FictitiousPlay(mfg_game)
  #   for _ in range(_NUMBER_OF_ITERATIONS_TESTS):
  #     fp.iteration()
  #   nash_conv.NashConv(mfg_game, fp.get_policy())

  def test_online_mirror_descent(self):
    """Test that online mirror descent can be used on this game."""
    mfg_game = pyspiel.load_game("python_mfg_dynamic_routing")
    omd = mirror_descent.MirrorDescent(mfg_game)
    for _ in range(_NUMBER_OF_ITERATIONS_TESTS):
      omd.iteration()
    nash_conv.NashConv(mfg_game, omd.get_policy())

  def test_online_mirror_descent_convergence(self):
    """Test that online mirror descent converges to equilibrium in default game."""
    mfg_game = pyspiel.load_game("python_mfg_dynamic_routing", {
        "time_step_length": 0.05,
        "max_num_time_step": 100
    })
    omd = mirror_descent.MirrorDescent(mfg_game, lr=1)
    for _ in range(50):
      omd.iteration()
    self.assertAlmostEqual(
        nash_conv.NashConv(mfg_game, omd.get_policy()).nash_conv(), 0)

  def test_braess_paradox(self):
    """Test that Braess paradox can be reproduced with the mean field game."""
    mfg_game = pyspiel.load_game("python_mfg_dynamic_routing", {
        "time_step_length": 0.05,
        "max_num_time_step": 100
    })

    class NashEquilibriumBraess(policy.Policy):

      def action_probabilities(self, state, player_id=None):
        legal_actions = state.legal_actions()
        if not legal_actions:
          return {dynamic_routing_utils.NO_POSSIBLE_ACTION: 1.0}
        elif len(legal_actions) == 1:
          return {legal_actions[0]: 1.0}
        else:
          if legal_actions[0] == 2:
            return {2: 0.75, 3: 0.25}
          elif legal_actions[0] == 4:
            return {4: 2 / 3, 5: 1 / 3}
        raise ValueError(f"{legal_actions} is not correct.")

    ne_policy = NashEquilibriumBraess(mfg_game, 1)
    self.assertEqual(
        -policy_value.PolicyValue(
            mfg_game, distribution.DistributionPolicy(mfg_game, ne_policy),
            ne_policy).value(mfg_game.new_initial_state()), 3.75)
    self.assertEqual(nash_conv.NashConv(mfg_game, ne_policy).nash_conv(), 0.0)

    class SocialOptimumBraess(policy.Policy):

      def action_probabilities(self, state, player_id=None):
        legal_actions = state.legal_actions()
        if not legal_actions:
          return {dynamic_routing_utils.NO_POSSIBLE_ACTION: 1.0}
        elif len(legal_actions) == 1:
          return {legal_actions[0]: 1.0}
        else:
          if legal_actions[0] == 2:
            return {2: 0.5, 3: 0.5}
          elif legal_actions[0] == 4:
            return {5: 1.0}
        raise ValueError(f"{legal_actions} is not correct.")

    so_policy = SocialOptimumBraess(mfg_game, 1)
    self.assertEqual(
        -policy_value.PolicyValue(
            mfg_game, distribution.DistributionPolicy(mfg_game, so_policy),
            so_policy).value(mfg_game.new_initial_state()), 3.5)
    self.assertEqual(nash_conv.NashConv(mfg_game, so_policy).nash_conv(), 0.75)

  def test_vehicle_origin_outside_network(self):
    """Check raise assertion if vehicle's origin is outside the Network."""
    od_demand = [
        dynamic_routing_utils.OriginDestinationDemand("I->O", "D->E", 0, 5)
    ]
    with self.assertRaises(ValueError):
      dynamic_routing.MeanFieldRoutingGame(
          {
              "max_num_time_step": 10,
              "time_step_length": 0.5,
              "players": -1
          },
          od_demand=od_demand)

  def test_vehicle_destination_outside_network(self):
    """Check raise assertion if vehicle's destination is outside the Network."""
    od_demand = [
        dynamic_routing_utils.OriginDestinationDemand("O->A", "E->F", 0, 5)
    ]
    with self.assertRaises(ValueError):
      dynamic_routing.MeanFieldRoutingGame(
          {
              "max_num_time_step": 10,
              "time_step_length": 0.5,
              "players": -1
          },
          od_demand=od_demand)

  def test_multiple_departure_time_vehicle(self):
    """Check that departure time can be define."""
    od_demand = [
        dynamic_routing_utils.OriginDestinationDemand("O->A", "D->E", 0, 5),
        dynamic_routing_utils.OriginDestinationDemand("O->A", "D->E", 0.5, 5),
        dynamic_routing_utils.OriginDestinationDemand("O->A", "D->E", 1.0, 5)
    ]
    game = dynamic_routing.MeanFieldRoutingGame(
        {
            "max_num_time_step": 10,
            "time_step_length": 0.5,
            "players": -1
        },
        od_demand=od_demand)
    pyspiel.random_sim_test(game, num_sims=10, serialize=False, verbose=True)

  def test_game_evolution_uniform_policy(self):
    """Check game evolution under uniform policy."""
    # TODO(cabannes): test evolution of the game as expected (test value of the
    # state).
    # TODO(cabannes): test legal_actions().

  def test_observer_correct(self):
    """Checks that the observer is correctly updated."""
    game = pyspiel.load_game("python_mfg_dynamic_routing")
    num_locations, steps = 8, 10
    self.assertEqual(game.num_distinct_actions(), num_locations)
    self.assertEqual(game.max_game_length(), steps)
    py_obs = make_observation(game)

    state = game.new_initial_state()
    self.assertEqual(state.current_player(), pyspiel.PlayerId.CHANCE)

    state.apply_action(0)
    self.assertEqual(state.current_player(), 0)

    location, destination = 1, 7
    self.assertEqual(state.get_location_as_int(), location)
    self.assertEqual(state.get_destination_as_int(), destination)

    py_obs.set_from(state, state.current_player())
    obs_size = num_locations * 2 + steps + 2
    expected_tensor = np.zeros(obs_size)
    # location = 1
    # destination + num_locations = 15
    # time + 2 * num_locations = 16
    # waiting bit at last index.
    expected_tensor[[1, 15, 16]] = 1
    npt.assert_array_equal(py_obs.tensor, expected_tensor)

  def test_apply_actions_error_no_movement_with_negative_waiting_time(self):
    """Check that a vehicle cannot choose to not move if it has to move."""
    # TODO(cabannes): test apply_actions().

  def test_apply_actions_error_wrong_movement_with_negative_waiting_time(self):
    """Check that a vehicle cannot choose to move to a not successor link."""
    # TODO(cabannes): test apply_actions().

  def test_apply_actions_error_movement_with_positive_waiting_time(self):
    """Check that a vehicle cannot choose to move if it cannot move yet."""
    # TODO(cabannes): test apply_actions().

  @absltest.skip(
      "Test of OMD on Sioux Falls is disabled as it takes a long time to run.")
  def test_online_mirror_descent_sioux_falls_dummy(self):
    """Test that online mirror descent can be used on the Sioux Falls game."""
    mfg_game = factory.create_game_with_setting(
        "python_mfg_dynamic_routing",
        "dynamic_routing_sioux_falls_dummy_demand")
    omd = mirror_descent.MirrorDescent(mfg_game)
    for _ in range(_NUMBER_OF_ITERATIONS_TESTS):
      omd.iteration()
    nash_conv.NashConv(mfg_game, omd.get_policy())


if __name__ == "__main__":
  absltest.main()
