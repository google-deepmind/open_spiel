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

"""Mean field routing game policy used in N-playerrouting game.

The policy class DerivedNPlayerPolicyFromMeanFieldPolicy convert a mean field
routing game policy to a N player routing game policy. It keep in memory the
mean field policy and convert a N player routing game state to a mean field
routing game state when calling action_probabilities. Therefore the mean field
policy can be used on a N player state. This makes the mean field equilibrium
policy (which is faster to compute) a policy that approximate well the
equilbrium N player policy (which is slower to compute) when N is large.
"""

from typing import Dict

from open_spiel.python import policy
from open_spiel.python.games import dynamic_routing
from open_spiel.python.games import dynamic_routing_utils
from open_spiel.python.mfg.games import dynamic_routing as mean_field_routing_game
import pyspiel


def _create_empty_mfg_state(game: dynamic_routing.DynamicRoutingGame):
  """Create an empty MFG state for the N player routing game.

  Args:
    game: the N player game.

  Returns:
    new_mfg_state: an empty MFG state corresponding to the N player game.
  """
  od_demand_dict = {}
  for vehicle in game._vehicles:  # pylint:disable=protected-access
    key = (vehicle.origin, vehicle.destination, vehicle.departure_time)
    if key not in od_demand_dict:
      od_demand_dict[key] = 0
    od_demand_dict[key] += 1
  od_demand = []
  for (origin, destination, departure_time), counts in od_demand_dict.items():
    od_demand.append(
        dynamic_routing_utils.OriginDestinationDemand(origin, destination,
                                                      departure_time, counts))
  return mean_field_routing_game.MeanFieldRoutingGame(
      {
          "max_num_time_step": game.max_game_length(),
          "time_step_length": game.time_step_length
      },
      network=game.network,
      od_demand=od_demand,
      perform_sanity_checks=game.perform_sanity_checks).new_initial_state()


class DerivedNPlayerPolicyFromMeanFieldPolicy(policy.Policy):
  """Policy that apply mean field policy to N player game for dynamic routing.

  Attributes:
    _mfg_policy: the mean field game policy.
    _mfg_empty_state: an empty mfg state to clone for the state conversion.
    _state_memoization: dictionary to memoize conversion of N player game state
      string representation to the corresponding MFG state.
  """

  def __init__(self, game: dynamic_routing.DynamicRoutingGame,
               mfg_policy: policy.Policy):
    """Initializes a uniform random policy for all players in the game."""
    super().__init__(game, list(range(game.num_players())))
    self._mfg_policy = mfg_policy
    self._mfg_empty_state = _create_empty_mfg_state(game)
    self._state_memoization = {}

  def _convert_state_to_mean_field_state(
      self, n_player_state: dynamic_routing.DynamicRoutingGameState,
      player_id: int) -> mean_field_routing_game.MeanFieldRoutingGameState:
    """Convert a N player state to a mean field state."""
    assert player_id >= 0, "player_id should be a positive integer."
    # create a string key for N player game.
    state_key = (str(n_player_state), player_id)
    mfg_state = self._state_memoization.get(state_key)
    if mfg_state is not None:
      return mfg_state
    mfg_state = self._mfg_empty_state.clone()
    # pylint:disable=protected-access
    mfg_state._is_chance_init = False
    mfg_state._current_time_step = n_player_state._current_time_step
    mfg_state._is_terminal = n_player_state._is_terminal
    mfg_state._player_id = pyspiel.PlayerId.DEFAULT_PLAYER_ID
    mfg_state._waiting_time = n_player_state._waiting_times[player_id]
    mfg_state._vehicle_at_destination = (
        player_id in n_player_state._vehicle_at_destination)
    mfg_state._vehicle_destination = n_player_state._vehicle_destinations[
        player_id]
    mfg_state._vehicle_final_travel_time = (
        n_player_state._vehicle_final_travel_times[player_id])
    mfg_state._vehicle_location = n_player_state._vehicle_locations[player_id]
    mfg_state._vehicle_without_legal_action = (
        player_id in n_player_state._vehicle_without_legal_actions)
    # pylint:enable=protected-access
    self._state_memoization[state_key] = mfg_state
    return mfg_state

  def action_probabilities(self,
                           state: dynamic_routing.DynamicRoutingGameState,
                           player_id=None) -> Dict[int, float]:
    """Returns the mean field action to apply in the N player state.

    Args:
      state: An N player dynamic routing game state.
      player_id: the player id for which we want an action. Should be given to
        the function.

    Returns:
      A `dict` of `{action: probability}` for the specified player in the
      supplied state.
    """
    assert player_id is not None
    mfg_state = self._convert_state_to_mean_field_state(state, player_id)
    # Due to memoization, action_probabilities should not change mfg_state. In
    # case action_probabilities changes mfg_state, then mfg_state.clone() should
    # be passed to the function.
    return self._mfg_policy.action_probabilities(mfg_state)
