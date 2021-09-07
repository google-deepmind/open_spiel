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
"""Dynamic routing game to be solved with mean field game."""

from typing import Any, Dict, List, Mapping

import pyspiel
from open_spiel.python import policy
from open_spiel.python.games.dynamic_routing import dynamic_routing_game
from open_spiel.python.games.dynamic_routing import dynamic_routing_game_utils
from open_spiel.python.games.dynamic_routing import mean_field_routing_game


class DynamicRoutingToMeanFieldGame(dynamic_routing_game.DynamicRoutingGame):
  """TODO

  Attributes:
    od_demand:
    mfg_game:
  """
  def __init__(
    self, params: Mapping[str, Any] = None,
    network: dynamic_routing_game_utils.Network = None,
    vehicles: List[dynamic_routing_game_utils.Vehicle] = None,
    max_num_time_step: int = 2,
    perform_sanity_checks: bool = True
  ):
    super().__init__(
      params, network, vehicles, max_num_time_step, perform_sanity_checks)

    od_demand = {}
    for vehicle in self._vehicles:
      key = (vehicle.origin, vehicle.destination, vehicle.departure_time)
      if key not in od_demand:
        od_demand[key] = 0
      od_demand[key] += 1
    self.od_demand = []
    for (origin, destination, departure_time), counts in od_demand.items():
      self.od_demand.append(
        dynamic_routing_game_utils.OriginDestinationDemand(
          origin, destination, departure_time, counts))
    self.mfg_game = mean_field_routing_game.MeanFieldRoutingGame(
      network=network, od_demand=self.od_demand,
      max_num_time_step=max_num_time_step,
      perform_sanity_checks=perform_sanity_checks)

  def new_initial_state(self) -> "DynamicRoutingToMeanFieldGameState":
    """Returns the state corresponding to the start of a game."""
    return DynamicRoutingToMeanFieldGameState(self, self._vehicles)


class DynamicRoutingToMeanFieldGameState(
  dynamic_routing_game.DynamicRoutingGameState
):
  """TODO"""

  def __init__(self, game, vehicles):
    super().__init__(game, vehicles)
    self.__state_memoization = {}
  #   self.current_player_ = 0
  #   self.actions_ = [0 for _ in range(self.get_game().num_players())]
  # TODO: generate all states and memoize them here?

  def convert_state_to_mean_field_state(self, player_id):
    """Convert a N player state to a mean field state."""
    # if player_id is None:
    #   player_id = self.current_player_
    assert player_id >= 0, "player_id should be a positive integer."
    # create a string key for N player game.
    state_key = (str(self), player_id)
    if state_key in self.__state_memoization:
      return self.__state_memoization[state_key].clone()
    mfg_state = mean_field_routing_game.MeanFieldRoutingGameState(
      self.get_game().mfg_game, self.get_game().od_demand)
    mfg_state._is_chance_init = False
    mfg_state._current_time_step = self._current_time_step
    mfg_state._is_terminal = self._is_terminal
    mfg_state._player_id = pyspiel.PlayerId.DEFAULT_PLAYER_ID
    mfg_state._can_vehicle_move = self._can_vehicles_move[player_id]
    mfg_state._vehicle_at_destination = (
      player_id in self._vehicle_at_destination)
    mfg_state._vehicle_destination = self._vehicle_destinations[player_id]
    mfg_state._vehicle_final_travel_time = self._vehicle_final_travel_times[
      player_id]
    mfg_state._vehicle_location = self._vehicle_locations[player_id]
    mfg_state._vehicle_without_legal_action = (
      player_id in self._vehicle_without_legal_actions)
    self.__state_memoization[state_key] = mfg_state
    return mfg_state.clone()

  # def is_simultaneous_node(self):
  #   return not self._is_chance and not self._is_terminal

  # def current_player(self):
  #   player_simultaneous_game = super().current_player()
  #   if player_simultaneous_game == pyspiel.PlayerId.SIMULTANEOUS:
  #     return self.current_player_
  #   return player_simultaneous_game

  # def _apply_action(self, action: int):
  #   """Wrap apply_action and apply_actions of simultaneous game in a turn based
  #   apply_action function."""
  #   if self.is_chance_node():
  #     return super()._apply_action(action)
  #   else:
  #     self.actions_[self.current_player_] = action
  #     self.current_player_ += 1
  #     if self.current_player_ >= self.get_game().num_players():
  #       self.current_player_ = 0
  #       self._apply_actions(self.actions_)
  #       self.actions_ = [0 for _ in range(self.get_game().num_players())]


class DerivedNPlayerPolicyFromMeanFieldPolicy(policy.Policy):
  """Policy where the action distribution is uniform over all legal actions.

  This is computed as needed, so can be used for games where a tabular policy
  would be unfeasibly large, but incurs a legal action computation every time.
  """
  # TODO: write rollback simultaneous game as a turn based game.

  def __init__(self, game: DynamicRoutingToMeanFieldGame, mfg_policy):
    """Initializes a uniform random policy for all players in the game.

    Args:
        game: N player game.
        mfg_policy: mean field game policy.
    """
    self.num_players = game.num_players()
    super().__init__(game, list(range(self.num_players)))
    self.mfg_policy = mfg_policy
    self.current_player_ = 0

  def action_probabilities(
    self, state: DynamicRoutingToMeanFieldGameState, player_id=None
  ) -> Dict[int, float]:
    """Returns a uniform random policy for a player in a state.

    Args:
      state: A N player game state.
      player_id: Optional, the player id for which we want an action. Optional
        unless this is a simultaneous state at which multiple players can act.

    Returns:
      A `dict` of `{action: probability}` for the specified player in the
      supplied state. This will contain all legal actions, each with the same
      probability, equal to 1 / num_legal_actions.
    """
    # if player_id is not None:
    assert player_id is not None
    mfg_state = state.convert_state_to_mean_field_state(player_id)
    return self.mfg_policy.action_probabilities(mfg_state)
    # mfg_state = state.convert_state_to_mean_field_state(None)
    # return self.mfg_policy.action_probabilities(mfg_state)
