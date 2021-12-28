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
"""Implementation of dynamic routing game.

The game is derived from https://arxiv.org/abs/2110.11943.
This dynamic routing game models the evolution of N vehicles in a road network.
The vehicles are described by their current link location, the time they have to
spend on the link before exiting it, and their destination. The action of a
vehicle is the successor link they want to reach when exiting a given link.
Actions are encoded as integer from 0 to K. Action 0 encodes not being able to
move on a successor link because the waiting time of the player is still
positive. Actions 1 to K correspond to the indices of the network links. Legal
actions for a player on link l, with a negative waiting time are the indices of
the successors link of l. When arriving on a link, the waiting time of the
player is assign based on the number of players on the link at this time. Over
time steps, the waiting time linearly decrease until it is negative, the vehicle
moves to a successor link and the waiting time get reassigned.
The cost of the vehicle is its travel time, it could be seen as a running cost
where +1 is added to the cost at any time step the vehicle is not on its
destination.
This dynamic routing game is a mesoscopic traffic model with explicit congestion
dynamics where vehicle minimizes their travel time.

The game is defined by:
- a network given by the class Network.
- a list of vehicles given by the class Vehicle.

The current game is implementated as a N player game. However this game can also
be extended to a mean field game, implemented as python_mfg_dynamic_routing.
"""

from typing import Any, Iterable, List, Mapping, Optional, Set

import numpy as np
from open_spiel.python.games import dynamic_routing_data
from open_spiel.python.games import dynamic_routing_utils
from open_spiel.python.observation import IIGObserverForPublicInfoGame
import pyspiel

_DEFAULT_PARAMS = {
    "max_num_time_step": 10,
    "time_step_length": 0.5,
    "players": -1
}
_GAME_TYPE = pyspiel.GameType(
    short_name="python_dynamic_routing",
    long_name="Python Dynamic Routing Game",
    dynamics=pyspiel.GameType.Dynamics.SIMULTANEOUS,
    chance_mode=pyspiel.GameType.ChanceMode.DETERMINISTIC,
    information=pyspiel.GameType.Information.PERFECT_INFORMATION,
    utility=pyspiel.GameType.Utility.GENERAL_SUM,
    reward_model=pyspiel.GameType.RewardModel.REWARDS,
    max_num_players=100,
    min_num_players=0,
    provides_information_state_string=True,
    provides_information_state_tensor=True,
    provides_observation_string=True,
    provides_observation_tensor=True,
    default_loadable=True,
    provides_factored_observation_string=True,
    parameter_specification=_DEFAULT_PARAMS)


class DynamicRoutingGame(pyspiel.Game):
  """Implementation of dynamic routing game.

  At each simultaneous-move time, each vehicle/player with negative waiting time
  chooses on which successor link they would like to go. When arriving on the
  link, a waiting time is assigned to the player based on the count of players
  on the link, after everyone has moved to their successors link. One vehicle
  travel time is equal to the time step when they first reach their destination.
  See module docstring for more information.

  Attributes inherited from GameInfo:
    max_chance_outcome: 0, the game is deterministic.
    max_game_length: maximum number of time step played. Passed during
      construction.
    max_utility: maximum utility is the opposite of the minimum travel time. Set
      to 0.
    min_utility: minimum utility is the opposite of the maximum travel time. Set
      to - max_game_length - 1.
    num_distinct_actions: maximum number of possible actions. This is equal to
      the number of links + 1 (corresponding to having no possible action
      _NO_POSSIBLE_ACTION).
    num_players: the number of vehicles. Choosen during by the constructor as
      the number of vehicles.
  Attributes:
    network: the network of the game.
    _vehicles: a list of the vehicle. Their origin and their destination should
      be road sections of the game. The number of vehicles in the list sets the
      num_players attribute.
    time_step_length: size of the time step, used to convert travel times into
      number of game time steps.
    perform_sanity_checks: if true, sanity checks are done during the game,
      should be set to false to speed up the game.
  """
  network: dynamic_routing_utils.Network
  _vehicles: List[dynamic_routing_utils.Vehicle]
  perform_sanity_checks: bool
  time_step_length: float

  def __init__(
      self,
      params: Mapping[str, Any],
      network: Optional[dynamic_routing_utils.Network] = None,
      vehicles: Optional[List[dynamic_routing_utils.Vehicle]] = None,
      perform_sanity_checks: bool = True,
  ):
    """Initiliaze the game.

    Args:
      params: game parameters. It should define max_num_time_step and
        time_step_length.
      network: the network of the game.
      vehicles: a list of the vehicle. Their origin and their destination should
        be road sections of the game. The number of vehicles in the list sets
        the num_players attribute.
      perform_sanity_checks: set the perform_sanity_checks attribute.
    """
    max_num_time_step = params["max_num_time_step"]
    time_step_length = params["time_step_length"]
    self.network = network if network else dynamic_routing_data.BRAESS_NETWORK
    self._vehicles = (
        vehicles
        if vehicles else dynamic_routing_data.BRAESS_NETWORK_VEHICLES_DEMAND)
    self.network.check_list_of_vehicles_is_correct(self._vehicles)
    self.perform_sanity_checks = perform_sanity_checks
    self.time_step_length = time_step_length
    game_info = pyspiel.GameInfo(
        num_distinct_actions=self.network.num_actions(),
        max_chance_outcomes=0,
        num_players=len(self._vehicles),
        min_utility=-max_num_time_step - 1,
        max_utility=0,
        max_game_length=max_num_time_step)
    super().__init__(_GAME_TYPE, game_info, params if params else {})

  def new_initial_state(self) -> "DynamicRoutingGameState":
    """Returns the state corresponding to the start of a game."""
    return DynamicRoutingGameState(self, self._vehicles, self.time_step_length)

  def make_py_observer(self, iig_obs_type=None, params=None):
    """Returns a NetworkObserver object used for observing game state."""
    if ((iig_obs_type is None) or
        (iig_obs_type.public_info and not iig_obs_type.perfect_recall)):
      return NetworkObserver(self.num_players(), self.max_game_length())
    return IIGObserverForPublicInfoGame(iig_obs_type, params)


class DynamicRoutingGameState(pyspiel.State):
  """State of the DynamicRoutingGame.

  One player is equal to one vehicle.
  See docstring of the game class and of the file for more information.
  Attributes:
    _current_time_step: current time step of the game.
    _is_terminal: boolean that encodes weither the game is over.
    _time_step_length: size of the time step, used to convert travel times into
      number of game time steps.
    _vehicle_at_destination: set of vehicles that have reached their
      destinations. When a vehicle has reached its destination but the game is
      not finished, it cannot do anything.
    _vehicle_destinations: the destination of each vehicle.
    _vehicle_final_travel_times: the travel times of each vehicle, the travel is
      either 0 if the vehicle is still in the network or its travel time if the
      vehicle has reached its destination.
    _vehicle_locations: current location of the vehicles as a network road
      section.
    _vehicle_without_legal_actions: list of vehicles without legal actions at
      next time step. This is required because if no vehicle has legal actions
      for a simultaneous node then an error if raised.
    _waiting_times: time that each vehicle should wait before being able to move
      to the next road section.
  """
  _current_time_step: int
  _is_terminal: bool
  _time_step_length: float
  _vehicle_at_destination: Set[int]
  _vehicle_destinations: List[str]
  _vehicle_final_travel_times: List[float]
  _vehicle_locations: List[str]
  _vehicle_without_legal_actions: Set[int]
  _waiting_times: List[int]

  def __init__(self, game: DynamicRoutingGame,
               vehicles: Iterable[dynamic_routing_utils.Vehicle],
               time_step_length: float):
    """Constructor; should only be called by Game.new_initial_state."""
    super().__init__(game)
    self._current_time_step = 0
    self._is_terminal = False
    self._time_step_length = time_step_length
    self._vehicle_at_destination = set()
    self._vehicle_destinations = [vehicle.destination for vehicle in vehicles]
    self._vehicle_final_travel_times = [0.0 for _ in vehicles]
    self._vehicle_locations = [vehicle.origin for vehicle in vehicles]
    self._vehicle_without_legal_actions = set()
    self._waiting_times = [
        int(veh._departure_time / self._time_step_length) for veh in vehicles
    ]
    self.running_cost = [0 for vehicle in vehicles]

  @property
  def current_time_step(self) -> int:
    """Return current time step."""
    return self._current_time_step

  def current_player(self) -> pyspiel.PlayerId:
    """Returns the current player.

    If the game is over, TERMINAL is returned. If the game is at a chance
    node then CHANCE is returned. Otherwise SIMULTANEOUS is returned.
    """
    if self._is_terminal:
      return pyspiel.PlayerId.TERMINAL
    return pyspiel.PlayerId.SIMULTANEOUS

  def assert_valid_player(self, vehicle: int):
    """Assert that a vehicle as a int between 0 and num_players."""
    assert isinstance(vehicle, int), f"{vehicle} is not a int."
    assert vehicle >= 0, f"player: {vehicle}<0."
    assert vehicle < self.get_game().num_players(), (
        f"player: {vehicle} >= num_players: {self.get_game().num_players()}")

  def _legal_actions(self, vehicle: int) -> List[int]:
    """Return the legal actions of the vehicle.

    Legal actions are the succesor road section of the vehicle current road
    section.
    Args:
      vehicle: the vehicle id.

    Returns:
      list_legal_actions: a list of legal actions. If the game is finished then
        the list is empty. If the vehicle is at its destination, has a positive
        waiting time or if it is on a node without successors then an empty list
        is returned. Otherwise the list of successors nodes of the current
        vehicle location is returned.
    """
    if self._is_terminal:
      return []
    if self.get_game().perform_sanity_checks:
      self.assert_valid_player(vehicle)
    if vehicle in self._vehicle_without_legal_actions:
      # If the vehicle is at destination it cannot do anything.
      return [dynamic_routing_utils.NO_POSSIBLE_ACTION]
    if self._waiting_times[vehicle] > 0:
      return [dynamic_routing_utils.NO_POSSIBLE_ACTION]
    _, end_section_node = dynamic_routing_utils._road_section_to_nodes(  # pylint:disable=protected-access
        self._vehicle_locations[vehicle])
    successors = self.get_game().network.get_successors(end_section_node)
    if successors:
      assert isinstance(successors, Iterable)
      actions = [
          self.get_game().network.get_action_id_from_movement(
              end_section_node, d) for d in successors
      ]
      if self.get_game().perform_sanity_checks:
        map(self.get_game().network.assert_valid_action, actions)
      return sorted(actions)
    return []

  def _apply_actions(self, actions: List[int]):
    """Applies the specified action to the state.

    For each vehicle's action, if the vehicle is not at a sink node, if the
    action is valid and if the waiting time is negative, then the vehicle will
    move to the successor link corresponding to its action.
    The function then detects if the vehicle has reached its destination or
    a sink node and updates _vehicle_at_destination,
    _vehicle_without_legal_actions and _vehicle_final_travel_times
    accordingly.
    The function then assigns waiting for each vehicle that have moved based on
    the new volume of cars on the link they reach.
    The function evolves the time and checks if the game is finished.
    Args:
        actions: the action chosen by each vehicle.
    """
    if self.get_game().perform_sanity_checks:
      assert not self._is_terminal
    if self.get_game().perform_sanity_checks:
      assert isinstance(actions, Iterable)
      assert len(actions) == self.get_game().num_players(), (
          f"Each player does not have an actions. Actions has {len(actions)} "
          f"elements, it should have {self.get_game().num_players()}.")
    for vehicle_id, action in enumerate(actions):
      if vehicle_id not in self._vehicle_at_destination:
        self.running_cost[vehicle_id] += self._time_step_length
      # Has the vehicle already reached a sink node?
      if vehicle_id in self._vehicle_without_legal_actions:
        if self.get_game().perform_sanity_checks:
          assert action == dynamic_routing_utils.NO_POSSIBLE_ACTION, (
              f"{action} should be {dynamic_routing_utils.NO_POSSIBLE_ACTION}.")
        continue
      if self._waiting_times[vehicle_id] > 0:
        continue
      if self.get_game().perform_sanity_checks:
        self.get_game().network.assert_valid_action(
            action, self._vehicle_locations[vehicle_id])
      self._vehicle_locations[vehicle_id] = (
          self.get_game().network.get_road_section_from_action_id(action))
      if (self._vehicle_locations[vehicle_id] ==
          self._vehicle_destinations[vehicle_id]):
        self._vehicle_final_travel_times[vehicle_id] = self._current_time_step
        self._vehicle_at_destination.add(vehicle_id)
        self._vehicle_without_legal_actions.add(vehicle_id)
      # Will the vehicle have a legal action for next time step?
      elif self.get_game().network.is_location_at_sink_node(
          self._vehicle_locations[vehicle_id]):
        self._vehicle_without_legal_actions.add(vehicle_id)
    self._current_time_step += 1
    volumes = {}
    for road_section in self._vehicle_locations:
      if road_section not in volumes:
        volumes[road_section] = 0
      # Each vehicle has a weight a one.
      volumes[road_section] += 1
    for vehicle_id, _ in enumerate(actions):
      # Has the vehicle already reached a sink node?
      if vehicle_id in self._vehicle_without_legal_actions:
        continue
      if self._waiting_times[vehicle_id] > 0:
        self._waiting_times[vehicle_id] -= 1
      else:
        self._waiting_times[vehicle_id] = int(self.get_game(
        ).network.get_travel_time(self._vehicle_locations[vehicle_id], volumes[
            self._vehicle_locations[vehicle_id]]) / self._time_step_length -
                                              1.0)
    # Is the game finished?
    if (self._current_time_step >= self.get_game().max_game_length() or len(
        self._vehicle_without_legal_actions) == self.get_game().num_players()):
      self._is_terminal = True
      for vehicle_id in range(self.get_game().num_players()):
        if vehicle_id not in self._vehicle_at_destination:
          self._vehicle_final_travel_times[vehicle_id] = (
              self._current_time_step)

  def _action_to_string(self, player, action) -> str:
    """Action -> string."""
    if self.get_game().perform_sanity_checks:
      self.assert_valid_player(player)
    if action == dynamic_routing_utils.NO_POSSIBLE_ACTION:
      return f"Vehicle {player} reach a sink node or its destination."
    if self.get_game().perform_sanity_checks:
      self.get_game().network.assert_valid_action(action)
    return (f"Vehicle {player} would like to move to "
            f"{self.get_game().network.get_road_section_from_action_id(action)}"
            ".")

  def is_terminal(self) -> bool:
    """Returns True if the game is over."""
    return self._is_terminal

  def rewards(self):
    """Reward at the previous step."""
    if self._is_terminal or self._current_time_step == 0:
      return [0 for _ in self._vehicle_locations]
    reward = [-self._time_step_length for _ in self._vehicle_locations]
    for vehicle in self._vehicle_at_destination:
      reward[vehicle] = 0
    return reward

  def returns(self) -> List[float]:
    """Total reward for each player over the course of the game so far."""
    if not self._is_terminal:
      returns = [
          -self._time_step_length * self.current_time_step
          for _ in self._vehicle_locations
      ]
      for vehicle in self._vehicle_at_destination:
        returns[vehicle] = -(
            self._vehicle_final_travel_times[vehicle] * self._time_step_length)
      return returns
    returns = [
        -travel_time * self._time_step_length
        for travel_time in self._vehicle_final_travel_times
    ]
    return returns

  def get_current_vehicle_locations(self) -> List[str]:
    """Get vehicle locations for debug purposes."""
    return self._vehicle_locations

  def get_location_as_int(self, vehicle: int) -> int:
    """Get the vehicle location."""
    origin, destination = dynamic_routing_utils._road_section_to_nodes(  # pylint:disable=protected-access
        self._vehicle_locations[vehicle])
    return self.get_game().network.get_action_id_from_movement(
        origin, destination)

  def get_current_vehicle_locations_as_int(self) -> List[int]:
    """Get locations of all vehicles for the observation tensor."""
    return [
        self.get_location_as_int(x)
        for x in range(self.get_game().num_players())
    ]

  def __str__(self) -> str:
    """String for debug purposes. No particular semantics are required."""
    if self._is_terminal:
      time = f"{self._current_time_step}, game finished."
    else:
      time = f"{self._current_time_step}"
    return (f"Vehicle locations: {self._vehicle_locations}, "
            f"time: {time}, waiting_time={self._waiting_times}.")


class NetworkObserver:
  """Network observer used by the learning algorithm.

  The state string is the state history string. The state tensor is an array
  of size max_game_length, num_players where each element is the location of
  the vehicle at this time.
  Attributes:
    dict: dictionary {"observation": tensor}.
    tensor: list of location for each time step.
  """

  def __init__(self, num_vehicles: int, num_time: int):
    """Initializes an empty observation tensor."""
    shape = (num_time + 1, num_vehicles + 1)
    self.tensor = np.zeros(np.prod(shape), np.float32)
    self.dict = {"observation": np.reshape(self.tensor, shape)}

  def set_from(self, state, player):
    """Update the state tensor.

    Put the locations of each players in the tensor row corresponding to
    the current time step. Insert the current player location at the
    beginning of the row.
    Args:
      state: the state,
      player: the player.
    """
    vehicles = state.get_current_vehicle_locations_as_int()
    vehicles.insert(0, state.get_location_as_int(player))
    self.dict["observation"][state.current_time_step, :] = vehicles

  def string_from(self, state, player):
    """Return the state history string."""
    return f"{player}: {state.history_str()}"


# Register the game with the OpenSpiel library
pyspiel.register_game(_GAME_TYPE, DynamicRoutingGame)
