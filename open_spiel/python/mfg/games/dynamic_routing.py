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
"""Implementation of a mean field routing game.

The game is derived from https://arxiv.org/abs/2110.11943.
It is the extension of the dynamic routing game python_dynamic_routing.
The list of vehicles decribing the N player of the dynamic routing game is
replaced by a list of OriginDestinationDemand. One OriginDestinationDemand
corresponds to one population of vehicles (with the same origin, destination and
departure time).

This game is a variant of the mean field route choice game as the vehicle
movement depends on the current network congestion. In the mean field route
choice game, the number of time step to reach the destination is constant and
does not depend on the network congestion, neither of the vehicle cost function.
In the dynamic driving and routing games the vehicle choose its speed to travel
on each link in order to minimize its cost function. Therefore the congestion is
encoded in the cost function.

More context can be found on the docstring of the python_dynamic_routing class.
"""
import functools
from typing import Any, Iterable, List, Mapping, Optional, Tuple

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
    short_name="python_mfg_dynamic_routing",
    long_name="Python Mean Field Routing Game",
    dynamics=pyspiel.GameType.Dynamics.MEAN_FIELD,
    chance_mode=pyspiel.GameType.ChanceMode.EXPLICIT_STOCHASTIC,
    information=pyspiel.GameType.Information.PERFECT_INFORMATION,
    utility=pyspiel.GameType.Utility.GENERAL_SUM,
    reward_model=pyspiel.GameType.RewardModel.REWARDS,
    max_num_players=1,
    min_num_players=1,
    provides_information_state_string=True,
    provides_information_state_tensor=True,
    provides_observation_string=True,
    provides_observation_tensor=True,
    default_loadable=True,
    provides_factored_observation_string=True,
    parameter_specification=_DEFAULT_PARAMS)

WAITING_TIME_NOT_ASSIGNED = -1


@functools.lru_cache(maxsize=None)
def _state_to_str(
    is_chance_init: bool,
    location: str,
    time_step: int,
    player_id: int,
    waiting_time: int,
    destination: str,
    final_travel_time: float,
) -> str:
  """Convert the state to a string representation.

  As the string representation will be used in dictionaries for various
  algorithms that computes the state value, expected return, best response or
  find the mean field Nash equilibrium.
  The state is uniquely define by the current time, the type of node
  (decision, mean field or chance), the vehicle location, its destination and
  its waiting time.
  Args:
    is_chance_init: True if at chance initialization.
    location: the location of the representative player.
    time_step: the current time step.
    player_id: the current node type as a player id.
    waiting_time: the representative player waiting time.
    destination: the destination of the representative player.
    final_travel_time: time of arrival.

  Returns:
    state_string: string representing uniquely the mean field game.
  """
  if is_chance_init:
    return "initial chance node"
  if player_id == pyspiel.PlayerId.DEFAULT_PLAYER_ID:
    time = str(time_step)
  elif player_id == pyspiel.PlayerId.MEAN_FIELD:
    time = f"{time_step}_mean_field"
  elif player_id == pyspiel.PlayerId.CHANCE:
    time = f"{time_step}_chance"
  else:
    raise ValueError(
        "Player id should be DEFAULT_PLAYER_ID, MEAN_FIELD or CHANCE")
  if final_travel_time:
    return (f"Arrived at {location}, with travel time "
            f"{final_travel_time}, t={time}")
  return (f"Location={location}, waiting_time={waiting_time},"
          f" t={time}, destination='{destination}'")


class MeanFieldRoutingGame(pyspiel.Game):
  """Implementation of mean field routing game.

  The representative vehicle/player is represented as a tuple current location,
  current waiting time and destination. When the waiting time is negative, the
  vehicle choose on with successor link it would like to go. When arriving on
  the link, a waiting time is assigned to the player based on the distribution
  of players on the link. The vehicle travel time is equal to the time step when
  they first reach their destination. See module docstring for more information.

  Attributes inherited from GameInfo:
    max_chance_outcome: 0, the game is deterministic.
    max_game_length: maximum number of time step played. Passed during
      construction.
    max_utility: maximum utility is the opposite of the minimum travel
      time. Set to 0.
    min_utility: minimum utility is the opposite of the maximum travel
      time. Set to - max_game_length - 1.
    num_distinct_actions: maximum number of possible actions. This is
      equal to the number of links + 1 (corresponding to having no
      possible action _NO_POSSIBLE_ACTION).
    num_players: the number of vehicles. Should be 1 as this mean field
      game is a one population game.
  Attributes:
    network: the network of the game.
    od_demand: a list of the vehicle. Their origin and their destination should
      be road sections of the game.
    time_step_length: size of the time step, used to convert travel times into
      number of game time steps.
    perform_sanity_checks: if true, sanity checks are done during the game,
      should be set to false to speed up the game.
    total_num_vehicle: total number of vehicles as the sum of the od_demand.
    chance_outcomes: chance outcomes based on the initial probability
      distribution and their probabilities.
  """
  network: dynamic_routing_utils.Network
  od_demand: List[dynamic_routing_utils.OriginDestinationDemand]
  perform_sanity_checks: bool
  time_step_length: float

  def __init__(self,
               params: Mapping[str, Any],
               network: Optional[dynamic_routing_utils.Network] = None,
               od_demand: Optional[List[
                   dynamic_routing_utils.OriginDestinationDemand]] = None,
               perform_sanity_checks: bool = True):
    """Initiliaze the game.

    Args:
      params: game parameters. It should define max_num_time_step and
        time_step_length.
      network: the network of the game.
      od_demand: a list of the vehicle. Their origin and their destination
        should be road sections of the game.
      perform_sanity_checks: set the perform_sanity_checks attribute.
    """
    max_num_time_step = params["max_num_time_step"]
    time_step_length = params["time_step_length"]
    self.network = network if network else dynamic_routing_data.BRAESS_NETWORK
    self.od_demand = (
        od_demand
        if od_demand else dynamic_routing_data.BRAESS_NETWORK_OD_DEMAND)
    self.network.check_list_of_od_demand_is_correct(self.od_demand)
    self.perform_sanity_checks = perform_sanity_checks
    self.time_step_length = time_step_length
    self.total_num_vehicle = sum(
        [od_demand_item.counts for od_demand_item in self.od_demand])
    self.chance_outcomes = [(i, od_demand_item.counts / self.total_num_vehicle)
                            for i, od_demand_item in enumerate(self.od_demand)]
    game_info = pyspiel.GameInfo(
        num_distinct_actions=self.network.num_actions(),
        max_chance_outcomes=len(self.od_demand),
        num_players=1,
        min_utility=-max_num_time_step - 1,
        max_utility=0,
        max_game_length=max_num_time_step)
    super().__init__(_GAME_TYPE, game_info, params if params else {})

  def new_initial_state(self) -> "MeanFieldRoutingGameState":
    """Returns the state corresponding to the start of a game."""
    return MeanFieldRoutingGameState(self, self.time_step_length)

  def make_py_observer(self, iig_obs_type=None, params=None):
    """Returns a NetworkObserver object used for observing game state."""
    if ((iig_obs_type is None) or
        (iig_obs_type.public_info and not iig_obs_type.perfect_recall)):
      return NetworkObserver(self.network.num_actions(), self.max_game_length())
    return IIGObserverForPublicInfoGame(iig_obs_type, params)

  def max_chance_nodes_in_history(self):
    """Maximun chance nodes in game history."""
    return self.max_game_length() + 1

  def get_road_section_as_int(self, section: Optional[str]) -> int:
    """Returns the integer representation of the road section."""
    if section is None:
      return 0
    start_node, end_node = (
        dynamic_routing_utils._road_section_to_nodes(section))  # pylint:disable=protected-access
    return self.network.get_action_id_from_movement(start_node, end_node)


class MeanFieldRoutingGameState(pyspiel.State):
  """State of the DynamicRoutingGame.

  One player is equal to one vehicle.
  See docstring of the game class and of the file for more information.
  Attributes:
    _current_time_step: current time step of the game.
    _is_chance_init: boolean that encodes weither the current node is the
      initial chance node.
    _is_terminal: boolean that encodes weither the game is over.
    _max_travel_time: int that encodes maximum travel time on any link in number
      of time steps. Needed to enumerate all the possible state of a vehicle
      being on a link to compute volume of cars on the link.
    _max_waiting_time: maximum time a vehicle can wait on a time. This is done
      in order to limit the number of possible state with a vehicle on a
      specific link.
    _normed_density_on_vehicle_link: density of vehicles on the link that is
      used by the representative vehicle. This is given by the mean field
      distribution.
    _time_step_length: size of the time step, used to convert travel times into
      number of game time steps.
    _vehicle_at_destination: boolean that encodes if the representative vehicle
      has reached its destination.
    _vehicle_destination: the destination of the representative vehicle
      corresponding to this state. It is associated to the representative
      vehicle after the initial chance node according to the od_demand
      distribution.
    _vehicle_final_travel_time: the travel time of the representative vehicle,
      the travel is either 0 if the vehicle is still in the network or its
      travel time if the vehicle has reached its destination.
    _vehicle_location: current location of the vehicle as a network road
      section.
    _vehicle_without_legal_action: boolean that encodes if the representative
      vehicle has reach a sink node, meaning that it will not be able to move
      anymore.
    _waiting_time: time that the vehicle has to wait before moving to the next
      link (equal to the link travel time when the vehicle just reached the
      link).
  """
  _current_time_step: int
  _is_chance_init: bool
  _is_terminal: bool
  _max_travel_time: int
  _max_waiting_time: int
  _normed_density_on_vehicle_link: float
  _time_step_length: float
  _vehicle_at_destination: bool
  _vehicle_destination: Optional[str]
  _vehicle_final_travel_time: float
  _vehicle_location: Optional[str]
  _vehicle_without_legal_action: bool
  _waiting_time: int

  def __init__(self, game: MeanFieldRoutingGame, time_step_length: float):
    """Constructor; should only be called by Game.new_initial_state."""
    super().__init__(game)
    self._current_time_step = 0
    self._is_chance_init = True  # is true for the first state of the game.
    self._is_terminal = False
    if self.get_game().perform_sanity_checks:
      assert game.num_players() == 1, (
          "This mean field routing game should have a unique player.")
    self._player_id = pyspiel.PlayerId.CHANCE
    self._time_step_length = time_step_length
    self._vehicle_at_destination = False
    self._vehicle_final_travel_time = 0.0
    self._vehicle_without_legal_action = False
    self._vehicle_location = None
    self._vehicle_destination = None
    self._max_travel_time = self.get_game().max_game_length()
    # TODO(cabannes): cap maximum link waiting time to faster simulations.
    self._max_waiting_time = self._max_travel_time
    self._waiting_time = WAITING_TIME_NOT_ASSIGNED

  @property
  def current_time_step(self) -> int:
    """Return current time step."""
    return self._current_time_step

  def current_player(self) -> pyspiel.PlayerId:
    """Returns the current player."""
    if self._is_terminal:
      return pyspiel.PlayerId.TERMINAL
    return self._player_id

  def state_to_str(self,
                   location: str,
                   time_step: int,
                   player_id: int = pyspiel.PlayerId.DEFAULT_PLAYER_ID,
                   waiting_time: int = 0,
                   destination: str = ""):
    """Convert the state to a string representation."""
    return _state_to_str(
        self._is_chance_init,
        location,
        time_step,
        player_id,
        waiting_time,
        destination or self._vehicle_destination,
        self._vehicle_final_travel_time,
    )

  def distribution_support(self) -> List[str]:
    """Returns the state that should be used for update_distribution.

    The distribution of the vehicle is used to determined the number of
    cars on the same link of the representative vehicle in order to define
    the waiting time of the representative vehicle when joining a link.
    Therefore, only the states corresponding to be on the link of the
    representative vehicle at this current time are useful.
    Returns:
      list of the two state: being on the link of the representative vehicle at
        the current time and being stuck in traffic or not.
    """
    if self._vehicle_without_legal_action:
      return []
    od_demand = self.get_game().od_demand
    dist = [
        self.state_to_str(  # pylint:disable=g-complex-comprehension
            self._vehicle_location,
            self._current_time_step,
            player_id=pyspiel.PlayerId.MEAN_FIELD,
            waiting_time=waiting_time,
            destination=destination)
        for waiting_time in range(WAITING_TIME_NOT_ASSIGNED,
                                  self._max_travel_time)
        for destination in {od._destination for od in od_demand}  # pylint:disable=protected-access
    ]
    assert len(set(dist)) == len(dist), (
        f"Distribution should not have duplicated states: {dist}.")
    return dist

  def update_distribution(self, distribution: List[float]):
    """Get the number of cars on the same link as the representative player.

    _normed_density_on_vehicle_link stores the number of cars on the link
    where the representative player is.
    Args:
      distribution: the probability for a vehicle to be in the states in
        distribution_support. The distribution is a list of probabilities.
    """
    game = self.get_game()
    if game.perform_sanity_checks:
      if self._player_id != pyspiel.PlayerId.MEAN_FIELD:
        raise ValueError(("update_distribution should only be called at"
                          " a MEAN_FIELD state."))
    self._player_id = pyspiel.PlayerId.DEFAULT_PLAYER_ID
    if not self._vehicle_without_legal_action:
      self._normed_density_on_vehicle_link = sum(distribution)
      if game.perform_sanity_checks:
        assert 0 <= self._normed_density_on_vehicle_link <= 1 + 1e-4, (
            f"{self._normed_density_on_vehicle_link} is not in [0, 1].")
      if self._waiting_time == WAITING_TIME_NOT_ASSIGNED:
        volume = (game.total_num_vehicle * self._normed_density_on_vehicle_link)
        self._waiting_time = int(
            game.network.get_travel_time(self._vehicle_location, volume) /
            self._time_step_length) - 1
        self._waiting_time = max(0, self._waiting_time)

  def chance_outcomes(self) -> List[Tuple[int, float]]:
    """Returns the initial probability distribution is returned.

    One chance outcome correspond to each possible OD pair with a departure
    time, the probability of each chance outcome is the proportion of vehicle in
    each OD pair with a departure time.
    Returns:
      list_tuple_outcome_probabilities: chance outcomes and their probability.
    """
    game = self.get_game()
    if game.perform_sanity_checks:
      assert self._player_id == pyspiel.PlayerId.CHANCE
      assert self._is_chance_init
    return game.chance_outcomes

  def _legal_actions(self, player: pyspiel.PlayerId) -> List[int]:
    """Return the legal actions of the vehicle.

    Legal actions are the succesor road section of the vehicle current road
    section.
    Args:
      player: the vehicle id.

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
      assert player == pyspiel.PlayerId.DEFAULT_PLAYER_ID, str(player)
    if self._vehicle_without_legal_action:
      # If the vehicle is at destination it cannot do anything.
      return [dynamic_routing_utils.NO_POSSIBLE_ACTION]
    if self._waiting_time > 0:
      return [dynamic_routing_utils.NO_POSSIBLE_ACTION]
    _, end_section_node = dynamic_routing_utils._road_section_to_nodes(  # pylint:disable=protected-access
        self._vehicle_location)
    successors = self.get_game().network.get_successors(end_section_node)
    if self.get_game().perform_sanity_checks:
      if not successors:
        raise ValueError(("If a vehicle is not without legal action, it"
                          " should have an action."))
      assert isinstance(successors, Iterable)
    actions = [
        self.get_game().network.get_action_id_from_movement(
            end_section_node, d) for d in successors
    ]
    map(self.get_game().network.assert_valid_action, actions)
    return sorted(actions)

  def _apply_action(self, action: int):
    """Apply the action to the state.

    This function can be either called on a chance node or on a decision
    node. If called on the initial chance node, the action gives in which OD
    demand the representative vehicle belongs too (it put the vehicle at
    this location and define its destination).
    If called on decision node, the action defines on which link the vehicle
    will move (if it is not stuck in traffic) and assign a waiting time to the
    vehicle.
    Args:
      action: the action to apply.
    """
    if self._player_id == pyspiel.PlayerId.CHANCE:
      self._player_id = pyspiel.PlayerId.DEFAULT_PLAYER_ID
      assert self._is_chance_init
      # Apply action is called on initial chance node to initialized
      # the vehicle position based on the initial location
      # distribution.
      od_demand = self.get_game().od_demand
      self._vehicle_destination = od_demand[action].destination
      self._vehicle_location = od_demand[action].origin
      self._waiting_time = int(od_demand[action].departure_time /
                               self._time_step_length)
      self._is_chance_init = False
      self._normed_density_on_vehicle_link = 0
    elif self._player_id == pyspiel.PlayerId.DEFAULT_PLAYER_ID:
      self._player_id = pyspiel.PlayerId.MEAN_FIELD
      # Apply action is called on a descision node. If the vehicle can
      # move, then it will move to the next road section.
      # Has the vehicle already reached a sink node?
      if not self._vehicle_without_legal_action:
        # If the vehicle is stuck in traffic it cannot move.
        if self._waiting_time > 0:
          self._waiting_time -= 1
        else:
          if self.get_game().perform_sanity_checks:
            self.get_game().network.assert_valid_action(action,
                                                        self._vehicle_location)
          self._vehicle_location = (
              self.get_game().network.get_road_section_from_action_id(action))
          # Has the vehicle just reached its destination?
          if self._vehicle_location == self._vehicle_destination:
            self._vehicle_final_travel_time = self._current_time_step
            self._vehicle_at_destination = True
            self._vehicle_without_legal_action = True
          # Will the vehicle have a legal action for next time step?
          elif self.get_game().network.is_location_at_sink_node(
              self._vehicle_location):
            self._vehicle_without_legal_action = True
            self._vehicle_final_travel_time = -self.get_game().min_utility()
          else:
            self._waiting_time = WAITING_TIME_NOT_ASSIGNED
      self._current_time_step += 1
    elif self.get_game().perform_sanity_checks:
      if self._is_terminal:
        raise ValueError(
            "_apply_action should not be called at a end of the game.")
      if self._player_id == pyspiel.PlayerId.MEAN_FIELD:
        raise ValueError(
            "_apply_action should not be called at a MEAN_FIELD state.")
    # Is the game finished?
    if self._current_time_step >= self.get_game().max_game_length():
      self._is_terminal = True
      if not self._vehicle_at_destination:
        self._vehicle_final_travel_time = -self.get_game().min_utility()

  def _action_to_string(self, player, action) -> str:
    """Action -> string."""
    if player == pyspiel.PlayerId.CHANCE:
      if self._is_chance_init:
        return f"Vehicle is assigned to population {action}."
      return f"Change node; the vehicle movement is {bool(action)}."
    if self.get_game().perform_sanity_checks:
      assert player == pyspiel.PlayerId.DEFAULT_PLAYER_ID
    if action == dynamic_routing_utils.NO_POSSIBLE_ACTION:
      return f"Vehicle {player} reach a sink node or its destination."
    if self.get_game().perform_sanity_checks:
      self.get_game().network.assert_valid_action(action)
    return (f"Vehicle {player} would like to move to " + str(
        self.get_game().network.get_road_section_from_action_id(action)) + ".")

  def is_terminal(self) -> bool:
    """Returns True if the game is over."""
    return self._is_terminal

  def is_waiting(self) -> bool:
    """Returns True if the wait time is non-zero."""
    return self._waiting_time > 0

  def returns(self) -> List[float]:
    """Total reward for each player over the course of the game so far."""
    if not self._is_terminal:
      return [0]
    return [-self._vehicle_final_travel_time * self._time_step_length]

  def get_location_as_int(self) -> int:
    """Returns the vehicle location.

    This will be 1-based action index of the location, or 0 when the location is
    None before the initial chance node.
    """
    return self.get_game().get_road_section_as_int(self._vehicle_location)

  def get_destination_as_int(self) -> int:
    """Returns the vehicle destination.


    This will be 1-based action index of the destination, or 0 when the
    destination is None before the initial chance node.
    """
    return self.get_game().get_road_section_as_int(self._vehicle_destination)

  def __str__(self) -> str:
    """String for debug purposes. No particular semantics are required."""
    if self._vehicle_location is not None:
      return self.state_to_str(
          self._vehicle_location,
          self._current_time_step,
          player_id=self._player_id,
          waiting_time=self._waiting_time)
    assert self._current_time_step == 0
    return "Before initial chance node"


class NetworkObserver:
  """Network observer used by the learning algorithm.

  The state string is the state history string. The state tensor is an array
  of size number of locations * 2 + maximum number of time steps + 2, which is
  the concatenation of one-hot encodings of the location, destination (1-based;
  if location or destination is None, then the 0th element will be set to 1) and
  the current time (0-based). The last element of the array will be set to 1 if
  waiting time is positive, or 0 otherwise.

  Attributes:
    dict: Dictionary of tensors for the components of the observation
      corresponding to the location, destination and time.
    tensor: The concatenated form of the observation.
  """

  def __init__(self, num_locations: int, max_num_time_step: int):
    """Initializes an empty observation tensor."""
    self.tensor = np.zeros(num_locations * 2 + max_num_time_step + 1 + 1,
                           np.float32)
    self.dict = {
        "location": self.tensor[:num_locations],
        "destination": self.tensor[num_locations:num_locations * 2],
        "time": self.tensor[num_locations * 2:-1],
        "waiting": self.tensor[-1:]
    }

  def set_from(self, state, player):
    """Sets the state tensor based on the specified state.

    Note that the function may be called with arbitrary states of the game, e.g.
    from different runs, and therefore the tensor should be cleared and updated
    instead of preserving any earlier values.

    Args:
      state: state of the game.
      player: player id that should play.
    """
    assert player == pyspiel.PlayerId.DEFAULT_PLAYER_ID
    self.tensor.fill(0)
    self.dict["location"][state.get_location_as_int()] = 1
    self.dict["destination"][state.get_destination_as_int()] = 1
    self.dict["time"][state.current_time_step] = 1
    self.dict["waiting"][0] = state.is_waiting()

  def string_from(self, state, player):
    """Return the state history string."""
    assert player == pyspiel.PlayerId.DEFAULT_PLAYER_ID
    return str(state)


# Register the game with the OpenSpiel library
pyspiel.register_game(_GAME_TYPE, MeanFieldRoutingGame)
