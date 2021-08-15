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
"""Implementation of a mean field routing game.

The mean field routing game is a variant of the ones described by:
- A mean field route choice game by R. Salhab, J. Le Ny and R. P. Malhamé, 2018
  IEEE CDC.
- Existence and uniqueness result for mean field games with congestion effect on
  graphs, O. Gueant, Applied Mathematics & Optimization, 2015
- Dynamic driving and routing games for autonomous vehicles on networks: A mean
  field game approach, K. Huang, X. Chen, X. Di and Q. Du, TRB part C, 2021
It is the extension of the dynamic routing game python_dynamic_routing_game.
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
"""

from typing import Any, Iterable, List, Mapping, Tuple

import numpy as np

from open_spiel.python.games.dynamic_routing import dynamic_routing_game_utils
import pyspiel


_GAME_TYPE = pyspiel.GameType(
  short_name="python_mean_field_routing_game",
  long_name="Python Mean Field Routing Game",
  dynamics=pyspiel.GameType.Dynamics.MEAN_FIELD,
  chance_mode=pyspiel.GameType.ChanceMode.EXPLICIT_STOCHASTIC,
  information=pyspiel.GameType.Information.PERFECT_INFORMATION,
  utility=pyspiel.GameType.Utility.GENERAL_SUM,
  reward_model=pyspiel.GameType.RewardModel.TERMINAL,
  max_num_players=1,
  min_num_players=1,
  provides_information_state_string=True,
  provides_information_state_tensor=True,
  provides_observation_string=True,
  provides_observation_tensor=True,
  default_loadable=True,
  provides_factored_observation_string=True,
  parameter_specification={"players": -1})


_DEFAULT_NETWORK = dynamic_routing_game_utils.Network(
  {"bef_O": "O", "O": ["A"], "A": ["D"], "D": ["aft_D"], "aft_D": []})
_DEFAULT_DEMAND = [dynamic_routing_game_utils.OriginDestinationDemand(
  "bef_O->O", "D->aft_D", 0, 100)]


class MeanFieldRoutingGame(pyspiel.Game):
  """Implementation of dynamic routing game.

  At each time, the representative vehicle/player chooses on which successor
  link they would like to go. At each chance node, the vehicle is assigned a
  probability to exit its current road section based on the current volume on
  its road section (given by the distribution of players) and the exit
  function of the road section (variant of the volume delay function). The
  vehicle travel time is equal to the time step when they first reach their
  destination. Therefore the game is mean field, explicitly stochastic, is a
  general sum game with a terminal reward model. See file docstring for more
  information.

  Attributes inherited from GameInfo:
    max_chance_outcome: maximum number of chance possibilities. This is
      equal to max(2, len(self._od_demand)) as the initial chance node
      assigns the representative vehicle to be in one of the OD demand
      population (len(self._od_demand) outcomes), and regular chance nodes
      decide if the vehicle can move or if it is stuck in traffic (2
      outcomes).
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
    _od_demand: a list of the vehicle. Their origin and their destination
      should be road sections of the game.
    perform_sanity_checks: if true, sanity checks are done during the game,
      should be set to false to speed up the game.
  """
  network: dynamic_routing_game_utils.Network
  _od_demand: List[dynamic_routing_game_utils.OriginDestinationDemand]
  perform_sanity_checks: bool

  def __init__(
    self, params: Mapping[str, Any] = None,
    network: dynamic_routing_game_utils.Network = None,
    od_demand: List[dynamic_routing_game_utils.OriginDestinationDemand] = None,
    max_num_time_step: int = 2,
    perform_sanity_checks: bool = True
  ):
    """Initiliaze the game.

    Args:
      params: game parameters.
      network: the network of the game.
      od_demand: a list of the vehicle. Their origin and their
        destination should be road sections of the game.
      max_num_time_step: set the max_game_length attribute.
      perform_sanity_checks: if true, sanity checks are done during the
        game, should be set to false to faster the game.
    """
    self.network = network if network else _DEFAULT_NETWORK
    self._od_demand = od_demand if od_demand else _DEFAULT_DEMAND
    self.network.check_list_of_od_demand_is_correct(self._od_demand)
    self.perform_sanity_checks = perform_sanity_checks
    game_info = pyspiel.GameInfo(
      num_distinct_actions=self.network.num_links() + dynamic_routing_game_utils.INDEX_FIRST_ACTION,
      max_chance_outcomes=max(2, len(self._od_demand)),
      num_players=1,
      min_utility=-max_num_time_step-1,
      max_utility=0,
      max_game_length=max_num_time_step)
    super().__init__(_GAME_TYPE, game_info, params if params else {})

  def new_initial_state(self) -> "MeanFieldRoutingGameState":
    """Returns the state corresponding to the start of a game."""
    return MeanFieldRoutingGameState(self, self._od_demand)

  def make_py_observer(self, iig_obs_type=None, params=None):
    """Returns a NetworkObserver object used for observing game state."""
    return NetworkObserver(self.max_game_length())

  def max_chance_nodes_in_history(self):
    """Maximun chance nodes in game history."""
    return self.max_game_length() + 1


class MeanFieldRoutingGameState(pyspiel.State):
  """State of the DynamicRoutingGame.

  One player is equal to one vehicle.
  See docstring of the game class and of the file for more information.
  Attributes:
    _can_vehicle_move: encodes if the vehicle is moving to the next road section
      (True) either it is stuck in traffic on its current road section (False).
    _current_time_step: current time step of the game.
    _init_distribution: probability at time 0 for the representative player
      to have the origin, the destination and the departure time given by
      _od_demand.
    _is_chance_init: boolean that encodes weither the current node is the
      initial chance node.
    _is_terminal: boolean that encodes weither the game is over.
    _normed_density_on_vehicle_link: density of vehicle on the link that is
      used by the representative vehicle. This is given by the mean field
      distribution.
    _total_num_vehicle: total number of vehicles as the sum of the _od_demand.
    _vehicle_at_destination: boolean that encodes if the representative vehicle
      has reached its destination.
    _vehicle_destination: the destination of the representative vehicle
      corresponding to this state (once the state is no longer in chance_init mode).
    _vehicle_final_travel_time: the travel time of the representative vehicle,
      the travel is either 0 if the vehicle is still in the network or its
      travel time if the vehicle has reached its destination.
    _vehicle_location: current location of the vehicle as a network
      road section.
    _vehicle_without_legal_action: boolean that encodes if the representative
      vehicle has reach a sink node, meaning that it will not be able to move
      anymore.
  """
  _can_vehicle_move: bool
  _current_time_step: int
  _init_distribution: List[float]
  _is_chance_init: bool
  _is_terminal: bool
  _normed_density_on_vehicle_link: float
  _od_demand: List[dynamic_routing_game_utils.OriginDestinationDemand]
  _total_num_vehicle: float
  _vehicle_at_destination: bool
  _vehicle_destination: str
  _vehicle_final_travel_time: float
  _vehicle_location: str
  _vehicle_without_legal_action: bool

  def __init__(
    self, game: MeanFieldRoutingGame,
    od_demand: List[dynamic_routing_game_utils.OriginDestinationDemand]
  ):
    """Constructor; should only be called by Game.new_initial_state."""
    super().__init__(game)
    self._current_time_step = 0
    self._is_chance_init = True  # is true for the first state of the game.
    self._is_terminal = False
    if self.get_game().perform_sanity_checks:
      assert game.num_players() == 1, (
        'This mean field routing game should have a unique player.')
    self._player_id = pyspiel.PlayerId.CHANCE
    self._vehicle_at_destination = False
    self._vehicle_final_travel_time = 0.0
    self._vehicle_without_legal_action = False
    # create distribution and total weight in the network.
    self._od_demand = od_demand
    self._total_num_vehicle = sum(
      [od_demand_item.counts for od_demand_item in od_demand])
    self._init_distribution = [
        od_demand_item.counts/self._total_num_vehicle
        for od_demand_item in od_demand
      ]
    self._vehicle_location = None

  def current_time_step(self) -> int:
    """Return current time step."""
    return self._current_time_step

  def current_player(self) -> pyspiel.PlayerId:
    """Returns the current player.

    If the game is over, TERMINAL is returned. If the game is at a chance
    node then CHANCE is returned. Otherwise SIMULTANEOUS is returned."""
    if self._is_terminal:
      return pyspiel.PlayerId.TERMINAL
    return self._player_id

  def state_to_str(
      self, location, time_step, player_id=pyspiel.PlayerId.DEFAULT_PLAYER_ID,
      vehicle_movement = True
  ):
    """TODO: update docstring.

    A string that uniquely identify a triplet x, t, player_id.
    Args:
      TODO:
    Returns:
      TODO:
    """
    # TODO: return other state str if before departure time.
    if self._is_chance_init:
      return "initial chance node"
    if player_id == pyspiel.PlayerId.DEFAULT_PLAYER_ID:
      time = str(time_step)
    elif player_id == pyspiel.PlayerId.MEAN_FIELD:
      time = f"{time_step}_mean_field"
    elif player_id == pyspiel.PlayerId.CHANCE:
      time = f"{time_step}_chance"
    else:
      raise ValueError(
        'Player id should be DEFAULT_PLAYER_ID, MEAN_FIELD or CHANCE')
    if self._vehicle_final_travel_time:
      return (f"Arrived at {location}, with travel time "
        f"{self._vehicle_final_travel_time}, t={time}")
    return (f"Location={location}, movement={vehicle_movement},"
            f" t={time}, destination='{self.destination}'")

  def distribution_support(self) -> List[str]:
    """Returns the state that should be used for update_distribution.

    The distribution of the vehicle is used to determined the number of
    cars on the same link of the representative vehicle is order to define
    the probability for the representative vehicle to exit the current link.
    Therefore, only the states corresponding to be on the link of the
    representative vehicle at this current time are useful.
    Returns:
        list of the two state: being on the link of the representative
            vehicle at the current time and being stuck in traffic or
            not.
    """
    return  [
      self.state_to_str(
        self._vehicle_location, self._current_time_step,
        player_id=pyspiel.PlayerId.MEAN_FIELD,
        vehicle_movement=vehicle_movement)
      for vehicle_movement in [True, False]
    ]

  def update_distribution(self, distribution: List[float]):
    """Get the number of cars on the same link as the representative player.

    _normed_density_on_vehicle_link stores the number of cars on the link
    where the representative player is.
    Args:
      distribution: the probability for a vehicle to be in the states in
        distribution_support. The distribution is a list of probabilities.
    """
    if self.get_game().perform_sanity_checks:
      if self._player_id != pyspiel.PlayerId.MEAN_FIELD:
        raise ValueError(("update_distribution should only be called at"
                          " a MEAN_FIELD state."))
    self._normed_density_on_vehicle_link = sum(distribution)
    if self.get_game().perform_sanity_checks:
      assert 0 <= self._normed_density_on_vehicle_link <= 1 + 1e-4, (
        f"{self._normed_density_on_vehicle_link} is not in [0, 1].")
    self._player_id = pyspiel.PlayerId.CHANCE


  def chance_outcomes(self) -> List[Tuple[int, float]]:
    """Returns chance outcomes and their probability.

    Returns:
      list_tuple_outcome_probabilities: if initial chance node, the
        initial distribution is returned. One chance outcome correspond
        to each possible OD pair with a departure time, the probability
        of each chance outcome is the proportion of vehicle in each
        OD pair with a departure time. If not initial chance node,
        the chance outcome for the representative vehicle is either to
        move (1) of to be stuck in traffic (0). The probability of each
        chance outcome is given by the volume of cars on the link of the
        representative vehicle and the exit probability function of the
        link.
    """
    if self.get_game().perform_sanity_checks:
      assert self._player_id == pyspiel.PlayerId.CHANCE
    if self._is_chance_init:
      return list(enumerate(self._init_distribution))
    if self._vehicle_without_legal_action:
      return [(0, 1)]
    volume = self._total_num_vehicle * self._normed_density_on_vehicle_link
    probability_to_move = self.get_game().network.get_probability_to_exit(
        self._vehicle_location, volume)
    return [(1, probability_to_move),
            (0, 1-probability_to_move)]

  def _legal_actions(self, player: pyspiel.PlayerId) -> List[int]:
    """Return the legal actions of the vehicle.

    Legal actions are the succesor road section of the vehicle current
    road section.
    Args:
      vehicle: the vehicle id.
    Returns:
      list_legal_actions: a list of legal actions. If the game is
        finished then the list is empty. If the vehicle is at its
        destination or on a node without successors then an empty list
        is returned. Otherwise the list of successors nodes of the
        current vehicle location is returned.
    """
    if self._is_terminal:
      return []
    if self.get_game().perform_sanity_checks:
      if player == pyspiel.PlayerId.MEAN_FIELD:
        raise ValueError(
          "_legal_actions should not be called at a MEAN_FIELD state."
          )
      assert player == pyspiel.PlayerId.DEFAULT_PLAYER_ID
    if self._vehicle_without_legal_action:
      # If the vehicle is at destination it cannot do anything.
      return [dynamic_routing_game_utils.NO_POSSIBLE_ACTION]
    if not self._can_vehicle_move:
      return [dynamic_routing_game_utils.NO_POSSIBLE_ACTION]
    _, end_section_node = dynamic_routing_game_utils._road_section_to_nodes(
      self._vehicle_location)
    successors = self.get_game().network.get_successors(end_section_node)
    if self.get_game().perform_sanity_checks:
      if not successors:
        raise ValueError(('If a vehicle is not without legal action, it'
                          ' should have an action.'))
      assert isinstance(successors, Iterable)
    actions = [self.get_game().network.get_action_id_from_movement(
      end_section_node, d) for d in successors]
    map(self.get_game().network.assert_valid_action, actions)
    return sorted(actions)

  def _apply_action(self, action: int):
    """Apply the action to the state.

    This function can be either called on a chance node or on a decision
    node. If called on the initial chance node, the action gives in which OD
    demand the representative vehicle belongs too (it put the vehicle at
    this location and define its destination). If called on regular chance
    node, the action defines if the vehicle can move or not.
    If called on decision node, the action defines on which link the vehicle
    will move (if it is not stuck in traffic).
    """
    if self.get_game().perform_sanity_checks:
      if self._is_terminal:
        raise ValueError(
          "_apply_action should not be called at a end of the game.")
      if self._player_id == pyspiel.PlayerId.MEAN_FIELD:
        raise ValueError(
          "_apply_action should not be called at a MEAN_FIELD state.")
    if self._player_id == pyspiel.PlayerId.CHANCE:
      self._player_id = pyspiel.PlayerId.DEFAULT_PLAYER_ID
      if self._is_chance_init:
        # Apply action is called on initial chance node to initialized
        # the vehicle position based on the initial location
        # distribution.
        self._vehicle_destination = self._od_demand[action].destination
        self._vehicle_location = self._od_demand[action].origin
        # TODO: enable movement based on departure time.
        self._can_vehicle_move = True
        self._is_chance_init = False
      else:
        # Apply action is called on chance node to enable vehicle
        # movement based on current traffic.
        if self.get_game().perform_sanity_checks:
          assert action in [0, 1]
        self._can_vehicle_move = bool(action)
    elif self._player_id == pyspiel.PlayerId.DEFAULT_PLAYER_ID:
      self._player_id = pyspiel.PlayerId.MEAN_FIELD
      # Apply action is called on a descision node. If the vehicle can
      # move, then it will move to the next road section.
      # Has the vehicle already reached a sink node?
      if not self._vehicle_without_legal_action:
        # If the vehicle is stuck in traffic it cannot move.
        if self._can_vehicle_move:
          if self.get_game().perform_sanity_checks:
            self.get_game().network.assert_valid_action(
              action, self._vehicle_location)
          self._vehicle_location = (
            self.get_game().network.get_road_section_from_action_id(
              action))
          # Has the vehicle just reached its destination?
          if (self._vehicle_location ==
                self._vehicle_destination):
            self._vehicle_final_travel_time = self._current_time_step
            self._vehicle_at_destination = True
            self._vehicle_without_legal_action = True
            self._is_terminal = True
          # Will the vehicle have a legal action for next time step?
          elif self.get_game().network.is_location_at_sink_node(
              self._vehicle_location):
            self._vehicle_without_legal_action = True
            self._is_terminal = True
            self._vehicle_final_travel_time = - self.get_game().min_utility()
      self._current_time_step += 1
    # Is the game finished?
    if self._current_time_step >= self.get_game().max_game_length():
      self._is_terminal = True
      if not self._vehicle_at_destination:
        self._vehicle_final_travel_time = - self.get_game().min_utility()

  def _action_to_string(self, player, action) -> str:
    """Action -> string."""
    if player == pyspiel.PlayerId.CHANCE:
      if self._is_chance_init:
        return f"Vehicle is assigned to population {action}."
      return f"Change node; the vehicle movement is {bool(action)}."
    if self.get_game().perform_sanity_checks:
      assert player == pyspiel.PlayerId.DEFAULT_PLAYER_ID
    if action == dynamic_routing_game_utils.NO_POSSIBLE_ACTION:
      return f"Vehicle {player} reach a sink node or its destination."
    if self.get_game().perform_sanity_checks:
      self.get_game().network.assert_valid_action(action)
    return (f"Vehicle {player} would like to move to " +
            str(self.get_game().network.get_road_section_from_action_id(
              action)) + ".")

  def is_terminal(self) -> bool:
    """Returns True if the game is over."""
    return self._is_terminal

  def returns(self) -> List[float]:
    """Total reward for each player over the course of the game so far."""
    if not self._is_terminal:
      return [0]
    return [- self._vehicle_final_travel_time]

  def get_location_as_int(self) -> int:
    """Get the vehicle location."""
    if self._vehicle_location is None:
      return -1
    start_node, end_node = dynamic_routing_game_utils._road_section_to_nodes(self._vehicle_location)
    return self.get_game().network.get_action_id_from_movement(
      start_node, end_node)

  def __str__(self) -> str:
    """String for debug purposes. No particular semantics are required."""
    if hasattr(self, '_vehicle_location'):
      return self.state_to_str(
        self._vehicle_location, self._current_time_step,
        player_id=self._player_id,
        vehicle_movement=self._can_vehicle_move)
    assert self._current_time_step == 0
    return "Before initial chance node"


class NetworkObserver:
  """Network observer used by the learning algorithm.

  The state string is the state history string. The state tensor is an array
  of size max_game_length, num_players where each element is the location of
  the vehicle at this time.
  Attributes:
    dict: dictionary {"observation": tensor}.
    tensor: TODO.
  """

  def __init__(self, num_time: int):
    """Initializes an empty observation tensor."""
    self.tensor = np.zeros(num_time+1, np.float32)
    self.dict = {"observation": self.tensor}

  def set_from(self, state, player):
    """Update the state tensor.

    Put the locations of each players in the tensor row corresponding to
    the current time step. Insert the current player location at the
    beginning of the row."""
    assert player == pyspiel.PlayerId.DEFAULT_PLAYER_ID
    self.dict["observation"][state.current_time_step()] = state.get_location_as_int()

  def string_from(self, state, player):
    """Return the state history string."""
    assert player == pyspiel.PlayerId.DEFAULT_PLAYER_ID
    return str(state)


# Register the game with the OpenSpiel library
pyspiel.register_game(_GAME_TYPE, MeanFieldRoutingGame)
