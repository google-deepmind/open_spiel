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
"""Simple implementation of dynamic routing game to test API.

The dynamic routing game is based on the paper: A mean field route choice game
by R. Salhab, J. Le Ny and R. P. Malhamé, 2018 IEEE CDC.
We consider:
- the following network:
    O --> A --> D
- two vehicles trying to reach D from O.
The two vehicles leave the origin O at the beginning of the game.
At O, they can only go at A. Then at A they can only go at D. At D, both
vehicles cannot do any actions and the game is finished.

Later, to add some complexity, we will add a third vehicle who leaves A to
reach D at the beginning of the game. In this case, at time O, vehicles 0 and 1
go from O to A and vehicle 2 goes from A to D. Then at time 1, vehicles 0 and 1
go from A to D, vehicle 2 cannot do anything and the game is finished.

Later a more complex network can considered, with routing choice at
intersections and probability to stay on a link as a function of the number of
cars on the link. Currently this is not done to test the API.
"""

import numpy as np
from typing import Dict, Iterable, List, NewType, Set, Tuple

import pyspiel

RoadSection = NewType("RoadSection", str)

_GAME_TYPE = pyspiel.GameType(
    short_name="python_dynamic_routing_game",
    long_name="Python Dynamic Routing Game",
    dynamics=pyspiel.GameType.Dynamics.SIMULTANEOUS,
    chance_mode=pyspiel.GameType.ChanceMode.DETERMINISTIC,
    information=pyspiel.GameType.Information.PERFECT_INFORMATION,
    utility=pyspiel.GameType.Utility.GENERAL_SUM,
    reward_model=pyspiel.GameType.RewardModel.TERMINAL,
    max_num_players=3,
    min_num_players=0,
    provides_information_state_string=True,
    provides_information_state_tensor=True,
    provides_observation_string=True,
    provides_observation_tensor=True,
    parameter_specification={})
_NETWORK_ADJACENCY_LIST = {"O": ["A"], "A": ["D"], "D": []}


# movement to actions and action to road section can be created from
# a function.
def _movement_to_string(origin: RoadSection, destination: RoadSection) -> str:
    return f"{origin}->{destination}"


def _create_movement_to_action_and_action_to_road_section(
    network_adjacency_list: Dict[RoadSection, Iterable[RoadSection]]
) -> Tuple[Dict[str, int], Dict[int, RoadSection]]:
    """Create dictionary that maps movement to action.

    The dictionary that maps movement to action is used to define the action
    from a movement that a vehicle would like to do. The dictionary that maps
    an action to the destintion of the movement is used to move a vehicle that
    does an action to the destination of its movement.
    Args:
        network_adjacency_list: adjacency list of the network.
    Returns:
        movement_to_action: dictionary with key begin a movement for example
            "O->A" and value the action numbers. Action numbers are succesive
            integers indexed from 0.
        action_to_road_section: map an action number to the end node of the
            movement. if movement_to_action["O->A"] = 0 then,
            action_to_road_section[0] = "A"
    """
    movement_to_action = {}
    action_to_road_section = {}
    action_number = 0
    for origin, successors in network_adjacency_list.items():
        for destination in successors:
            movement_to_action[_movement_to_string(origin, destination)] =\
                action_number
            action_to_road_section[action_number] = destination
            action_number += 1
    return movement_to_action, action_to_road_section


_MOVEMENT_TO_ACTION, _ACTION_TO_ROAD_SECTION =\
    _create_movement_to_action_and_action_to_road_section(
        _NETWORK_ADJACENCY_LIST)


class DynamicRoutingGame(pyspiel.Game):
    """Simple implementation of dynamic routing game to test API.

    At each time, each vehicle/player choose on which successor link they would
    like to go. Their travel time is equald to the time step when they first
    reach their destination. Therefore the game is simultaneous, deterministic,
    is a general sum game with a terminal reward model.
    See file docstring for more information.
    """
    def __init__(self, params=None, num_players: int = 2):
        max_number_time_step = 2
        game_info = pyspiel.GameInfo(
            num_distinct_actions=len(_ACTION_TO_ROAD_SECTION),
            max_chance_outcomes=0,
            num_players=num_players,
            min_utility=-max_number_time_step-1,
            max_utility=0,
            max_game_length=max_number_time_step)
        super().__init__(_GAME_TYPE, game_info, dict())

    def new_initial_state(self) -> "DynamicRoutingGameState":
        """Returns a state corresponding to the start of a game."""
        return DynamicRoutingGameState(self)

    def make_py_observer(self, iig_obs_type=None, params=None):
        """Returns an object used for observing game state."""
        return NetworkObserver()


class DynamicRoutingGameState(pyspiel.State):
    """State of the DynamicRoutingGame.

    See docstring of the game class and of the file for more information.

    Attributes:
      _current_time_step: current time step of the game.
      _is_terminal: bool that encodes is the game is over.
      _max_time_step: maximum number of time step played.
      _max_travel_time: travel time given to vehicles that have not reached
        their destination when the game is over. Current set to min_utility.
      _num_distinct_actions: from Game.
      _num_players: from Game.
      _vehicle_at_destination: Set of vehicles that have reached their
        destinations. When a vehicle has reached its destination but the game
        is not finished, it cannot do anything.
      _vehicle_destinations: a list of the destination of each vehicle/player.
      _vehicle_final_travel_times: the travel times of each vehicle, the travel
        is either 0 if the vehicle is still in the network or its travel time
        if the vehicle has reached its destination.
      _vehicle_locations: current location of the vehicles.
    """
    _current_time_step: int
    _is_terminal: bool
    _max_time_step: int
    _max_travel_time: float
    _num_distinct_actions: int
    _num_players: int
    _vehicle_at_destination: Set[int]
    _vehicle_destinations: List[RoadSection]
    _vehicle_final_travel_times: List[float]
    _vehicle_locations: List[RoadSection]

    def __init__(self, game: DynamicRoutingGame):
        """Constructor; should only be called by Game.new_initial_state."""
        super().__init__(game)
        self._current_time_step = 0
        self._is_terminal = False
        self._max_time_step = game.max_game_length()
        self._max_travel_time = - game.min_utility()
        self._num_distinct_actions = game.num_distinct_actions()
        self._num_players = game.num_players()
        self._vehicle_at_destination = set()
        self._vehicle_destinations = ["D" for _ in range(self._num_players)]
        self._vehicle_final_travel_times = [0.0 for _ in range(
            self._num_players)]
        if self._num_players == 2:
            self._vehicle_locations = ["O", "O"]
        elif self._num_players == 3:
            self._vehicle_locations = ["O", "O", "A"]
        else:
            raise ValueError("The game should have 2 or 3 players.")

    def current_player(self) -> pyspiel.PlayerId:
        """Returns SIMULTANEOUS if the game is not over, otherwise TERMINAL."""
        return (pyspiel.PlayerId.TERMINAL if self._is_terminal
                else pyspiel.PlayerId.SIMULTANEOUS)

    def assert_valid_action(self, action: int, location: str = ""):
        """Assert that an action as a int is valid."""
        assert isinstance(action, int), f"{action} is not a int."
        assert action >= 0
        assert action < self._num_distinct_actions
        if location:
            successors = _NETWORK_ADJACENCY_LIST[location]
            assert _ACTION_TO_ROAD_SECTION[action] in successors, (
                  f"Invalid action {_ACTION_TO_ROAD_SECTION[action]}. It is "
                  f"not a successors of {location}: {successors}.")

    def assert_valid_player(self, player: int):
        """Assert that an player as a int is valid."""
        assert isinstance(player, int)
        assert player >= 0
        assert player < self._num_players

    def legal_actions(self, player: int = None) -> List[int]:
        """Return the legal actions of the player.

        Args:
            player: the player id.
        Returns:
            list_legal_actions: a list of legal actions. If the game is
                finished then the list is empty. If the player is at its
                destination or on a node without successors then INVALID_ACTION
                is returned. Otherwise the list of successors nodes of the
                current player location is returned.
        """
        if self._is_terminal:
            return []
        self.assert_valid_player(player)
        if player in self._vehicle_at_destination:
            # If the vehicle is at destination it cannot do anything.
            return [pyspiel.INVALID_ACTION]
        location = self._vehicle_locations[player]
        successors = _NETWORK_ADJACENCY_LIST[location]
        if successors:
            assert isinstance(successors, Iterable)
            actions = [_MOVEMENT_TO_ACTION[_movement_to_string(location,
                                                               successor)]
                       for successor in successors]
            map(self.assert_valid_action, actions)
            return sorted(actions)
        return [pyspiel.INVALID_ACTION]

    def _apply_actions(self, actions: List[int]):
        """Applies the specified action to the state.

        For each player's action, if the player is not at destination or has
        not an invalid action and if the action is valid, then the player will
        move to the successor link corresponding to its action.
        This function also detectors if a player has reached its destination at
        the previous time step. The function evolves the time and check if the
        game is finished.
        Args:
            actions: the action choosen by each player.
        """
        assert len(actions) == self._num_players, (
          f"Each player does not have an actions. Actions has {len(actions)} "
          f"elements, it should have {self._num_players}.")
        for vehicle_id, action in enumerate(actions):
            # Has the vehicle already reached its destination?
            if vehicle_id in self._vehicle_at_destination:
                continue
            location = self._vehicle_locations[vehicle_id]
            if action != pyspiel.INVALID_ACTION:
                self.assert_valid_action(action, location)
                self._vehicle_locations[vehicle_id] =\
                    _ACTION_TO_ROAD_SECTION[action]
                # TODO(Theo): Enable mix strategies.
            # If the action was invalid, has the vehicle just reached its
            # destination?
            elif (location == self._vehicle_destinations[vehicle_id]):
                self._vehicle_final_travel_times[vehicle_id] =\
                  self._current_time_step
                self._vehicle_at_destination.add(vehicle_id)
        self._current_time_step += 1
        # Is the game finished?
        if (self._current_time_step >= self._max_time_step or
                all(map(lambda a: a == pyspiel.INVALID_ACTION, actions))):
            self._is_terminal = True
            for vehicle_id in range(self._num_players):
                if vehicle_id not in self._vehicle_at_destination:
                    self._vehicle_final_travel_times[vehicle_id] =\
                      self._max_travel_time
        elif len(self._vehicle_at_destination) == self._num_players:
            self._is_terminal = True

    def _action_to_string(self, player, action) -> str:
        """Action -> string."""
        self.assert_valid_player(player)
        if action == pyspiel.INVALID_ACTION:
            return f"Vehicle {player} reach a sink node or its destination."
        self.assert_valid_action(action)
        return (f"Vehicle {player} would like to move to "
                f"{_ACTION_TO_ROAD_SECTION[action]}.")

    def is_terminal(self) -> bool:
        """Returns True if the game is over."""
        return self._is_terminal

    def returns(self) -> List[float]:
        """Total reward for each player over the course of the game so far."""
        return [- travel_time
                for travel_time in self._vehicle_final_travel_times]

    def __str__(self) -> str:
        """String for debug purposes. No particular semantics are required."""
        return (f"Vehicle locations: {self._vehicle_locations}, "
                f"time: {self._current_time_step}.")


class NetworkObserver:
    """Dummy observer for algorithm to work."""

    def __init__(self, shape: int = 1):
        """Dummy function for debugging."""
        self.tensor = np.zeros(np.prod(shape), np.float32)
        self.dict = {"observation": np.reshape(self.tensor, shape)}

    def set_from(self, state, player):
        """Dummy function for debugging."""
        obs = self.dict["observation"]
        obs.fill(0)

    def string_from(self, state, player):
        """Dummy function for debugging."""
        return ""


# Register the game with the OpenSpiel library
pyspiel.register_game(_GAME_TYPE, DynamicRoutingGame)
