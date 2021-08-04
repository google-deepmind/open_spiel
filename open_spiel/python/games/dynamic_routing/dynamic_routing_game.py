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
"""Implementation of dynamic routing game.

The dynamic routing game is very similar to the game described in:
- Multi-Agent Reinforcement Learning for Dynamic Routing Games: A Unified
  Paradigm by Z. Shoua, X. Di, 2020
We consider:
- a network given by the class Network.
- a list of vehicles given by the class Vehicle.
Each vehicle has an origin road section and would like to reach a destination
road section in the network. At every time step the vehicle might exit its
current road section, based on a probability given as a function of the volume
of vehicle on its road section. If the vehicle exit the road section then it can
choose on which successor road section it would like to go. When the vehicle
reaches its destination, it gets the current time as its cost. Therefore each
vehicle would like to minimize their travel time to reach their destination. If
the vehicle does not reached its destination by the end of the game, then its
cost is the number of time steps + 1.

The current game implementation is a N player game. However this game can also
be extended to a mean field game, implemented as python_mean_field_routing_game.
"""

from typing import Any, Iterable, List, Mapping, Set

import numpy as np

import pyspiel
from open_spiel.python.games.dynamic_routing.dynamic_routing_game_utils import (
    Network, Vehicle, INDEX_FIRST_ACTION, _road_section_to_nodes,
    NO_POSSIBLE_ACTION)


_GAME_TYPE = pyspiel.GameType(
    short_name="python_dynamic_routing_game",
    long_name="Python Dynamic Routing Game",
    dynamics=pyspiel.GameType.Dynamics.SIMULTANEOUS,
    chance_mode=pyspiel.GameType.ChanceMode.EXPLICIT_STOCHASTIC,
    information=pyspiel.GameType.Information.PERFECT_INFORMATION,
    utility=pyspiel.GameType.Utility.GENERAL_SUM,
    reward_model=pyspiel.GameType.RewardModel.TERMINAL,
    max_num_players=100,
    min_num_players=0,
    provides_information_state_string=True,
    provides_information_state_tensor=True,
    provides_observation_string=True,
    provides_observation_tensor=True,
    default_loadable=True,
    provides_factored_observation_string=True,
    parameter_specification={"players": -1})


_DEFAULT_NETWORK = Network({"bef_O": "O", "O": ["A"], "A": ["D"],
                            "D": ["aft_D"], "aft_D": []})
_DEFAULT_VEHICLES = [Vehicle("bef_O->O", "D->aft_D") for _ in range(2)]


class DynamicRoutingGame(pyspiel.Game):
    """Implementation of dynamic routing game.

    At each simultaneous-move time, each vehicle/player chooses on which
    successor link they would like to go. At each chance node, each vehicle is
    assigned to a probability to exit its current road section based on the
    current volume on its road section and the exit function of the road
    section (variant of the volume delay function).
    One vehicle travel time is equal to the time step when they first reach
    their destination. Therefore the game is simultaneous, explicitly
    stochastic, is a general sum game with a terminal reward model.
    See file docstring for more information.

    Attributes inherited from GameInfo:
        max_chance_outcome: maximum number of chance possibilities. This is
            equal to 2**num_player as each vehicle can either move or be stuck
            in traffic.
        max_game_length: maximum number of time step played. Chosed during by
            the constructor.
        max_utility: maximum utility is the opposite of the minimum travel
            time. Set to 0.
        min_utility: minimum utility is the opposite of the maximum travel
            time. Set to - max_game_length - 1.
        num_distinct_actions: maximum number of action possibles. This is
            equal to the number of links + 1 (corresponding to having no
            possible action _NO_POSSIBLE_ACTION).
        num_player: the number of vehicles. Choosen during by the constructor
            as the number of vehicles.
    Attributes:
        network: the network of the game.
        vehicles: a list of the vehicle. Their origin and their destination
            should be road sections of the game. The number of vehicles in the
            list set the num_player attribute.
        perform_sanity_checks: if true, sanity checks are done during the game,
            should be set to false to faster the game.
    """
    def __init__(
        self, params: Mapping[str, Any] = None,
        network: Network = None,
        vehicles: List[Vehicle] = None,
        max_num_time_step: int = 2,
        perform_sanity_checks: bool = True
    ):
        """Initiliaze the game.

        Args:
            params: game parameters.
            network: the network of the game.
            vehicles: a list of the vehicle. Their origin and their destination
                should be road sections of the game. The number of vehicles in
                the list set the num_player attribute.
            max_num_time_step: set the max_game_length attribute.
            perform_sanity_checks: if true, sanity checks are done during the
                game, should be set to false to faster the game.
        """
        self.network = network if network else _DEFAULT_NETWORK
        self._vehicles = vehicles if vehicles else _DEFAULT_VEHICLES
        self.network.check_list_of_vehicles_is_correct(self._vehicles)
        self.perform_sanity_checks = perform_sanity_checks
        game_info = pyspiel.GameInfo(
            num_distinct_actions=self.network.num_links() + INDEX_FIRST_ACTION,
            max_chance_outcomes=2**len(self._vehicles),
            num_players=len(self._vehicles),
            min_utility=-max_num_time_step-1,
            max_utility=0,
            max_game_length=max_num_time_step)
        super().__init__(_GAME_TYPE, game_info, params if params else {})

    def new_initial_state(self) -> "DynamicRoutingGameState":
        """Returns the state corresponding to the start of a game."""
        return DynamicRoutingGameState(self, self._vehicles)

    def make_py_observer(self, iig_obs_type=None, params=None):
        """Returns a NetworkObserver object used for observing game state."""
        return NetworkObserver(self.num_players(), self.max_game_length())


class DynamicRoutingGameState(pyspiel.State):
    """State of the DynamicRoutingGame.

    One player is equal to one vehicle.
    See docstring of the game class and of the file for more information.
    Attributes:
      _current_time_step: current time step of the game.
      _is_chance: boolean that encodes is the current node is a chance node.
      _is_terminal: boolean that encodes is the game is over.
      _vehicle_at_destination: set of vehicles that have reached their
        destinations. When a vehicle has reached its destination but the game
        is not finished, it cannot do anything.
      _vehicle_destinations: the destination of each vehicle.
      _vehicle_final_travel_times: the travel times of each vehicle, the travel
        is either 0 if the vehicle is still in the network or its travel time
        if the vehicle has reached its destination.
      _vehicle_locations: current location of the vehicles as a network
        road section.
      _vehicle_movements: movements of each vehicle as a list of boolean
        variables. Either a vehicle is moving to the next road section (True)
        either it is stuck in traffic on its current road section (False).
      _vehicle_without_legal_actions: list of vehicles without legal actions at
        next time step. This is required because if no vehicle has legal
        actions for a simultaneous node then an error if raised.
    """
    _current_time_step: int
    _is_chance: bool
    _is_terminal: bool
    _vehicle_at_destination: Set[int]
    _vehicle_destinations: List[str]
    _vehicle_final_travel_times: List[float]
    _vehicle_locations: List[str]
    _vehicle_movements: List[bool]
    _vehicle_without_legal_actions: Set[int]

    def __init__(self, game: DynamicRoutingGame,
                 vehicles: Iterable[Vehicle]):
        """Constructor; should only be called by Game.new_initial_state."""
        super().__init__(game)
        self._current_time_step = 0
        self._is_chance = False
        self._is_terminal = False
        self._vehicle_at_destination = set()
        self._vehicle_destinations = [vehicle.destination
                                      for vehicle in vehicles]
        self._vehicle_final_travel_times = [0.0 for _ in vehicles]
        self._vehicle_locations = [vehicle.origin for vehicle in vehicles]
        self._vehicle_movements = [True for _ in vehicles]
        self._vehicle_without_legal_actions = set()

    def current_time_step(self) -> int:
        """Return current time step."""
        return self._current_time_step

    def current_player(self) -> pyspiel.PlayerId:
        """Returns the current player.

        If the game is over, TERMINAL is returned. If the game is at a chance
        node then CHANCE is returned. Otherwise SIMULTANEOUS is returned."""
        if self._is_terminal:
            return pyspiel.PlayerId.TERMINAL
        if self._is_chance:
            return pyspiel.PlayerId.CHANCE
        return pyspiel.PlayerId.SIMULTANEOUS

    def assert_valid_action(self, action: int, road_section: str = None):
        """Assert that an action as a int is valid.

        The action should a a int between _INDEX_FIRST_ACTION and
        self._num_distinct_actions. In case road_section is not None then
        it is test if the action correspond to going on a road section which
        is a successor of road_section.
        """
        assert isinstance(action, int), f"{action} is not a int."
        assert action >= INDEX_FIRST_ACTION
        assert action < self.get_game().num_distinct_actions()
        if road_section is not None:
            new_road_section = (
                self.get_game().network.get_road_section_from_action_id(action))
            origin_new_section, end_new_section = _road_section_to_nodes(
                new_road_section)
            _, end_section_node = _road_section_to_nodes(road_section)
            assert end_section_node == origin_new_section, (
                f"The action is not legal, trying to go to {new_road_section} "
                f"from {road_section} without going through {end_section_node}"
                ".")
            successors = self.get_game().network.get_successors(
                origin_new_section)
            assert end_new_section in successors, (
                f"Invalid action {new_road_section}. It is not a successors of"
                f" {end_section_node}: {successors}.")

    def assert_valid_player(self, vehicle: int):
        """Assert that a vehicle as a int between 0 and num_players."""
        assert isinstance(vehicle, int), f"{vehicle} is not a int."
        assert vehicle >= 0, f"player: {vehicle}<0."
        assert vehicle < self.get_game().num_players(), (
            f"player: {vehicle} >= num_players: {self.get_game().num_players()}"
            )

    def chance_outcomes(self):
        """Returns the possible chance outcomes and their probabilities.

        Each chance outcome correspond to enabling each vehicle to move. The
        movement of each vehicle is encoded using bit. For example 1=b001 means
        that vehicle 0 will move but vehicle 1 and 2 are stuck in traffic. To
        determine if a vehicle is stuck in traffic, the probability exit
        functions of the network are used. For one vehicle its probability to
        move is the probability to exit its current road section given the
        current number of vehicles on the road section.
        """
        if self.get_game().perform_sanity_checks:
            assert self._is_chance
        volumes = {}
        for road_section in self._vehicle_locations:
            if road_section not in volumes:
                volumes[road_section] = 0
            # Each vehicle has a weight a one.
            volumes[road_section] += 1
        probabilities = {}
        for i in range(self.get_game().max_chance_outcomes()):
            prob = 1
            for vehicle in range(self.get_game().num_players()):
                # The vehicle movement is encoded on the vehicle bit of the
                # outcome.
                encode_vehicle_move = (i >> vehicle) % 2
                # Its probability to exit its road section is given by the
                # network and the current volume of vehicles on the road
                # section.
                p_movement_vehicle = (
                    self.get_game().network.get_probability_to_exit(
                        self._vehicle_locations[vehicle],
                        volumes[self._vehicle_locations[vehicle]]))
                if encode_vehicle_move:
                    prob = prob * p_movement_vehicle
                else:
                    prob = prob * (1 - p_movement_vehicle)
            probabilities[i] = prob
        return list(probabilities.items())

    def _legal_actions(self, vehicle: int) -> List[int]:
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
            self.assert_valid_player(vehicle)
        # TODO: enable movement based on departure time.
        if vehicle in self._vehicle_without_legal_actions:
            # If the vehicle is at destination it cannot do anything.
            return []
        _, end_section_node = _road_section_to_nodes(
            self._vehicle_locations[vehicle])
        successors = self.get_game().network.get_successors(end_section_node)
        if successors:
            assert isinstance(successors, Iterable)
            actions = [self.get_game().network.get_action_id_from_movement(
                end_section_node, d) for d in successors]
            if self.get_game().perform_sanity_checks:
                map(self.assert_valid_action, actions)
            return sorted(actions)
        return []

    def _apply_action(self, action: int):
        """Applies the specified chance action to the state.

        The action is a int that encodes the fact that the vehicle can move on
        its bit. For example 1=b001 means that vehicle 0 will move but vehicle
        1 and 2 are stuck in traffic. This function converts the action to
        movement for each vehicle and populates self._vehicle_movements
        accordingly.
        Args:
            action: int between 0 and max_chance_outcomes.
        """
        # This is not called at simultaneous-move states.
        if self.get_game().perform_sanity_checks:
            assert self._is_chance and not self._is_terminal
            assert (isinstance(action, int)
                    and 0 <= action <= self.get_game().max_chance_outcomes())
        self._is_chance = False
        self._vehicle_movements = [
            bool((action >> vehicle) % 2)
            for vehicle in range(self.get_game().num_players())]

    def _apply_actions(self, actions: List[int]):
        """Applies the specified action to the state.

        For each vehicle's action, if the vehicle is not at a sink node, if the
        action is valid and if the chance node has authorized the vehicle to
        move, then the vehicle will move to the successor link corresponding to
        its action.
        The function then detects if the vehicle has reached its destination or
        a sink node and updates _vehicle_at_destination,
        _vehicle_without_legal_actions and _vehicle_final_travel_times
        accordingly.
        The function evolves the time and check if the game is finished.
        Args:
            actions: the action choosen by each vehicle.
        """
        if self.get_game().perform_sanity_checks:
            assert not self._is_chance and not self._is_terminal
        self._is_chance = True
        if self.get_game().perform_sanity_checks:
            assert isinstance(actions, Iterable)
            assert len(actions) == self.get_game().num_players(), (
              f"Each player does not have an actions. Actions has {len(actions)} "
              f"elements, it should have {self.get_game().num_players()}.")
        self._current_time_step += 1
        for vehicle_id, action in enumerate(actions):
            # Has the vehicle already reached a sink node?
            if vehicle_id in self._vehicle_without_legal_actions:
                if self.get_game().perform_sanity_checks:
                    assert action == NO_POSSIBLE_ACTION, (
                        f"{action} should be 0.")
                continue
            # If the vehicle is stuck in traffic it cannot move.
            # TODO: Implement deterministic travel time option: when entering on
            # the link, the vehicle get assigned to a travel time. Currently, at
            # each time step the vehicle has a probability to exit, so the
            # travel time is stochastic, which makes the game stochastic.
            if not self._vehicle_movements[vehicle_id]:
                continue
            if self.get_game().perform_sanity_checks:
                self.assert_valid_action(
                    action, self._vehicle_locations[vehicle_id])
            self._vehicle_locations[vehicle_id] =\
                self.get_game().network.get_road_section_from_action_id(action)
            # Has the vehicle just reached its destination?
            if (self._vehicle_locations[vehicle_id] ==
                    self._vehicle_destinations[vehicle_id]):
                self._vehicle_final_travel_times[vehicle_id] =\
                  self._current_time_step
                self._vehicle_at_destination.add(vehicle_id)
                self._vehicle_without_legal_actions.add(vehicle_id)
            # Will the vehicle have a legal action for next time step?
            if self.get_game().network.is_location_at_sink_node(
                    self._vehicle_locations[vehicle_id]):
                self._vehicle_without_legal_actions.add(vehicle_id)
        # Is the game finished?
        if (self._current_time_step >= self.get_game().max_game_length() or
                len(self._vehicle_without_legal_actions) == self.get_game(
                    ).num_players()):
            self._is_terminal = True
            for vehicle_id in range(self.get_game().num_players()):
                if vehicle_id not in self._vehicle_at_destination:
                    self._vehicle_final_travel_times[vehicle_id] =\
                      - self.get_game().min_utility()

    def _action_to_string(self, player, action) -> str:
        """Action -> string."""
        if player == pyspiel.PlayerId.CHANCE:
            return (f"Change node {action}. I will convert it later to human "
                    "readable chance outcome.")
        if self.get_game().perform_sanity_checks:
            self.assert_valid_player(player)
        if action == NO_POSSIBLE_ACTION:
            return f"Vehicle {player} reach a sink node or its destination."
        if self.get_game().perform_sanity_checks:
            self.assert_valid_action(action)
        return (
            f"Vehicle {player} would like to move to "
            f"{self.get_game().network.get_road_section_from_action_id(action)}"
            ".")

    def is_terminal(self) -> bool:
        """Returns True if the game is over."""
        return self._is_terminal

    def returns(self) -> List[float]:
        """Total reward for each player over the course of the game so far."""
        if not self._is_terminal:
            return [0 for _ in self._vehicle_final_travel_times]
        return [- travel_time
                for travel_time in self._vehicle_final_travel_times]

    def get_current_vehicle_locations(self) -> List[str]:
        """Get vehicle locations for debug purposes."""
        return self._vehicle_locations

    def get_location_as_int(self, vehicle: int) -> int:
        """Get the vehicle location."""
        origin, destination = _road_section_to_nodes(
            self._vehicle_locations[vehicle])
        return self.get_game().network.get_action_id_from_movement(
            origin, destination)

    def get_current_vehicle_locations_as_int(self) -> List[int]:
        """Get locations of all vehicles for the observation tensor."""
        return [self.get_location_as_int(x)
                for x in range(self.get_game().num_players())]

    def __str__(self) -> str:
        """String for debug purposes. No particular semantics are required."""
        return (f"Vehicle locations: {self._vehicle_locations}, "
                f"time: {self._current_time_step}.")


class NetworkObserver:
    """Network observer used by the learning algorithm.

    The state string is the state history string. The state tensor is an array
    of size max_game_length, num_players where each element is the location of
    the vehicle at this time.
    """

    def __init__(self, num_vehicles: int, num_time: int):
        """Initializes an empty observation tensor."""
        shape = (num_time+1, num_vehicles+1)
        self.tensor = np.zeros(np.prod(shape), np.float32)
        self.dict = {"observation": np.reshape(self.tensor, shape)}

    def set_from(self, state, player):
        """Update the state tensor.

        Put the locations of each players in the tensor row corresponding to
        the current time step. Insert the current player location at the
        beginning of the row."""
        vehicles = state.get_current_vehicle_locations_as_int()
        vehicles.insert(0, state.get_location_as_int(player))
        self.dict["observation"][state.current_time_step(), :] =\
            vehicles

    def string_from(self, state, player):
        """Return the state history string."""
        del player
        return state.history_str()


# Register the game with the OpenSpiel library
pyspiel.register_game(_GAME_TYPE, DynamicRoutingGame)
