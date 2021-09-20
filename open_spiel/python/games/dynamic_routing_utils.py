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
"""Utils module for dynamic routing game and mean field routing game.

This module has three main classes:
- Network
- Vehicle
- OriginDestinationDemand
"""

# pylint: disable=g-bad-todo
# pylint: disable=eval-used

from typing import Dict, Iterable, List, Tuple

# In case one vehicle has reached a end node, then it cannot do anything. In
# this case its action is 0. Action 0 is reserved to encode no possible action
# as requested by Open Spiel.
NO_POSSIBLE_ACTION = 0


def _nodes_to_road_section(origin: str, destination: str) -> str:
  """Create a road section 'A->B' from two nodes 'A' and 'B'."""
  return f"{origin}->{destination}"


def _road_section_to_nodes(movement: str) -> Tuple[str, str]:
  """Split a road section 'A->B' to two nodes 'A' and 'B'."""
  origin, destination = movement.split("->")
  return origin, destination


class Network:
  """Network implementation.

  A network is basically a directed graph with a volume delay function on each
  of its edges. Each vertex is refered to as a string (for example "A") and each
  edge as a string f"{node1}->{node2}" (for example "A->B"). The network is
  created from a adjacency list. Each road section is mapped to an action index
  (positive integer) in _road_section_to_action, and vice versa in
  _action_to_road_section. The volume delay function on each link is given by
  _probability_to_exit_functions.

  If one would like to plot the network then node position should be passed
  in the constructor. Then return_list_for_matplotlib_quiver can be used with
  Matplotlib:
  ```python3
  fig, ax = plt.subplots()
  o_xs, o_ys, d_xs, d_ys = g.return_list_for_matplotlib_quiver()
  ax.quiver(o_xs, o_ys, np.subtract(d_xs, o_xs), np.subtract(d_ys, o_ys),
            color="b", angles='xy', scale_units='xy', scale=1)
  ```

  See the Network tests for an example.
  Attributes:
    _action_to_road_section: dictionary that maps action id to road section.
    _adjacency_list: adjacency list of the line graph of the road network.
    _node_position: dictionary that maps node to couple of float encoding x and
      y position of the node. None by default.
    _probability_to_exit_functions: dictionary of functions as string assigned
      to road sections. A function is a string that will be evaluated with x as
      parameter (for example '1/(1+x)'). Each function takes as input x; the
      volume of cars on the road section, and output the probability that a
      given car exits the road section in the next time step. If average over
      all the cars on the road section, this function gives the volume of cars
      exiting the road section during a given time step as a function of the
      volume of cars on the road section. Such functions are called fundamental
      diagram of traffic flow.
    _road_section_to_action: dictionary that maps road section to action id.
  """
  _action_to_road_section: Dict[int, str]
  _adjacency_list: Dict[str, Iterable[str]]
  _node_position: Dict[str, Tuple[float, float]]
  _probability_to_exit_functions: Dict[str, str]
  _road_section_to_action: Dict[str, int]

  def __init__(self,
               adjacency_list: Dict[str, Iterable[str]],
               node_position: Dict[str, Tuple[float, float]] = None,
               probability_to_exit_functions: Dict[str, str] = None):
    self._adjacency_list = adjacency_list
    self._road_section_to_action, self._action_to_road_section = (
        self._create_movement_to_action_and_action_to_road_section())

    nodes = set(adjacency_list)
    # pylint: disable=g-complex-comprehension
    assert all(destination_node in nodes
               for destination_nodes in self._adjacency_list.values()
               for destination_node in destination_nodes), (
                   "Adjacency list is not correct.")
    if node_position:
      assert set(node_position) == nodes
      self._node_position = node_position
    else:
      self._node_position = None

    if probability_to_exit_functions:
      assert set(probability_to_exit_functions) == set(
          self._road_section_to_action), (
              "Exit functions are not defined for each road sections.")
      self._probability_to_exit_functions = probability_to_exit_functions
    else:
      self._probability_to_exit_functions = {}
      for road_section in self._road_section_to_action:
        self._probability_to_exit_functions[road_section] = "1 / (1+x)"
    assert hasattr(self, "_adjacency_list")
    assert hasattr(self, "_node_position")
    assert hasattr(self, "_probability_to_exit_functions")

  def _create_movement_to_action_and_action_to_road_section(
      self) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Create dictionary that maps movement to action.

    The dictionary that maps movement to action is used to define the action
    from a movement that a vehicle would like to do. The dictionary that maps an
    action to the destintion of the movement is used to move a vehicle that does
    an action to the destination of its movement.
    Returns:
      road_section_to_action: dictionary with key begin a movement for example
        "O->A" and value the action numbers. Action numbers are succesive
        integers indexed from 1.
      action_to_road_section: map an action number to the end node of the
        movement. if road_section_to_action["O->A"] = 0 then,
        action_to_road_section[0] = "O->A"
    """
    road_section_to_action = {}
    action_to_road_section = {}
    action_number = 1
    for origin, successors in self._adjacency_list.items():
      for destination in successors:
        road_section = _nodes_to_road_section(origin, destination)
        if road_section in road_section_to_action:
          # TODO: enable parallel links.
          raise ValueError((
              f"{road_section} exists twice in the adjacency list. The current "
              "network implementation does not enable parallel links."))
        road_section_to_action[road_section] = action_number
        action_to_road_section[action_number] = road_section
        action_number += 1
    return road_section_to_action, action_to_road_section

  def num_links(self) -> int:
    """Returns the number of road sections."""
    return len(self._road_section_to_action)

  def num_actions(self) -> int:
    """Returns the number of possible actions.

    Equal to the number of road section + 1. An action could either be moving to
    a specific road section or not move.
    """
    return 1 + self.num_links()

  def links(self) -> List[str]:
    """Returns the road sections as a list."""
    return list(self._road_section_to_action)

  def get_successors(self, node: str) -> Iterable[str]:
    """Returns the successor nodes of the node."""
    return self._adjacency_list[node]

  def get_action_id_from_movement(self, origin: str, destination: str) -> int:
    """Maps two connected nodes to an action."""
    return self._road_section_to_action[_nodes_to_road_section(
        origin, destination)]

  def get_road_section_from_action_id(self, action_id: int) -> str:
    """Maps a action to the corresponding road section."""
    return self._action_to_road_section[action_id]

  def is_location_at_sink_node(self, road_section: str) -> bool:
    """Returns True if the road section has no successors."""
    start_section, end_section_node = _road_section_to_nodes(road_section)
    if start_section not in self._adjacency_list:
      raise KeyError(f"{start_section} is not a network node.")
    return not self.get_successors(end_section_node)

  def check_list_of_vehicles_is_correct(self, vehicles: Iterable["Vehicle"]):
    """Assert that vehicles have valid origin and destination."""
    for vehicle in vehicles:
      if (vehicle.origin not in self._road_section_to_action or
          vehicle.destination not in self._road_section_to_action):
        raise ValueError(f"Incorrect origin or destination for {vehicle}")

  def check_list_of_od_demand_is_correct(
      self, vehicles: Iterable["OriginDestinationDemand"]):
    """Assert that OD demands have valid origin and destination."""
    for vehicle in vehicles:
      if (vehicle.origin not in self._road_section_to_action or
          vehicle.destination not in self._road_section_to_action):
        raise ValueError(f"Incorrect origin or destination for {vehicle}")

  def __str__(self) -> str:
    return str(self._adjacency_list)

  def get_probability_to_exit(self, road_section: str, volume: float) -> float:
    """Returns probability to exit road_section with volume cars."""

    # TODO: find another way to pass the function.
    # pylint: disable=unused-argument
    def probability_to_exit(x):
      return eval(self._probability_to_exit_functions[road_section])

    prob = probability_to_exit(volume)
    assert 0 <= prob <= 1
    return prob

  def assert_valid_action(self, action: int, road_section: str = None):
    """Assert that an action as a int is valid.

    The action should be a int between 1 and num_actions. In case road_section
    is not None then it is test if the action correspond to going on a road
    section which is a successor of road_section.

    Args:
      action: the action,
      road_section: the road section.
    """
    assert isinstance(action, int), f"{action} is not a int."
    assert 1 <= action < self.num_actions()
    if road_section is not None:
      new_road_section = self.get_road_section_from_action_id(action)
      origin_new_section, end_new_section = _road_section_to_nodes(
          new_road_section)
      _, end_section_node = _road_section_to_nodes(road_section)
      assert end_section_node == origin_new_section, (
          f"The action is not legal, trying to go to {new_road_section} "
          f"from {road_section} without going through {end_section_node}"
          ".")
      successors = self.get_successors(origin_new_section)
      assert end_new_section in successors, (
          f"Invalid action {new_road_section}. It is not a successors of"
          f" {end_section_node}: {successors}.")

  def return_position_of_road_section(self,
                                      road_section: str) -> Tuple[float, float]:
    """Returns position of the middle of theroad section as (x,y)."""
    assert self._node_position is not None, (
        "The network should have node positions in order to be plot.")
    o_link, d_link = _road_section_to_nodes(road_section)
    o_x, o_y = self._node_position[o_link]
    d_x, d_y = self._node_position[d_link]
    return (o_x + d_x) / 2, (o_y + d_y) / 2

  def return_list_for_matplotlib_quiver(
      self) -> Tuple[List[float], List[float], List[float], List[float]]:
    """Returns 4 list of encoding the positions of the road sections.

    ```python3
    fig, ax = plt.subplots()
    o_xs, o_ys, d_xs, d_ys = g.return_list_for_matplotlib_quiver()
    ax.quiver(o_xs, o_ys, np.subtract(d_xs, o_xs), np.subtract(d_ys, o_ys),
              color="b", angles='xy', scale_units='xy', scale=1)
    ```
    will show the network.
    Returns:
      o_xs, o_ys, d_xs, d_ys: list of the start x and y positions and of the end
        x and y postions of each road section. Each element of each list
        corresponds to one road section.
    """
    assert self._node_position is not None, (
        "The network should have node positions in order to be plot.")
    o_xs = []
    o_ys = []
    d_xs = []
    d_ys = []
    for road_section in self._road_section_to_action:
      o_link, d_link = _road_section_to_nodes(road_section)
      o_x, o_y = self._node_position[o_link]
      d_x, d_y = self._node_position[d_link]
      o_xs.append(o_x)
      o_ys.append(o_y)
      d_xs.append(d_x)
      d_ys.append(d_y)
    return o_xs, o_ys, d_xs, d_ys


class Vehicle:
  """A Vehicle is one origin and one destination.

  Both the origin and the destination of the vehicle are road section, therefore
  they are string formatted as "{str}->{str}".
  Attributes:
    destination: destination of the vehicle.
    origin: origin of the vehicle.
    departure_time: departure time of the vehicle.
  """
  _destination: str
  _origin: str
  _departure_time: float

  def __init__(self,
               origin: str,
               destination: str,
               departure_time: float = 0.0):
    assert all("->" in node for node in [origin, destination])
    self._origin = origin
    self._destination = destination
    self._departure_time = departure_time

  @property
  def origin(self) -> str:
    """Returns vehicle's origin."""
    return self._origin

  @property
  def destination(self) -> str:
    """Returns vehicle's destination."""
    return self._destination

  @property
  def departure_time(self) -> float:
    """Returns vehicle's departure time."""
    return self._departure_time

  def __str__(self):
    return (f"Vehicle with origin {self.origin}, destination {self.destination}"
            f" and departure time {self._departure_time}.")


class OriginDestinationDemand(Vehicle):
  """Number of trips from origin to destination for a specific departure time.

  Both the origin and the destination of the vehicle are road section, therefore
  they are string formatted as "{str}->{str}".
  Attributes:
    destination: destination of the vehicles.
    origin: origin of the vehicles.
    departure_time: departure time of the vehicles.
    counts: the number of vehicles with the origin, destination and departure
      time.
  """
  _counts: float

  def __init__(self, origin: str, destination: str, departure_time: float,
               counts: float):
    super().__init__(origin, destination, departure_time)
    self._counts = counts

  @property
  def counts(self) -> float:
    """Returns the number of vehicles in the instance."""
    return self._counts

  def __str__(self):
    return (f"{self._counts} with origin {self.origin}, destination "
            f"{self.destination} and departure time {self._departure_time}.")
