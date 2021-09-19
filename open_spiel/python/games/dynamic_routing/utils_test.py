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
"""Tests for Python dynamic routing game utils."""

from absl.testing import absltest
from open_spiel.python.games.dynamic_routing import utils


class NetworkTest(absltest.TestCase):
  """Tests for Network class."""

  def setUp(self):
    """Create a network O->A->D for testing."""
    self.network = utils.Network(
      {"O": ["A"], "A": ["D"], "D": []})

  def test_adjacency_list_init(self):
    """Test class instanciation with adjacency list."""
    self.assertEqual(self.network.num_links(), 2)
    self.assertEqual(self.network.get_successors("O"), ["A"])
    self.assertEqual(self.network.get_successors("A"), ["D"])
    self.assertEqual(self.network.get_successors("D"), [])
    self.assertTrue(self.network.is_location_at_sink_node("A->D"))
    self.assertFalse(self.network.is_location_at_sink_node("O->A"))
    self.assertEqual(self.network.get_action_id_from_movement("O", "A"), 1)
    self.assertEqual(self.network.get_action_id_from_movement("A", "D"), 2)
    self.assertEqual(self.network.get_road_section_from_action_id(1), "O->A")
    self.assertEqual(self.network.get_road_section_from_action_id(2), "A->D")

  def test_get_successors_with_wrong_node(self):
    """Test get successors on non existing node."""
    with self.assertRaises(KeyError):
      self.network.get_successors("Z")

  def test_get_action_id_without_connected_nodes(self):
    """Test get actions id on non connected nodes."""
    with self.assertRaises(KeyError):
      self.network.get_action_id_from_movement("O", "D")

  def test_get_action_id_with_wrong_nodes(self):
    """Test get actions id on non existing node."""
    with self.assertRaises(KeyError):
      self.network.get_action_id_from_movement("Z", "D")

  def test_is_location_at_sink_noded_with_wrong_road_section(self):
    """Test is_location_at_sink_node on non existing second node."""
    with self.assertRaises(KeyError):
      self.network.is_location_at_sink_node("A->Z")

  def test_is_location_at_sink_noded_with_wrong_road_section_2(self):
    """Test is_location_at_sink_node on non existing first node."""
    with self.assertRaises(KeyError):
      self.network.is_location_at_sink_node("Z->D")

  def test_is_location_at_sink_noded_with_wrong_arg(self):
    """Test is_location_at_sink_node on wrong link str representation."""
    with self.assertRaises(ValueError):
      self.network.is_location_at_sink_node("D")

  def test_get_road_section_with_action_id(self):
    """Test get_road_section_from_action_id on non possible action."""
    with self.assertRaises(KeyError):
      self.network.get_road_section_from_action_id(0)

  def test_get_probability_to_exit_expected(self):
    """Test get_probability_to_exit with default functions."""
    self.assertEqual(self.network.get_probability_to_exit("O->A", 0), 1)
    self.assertEqual(self.network.get_probability_to_exit("O->A", 1), .5)
    self.assertEqual(self.network.get_probability_to_exit("O->A", 3), .25)
    self.assertEqual(self.network.get_probability_to_exit("A->D", 0), 1)
    self.assertEqual(self.network.get_probability_to_exit("A->D", 1), .5)
    self.assertEqual(self.network.get_probability_to_exit("A->D", 3), .25)

  def test_get_probability_to_exit_wrong_road_section(self):
    """Test get_probability_to_exit with user defined functions."""
    with self.assertRaises(KeyError):
      self.network.get_probability_to_exit("Z->D", 0)


class VehicleTest(absltest.TestCase):
  """Tests for Vehicle class."""

  def test_vehicle_1(self):
    """Test instanciation of Vehicle."""
    vehicle = utils.Vehicle('O->A', 'B->D')
    self.assertEqual(vehicle.destination, 'B->D')
    self.assertEqual(vehicle.origin, 'O->A')
    self.assertEqual(vehicle.departure_time, 0)

  def test_vehicle_2(self):
    """Test instanciation of with departure time."""
    vehicle = utils.Vehicle('O->A', 'B->D', 10.5)
    self.assertEqual(vehicle.origin, 'O->A')
    self.assertEqual(vehicle.destination, 'B->D')
    self.assertEqual(vehicle.departure_time, 10.5)


class OriginDestinationDemandTest(absltest.TestCase):
  """Tests for OriginDestinationDemand class."""

  def test_od_demand_1(self):
    """Test instanciation of OD demand."""
    od_demand = utils.OriginDestinationDemand(
      'O->A', 'B->D', 0, 30)
    self.assertEqual(od_demand.destination, 'B->D')
    self.assertEqual(od_demand.origin, 'O->A')
    self.assertEqual(od_demand.departure_time, 0)
    self.assertEqual(od_demand.counts, 30)

  def test_od_demand_2(self):
    """Test instanciation of OD demand."""
    od_demand = utils.OriginDestinationDemand(
      'O->A', 'B->D', 10.5, 43.2)
    self.assertEqual(od_demand.origin, 'O->A')
    self.assertEqual(od_demand.destination, 'B->D')
    self.assertEqual(od_demand.departure_time, 10.5)
    self.assertEqual(od_demand.counts, 43.2)


if __name__ == "__main__":
  absltest.main()
