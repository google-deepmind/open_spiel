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

"""Utils function for routing game experiment."""
# pylint:disable=too-many-lines,import-error,missing-function-docstring,protected-access,too-many-locals,invalid-name,too-many-arguments,too-many-branches,missing-class-docstring,too-few-public-methods

# pylint:disable=line-too-long
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow.compat.v1 as tf

from open_spiel.python import policy as policy_module
from open_spiel.python import rl_environment
from open_spiel.python.algorithms import cfr
from open_spiel.python.algorithms import expected_game_score
from open_spiel.python.algorithms import exploitability
from open_spiel.python.algorithms import external_sampling_mccfr as external_mccfr
from open_spiel.python.algorithms import fictitious_play
from open_spiel.python.algorithms import nfsp
from open_spiel.python.algorithms import noisy_policy
from open_spiel.python.games import dynamic_routing
from open_spiel.python.games import dynamic_routing_utils
from open_spiel.python.mfg.algorithms import distribution as distribution_module
from open_spiel.python.mfg.algorithms import fictitious_play as mean_field_fictitious_play_module
from open_spiel.python.mfg.algorithms import mirror_descent
from open_spiel.python.mfg.algorithms import nash_conv as nash_conv_module
from open_spiel.python.mfg.algorithms import policy_value
from open_spiel.python.mfg.games import dynamic_routing as mean_field_routing_game
import pyspiel
# pylint:enable=line-too-long


def create_games(origin,
                 destination,
                 num_vehicles,
                 graph,
                 max_time_step,
                 time_step_length=1.0,
                 departure_time=None):
  if departure_time is not None:
    raise NotImplementedError("To do.")
  list_of_vehicles = [
      dynamic_routing_utils.Vehicle(origin, destination)
      for _ in range(num_vehicles)
  ]
  game = dynamic_routing.DynamicRoutingGame(
      {
          "max_num_time_step": max_time_step,
          "time_step_length": time_step_length
      },
      network=graph,
      vehicles=list_of_vehicles)
  seq_game = pyspiel.convert_to_turn_based(game)
  od_demand = [
      dynamic_routing_utils.OriginDestinationDemand(origin, destination, 0,
                                                    num_vehicles)
  ]
  mfg_game = mean_field_routing_game.MeanFieldRoutingGame(
      {
          "max_num_time_step": max_time_step,
          "time_step_length": time_step_length
      },
      network=graph,
      od_demand=od_demand)
  return game, seq_game, mfg_game


def create_braess_network(capacity):
  graph_dict = {
      "A": {
          "connection": {
              "B": {
                  "a": 0,
                  "b": 1.0,
                  "capacity": capacity,
                  "free_flow_travel_time": 0
              }
          },
          "location": [0, 0]
      },
      "B": {
          "connection": {
              "C": {
                  "a": 1.0,
                  "b": 1.0,
                  "capacity": capacity,
                  "free_flow_travel_time": 1.0
              },
              "D": {
                  "a": 0,
                  "b": 1.0,
                  "capacity": capacity,
                  "free_flow_travel_time": 2.0
              }
          },
          "location": [1, 0]
      },
      "C": {
          "connection": {
              "D": {
                  "a": 0,
                  "b": 1.0,
                  "capacity": capacity,
                  "free_flow_travel_time": 0.25
              },
              "E": {
                  "a": 0,
                  "b": 1.0,
                  "capacity": capacity,
                  "free_flow_travel_time": 2.0
              }
          },
          "location": [2, 1]
      },
      "D": {
          "connection": {
              "E": {
                  "a": 1,
                  "b": 1.0,
                  "capacity": capacity,
                  "free_flow_travel_time": 1.0
              }
          },
          "location": [2, -1]
      },
      "E": {
          "connection": {
              "F": {
                  "a": 0,
                  "b": 1.0,
                  "capacity": capacity,
                  "free_flow_travel_time": 0.0
              }
          },
          "location": [3, 0]
      },
      "F": {
          "connection": {},
          "location": [4, 0]
      }
  }
  adjacency_list = {
      key: list(value["connection"].keys())
      for key, value in graph_dict.items()
  }
  bpr_a_coefficient = {}
  bpr_b_coefficient = {}
  capacity = {}
  free_flow_travel_time = {}
  for o_node, value_dict in graph_dict.items():
    for d_node, section_dict in value_dict["connection"].items():
      road_section = dynamic_routing_utils._nodes_to_road_section(
          origin=o_node, destination=d_node)
      bpr_a_coefficient[road_section] = section_dict["a"]
      bpr_b_coefficient[road_section] = section_dict["b"]
      capacity[road_section] = section_dict["capacity"]
      free_flow_travel_time[road_section] = section_dict[
          "free_flow_travel_time"]
  node_position = {key: value["location"] for key, value in graph_dict.items()}
  return dynamic_routing_utils.Network(
      adjacency_list,
      node_position=node_position,
      bpr_a_coefficient=bpr_a_coefficient,
      bpr_b_coefficient=bpr_b_coefficient,
      capacity=capacity,
      free_flow_travel_time=free_flow_travel_time)


def create_augmented_braess_network(capacity):
  graph_dict = {
      "A": {
          "connection": {
              "B": {
                  "a": 0,
                  "b": 1.0,
                  "capacity": capacity,
                  "free_flow_travel_time": 0
              }
          },
          "location": [0, 0]
      },
      "B": {
          "connection": {
              "C": {
                  "a": 1.0,
                  "b": 1.0,
                  "capacity": capacity,
                  "free_flow_travel_time": 1.0
              },
              "D": {
                  "a": 0,
                  "b": 1.0,
                  "capacity": capacity,
                  "free_flow_travel_time": 2.0
              }
          },
          "location": [1, 0]
      },
      "C": {
          "connection": {
              "D": {
                  "a": 0,
                  "b": 1.0,
                  "capacity": capacity,
                  "free_flow_travel_time": 0.25
              },
              "E": {
                  "a": 0,
                  "b": 1.0,
                  "capacity": capacity,
                  "free_flow_travel_time": 2.0
              }
          },
          "location": [2, 1]
      },
      "D": {
          "connection": {
              "E": {
                  "a": 1,
                  "b": 1.0,
                  "capacity": capacity,
                  "free_flow_travel_time": 1.0
              },
              "G": {
                  "a": 0,
                  "b": 1.0,
                  "capacity": capacity,
                  "free_flow_travel_time": 0.0
              }
          },
          "location": [2, -1]
      },
      "E": {
          "connection": {
              "F": {
                  "a": 0,
                  "b": 1.0,
                  "capacity": capacity,
                  "free_flow_travel_time": 0.0
              }
          },
          "location": [3, 0]
      },
      "F": {
          "connection": {},
          "location": [4, 0]
      },
      "G": {
          "connection": {},
          "location": [3, -1]
      }
  }
  adjacency_list = {
      key: list(value["connection"].keys())
      for key, value in graph_dict.items()
  }
  bpr_a_coefficient = {}
  bpr_b_coefficient = {}
  capacity = {}
  free_flow_travel_time = {}
  for o_node, value_dict in graph_dict.items():
    for d_node, section_dict in value_dict["connection"].items():
      road_section = dynamic_routing_utils._nodes_to_road_section(
          origin=o_node, destination=d_node)
      bpr_a_coefficient[road_section] = section_dict["a"]
      bpr_b_coefficient[road_section] = section_dict["b"]
      capacity[road_section] = section_dict["capacity"]
      free_flow_travel_time[road_section] = section_dict[
          "free_flow_travel_time"]
  node_position = {key: value["location"] for key, value in graph_dict.items()}
  return dynamic_routing_utils.Network(
      adjacency_list,
      node_position=node_position,
      bpr_a_coefficient=bpr_a_coefficient,
      bpr_b_coefficient=bpr_b_coefficient,
      capacity=capacity,
      free_flow_travel_time=free_flow_travel_time)


def create_series_parallel_network(num_network_in_series,
                                   time_step_length=1,
                                   capacity=1):
  i = 0
  origin = "A_0->B_0"
  graph_dict = {}
  while i < num_network_in_series:
    tt_up = random.random() + time_step_length
    tt_down = random.random() + time_step_length
    graph_dict.update({
        f"A_{i}": {
            "connection": {
                f"B_{i}": {
                    "a": 0,
                    "b": 1.0,
                    "capacity": capacity,
                    "free_flow_travel_time": time_step_length
                }
            },
            "location": [0 + 3 * i, 0]
        },
        f"B_{i}": {
            "connection": {
                f"C_{i}": {
                    "a": 1.0,
                    "b": 1.0,
                    "capacity": capacity,
                    "free_flow_travel_time": tt_up
                },
                f"D_{i}": {
                    "a": 1.0,
                    "b": 1.0,
                    "capacity": capacity,
                    "free_flow_travel_time": tt_down
                }
            },
            "location": [1 + 3 * i, 0]
        },
        f"C_{i}": {
            "connection": {
                f"A_{i+1}": {
                    "a": 0,
                    "b": 1.0,
                    "capacity": capacity,
                    "free_flow_travel_time": time_step_length
                }
            },
            "location": [2 + 3 * i, 1]
        },
        f"D_{i}": {
            "connection": {
                f"A_{i+1}": {
                    "a": 0,
                    "b": 1.0,
                    "capacity": capacity,
                    "free_flow_travel_time": time_step_length
                }
            },
            "location": [2 + 3 * i, -1]
        }
    })
    i += 1
  graph_dict[f"A_{i}"] = {
      "connection": {
          "END": {
              "a": 0,
              "b": 1.0,
              "capacity": capacity,
              "free_flow_travel_time": time_step_length
          }
      },
      "location": [0 + 3 * i, 0]
  }
  graph_dict["END"] = {"connection": {}, "location": [1 + 3 * i, 0]}
  time_horizon = int(3.0 * (num_network_in_series + 1) / time_step_length)
  destination = f"A_{i}->END"
  adjacency_list = {
      key: list(value["connection"].keys())
      for key, value in graph_dict.items()
  }
  bpr_a_coefficient = {}
  bpr_b_coefficient = {}
  capacity = {}
  free_flow_travel_time = {}
  for o_node, value_dict in graph_dict.items():
    for d_node, section_dict in value_dict["connection"].items():
      road_section = dynamic_routing_utils._nodes_to_road_section(
          origin=o_node, destination=d_node)
      bpr_a_coefficient[road_section] = section_dict["a"]
      bpr_b_coefficient[road_section] = section_dict["b"]
      capacity[road_section] = section_dict["capacity"]
      free_flow_travel_time[road_section] = section_dict[
          "free_flow_travel_time"]
  node_position = {key: value["location"] for key, value in graph_dict.items()}
  return dynamic_routing_utils.Network(
      adjacency_list,
      node_position=node_position,
      bpr_a_coefficient=bpr_a_coefficient,
      bpr_b_coefficient=bpr_b_coefficient,
      capacity=capacity,
      free_flow_travel_time=free_flow_travel_time
  ), origin, destination, time_horizon


def create_sioux_falls_network():
  sioux_falls_adjacency_list = {}
  sioux_falls_node_position = {}
  bpr_a_coefficient = {}
  bpr_b_coefficient = {}
  capacity = {}
  free_flow_travel_time = {}

  content = open("./SiouxFalls_node.csv", "r").read()
  for line in content.split("\n")[1:]:
    row = line.split(",")
    sioux_falls_node_position[row[0]] = [int(row[1]) / 1e5, int(row[2]) / 1e5]
    sioux_falls_node_position[f"bef_{row[0]}"] = [
        int(row[1]) / 1e5, int(row[2]) / 1e5
    ]
    sioux_falls_node_position[f"aft_{row[0]}"] = [
        int(row[1]) / 1e5, int(row[2]) / 1e5
    ]
    sioux_falls_adjacency_list[f"bef_{row[0]}"] = [row[0]]
    sioux_falls_adjacency_list[row[0]] = [f"aft_{row[0]}"]
    sioux_falls_adjacency_list[f"aft_{row[0]}"] = []

    bpr_a_coefficient[f"{row[0]}->aft_{row[0]}"] = 0.0
    bpr_b_coefficient[f"{row[0]}->aft_{row[0]}"] = 1.0
    capacity[f"{row[0]}->aft_{row[0]}"] = 0.0
    free_flow_travel_time[f"{row[0]}->aft_{row[0]}"] = 0.0

    bpr_a_coefficient[f"bef_{row[0]}->{row[0]}"] = 0.0
    bpr_b_coefficient[f"bef_{row[0]}->{row[0]}"] = 1.0
    capacity[f"bef_{row[0]}->{row[0]}"] = 0.0
    free_flow_travel_time[f"bef_{row[0]}->{row[0]}"] = 0.0

  content = open("./SiouxFalls_net.csv", "r").read()
  for l in content.split("\n")[1:-1]:
    _, origin, destination, a0, a1, a2, a3, a4 = l.split(",")
    assert all(int(x) == 0 for x in [a1, a2, a3])
    sioux_falls_adjacency_list[origin].append(destination)
    road_section = f"{origin}->{destination}"
    bpr_a_coefficient[road_section] = float(a4)
    bpr_b_coefficient[road_section] = 4.0
    capacity[road_section] = 1.0
    free_flow_travel_time[road_section] = float(a0)

  sioux_falls_od_demand = []
  content = open("./SiouxFalls_od.csv", "r").read()
  for line in content.split("\n")[1:-1]:
    row = line.split(",")
    sioux_falls_od_demand.append(
        dynamic_routing_utils.OriginDestinationDemand(
            f"bef_{row[0]}->{row[0]}", f"{row[1]}->aft_{row[1]}", 0,
            float(row[2])))

  return dynamic_routing_utils.Network(
      sioux_falls_adjacency_list,
      node_position=sioux_falls_node_position,
      bpr_a_coefficient=bpr_a_coefficient,
      bpr_b_coefficient=bpr_b_coefficient,
      capacity=capacity,
      free_flow_travel_time=free_flow_travel_time), sioux_falls_od_demand


def plot_network_n_player_game(g: dynamic_routing_utils.Network,
                               vehicle_locations=None):
  """Plot the network.

  Args:
    g: network to plot
    vehicle_locations: vehicle location
  """
  _, ax = plt.subplots()
  o_xs, o_ys, d_xs, d_ys = g.return_list_for_matplotlib_quiver()
  ax.quiver(
      o_xs,
      o_ys,
      np.subtract(d_xs, o_xs),
      np.subtract(d_ys, o_ys),
      color="b",
      angles="xy",
      scale_units="xy",
      scale=1)
  ax.set_xlim([
      np.min(np.concatenate((o_xs, d_xs))) - 0.5,
      np.max(np.concatenate((o_xs, d_xs))) + 0.5
  ])
  ax.set_ylim([
      np.min(np.concatenate((o_ys, d_ys))) - 0.5,
      np.max(np.concatenate((o_ys, d_ys))) + 0.5
  ])

  if vehicle_locations is not None:
    num_vehicle = len(vehicle_locations)
    dict_location = {}
    for vehicle_location in vehicle_locations:
      if vehicle_location not in dict_location:
        dict_location[vehicle_location] = 0.0
      dict_location[vehicle_location] += 0.3 / num_vehicle
    for point, width in dict_location.items():
      circle = plt.Circle(point, width, color="r")
      ax.add_patch(circle)


def plot_network_mean_field_game(g: dynamic_routing_utils.Network,
                                 distribution=None,
                                 scaling=1):
  """Plot the network.

  Args:
    g: network to plot
    distribution: the distribution.
    scaling: scaling factor. for plot rendering.
  """
  _, ax = plt.subplots()
  o_xs, o_ys, d_xs, d_ys = g.return_list_for_matplotlib_quiver()
  ax.quiver(
      o_xs,
      o_ys,
      np.subtract(d_xs, o_xs),
      np.subtract(d_ys, o_ys),
      color="b",
      angles="xy",
      scale_units="xy",
      scale=1)
  ax.set_xlim([
      np.min(np.concatenate((o_xs, d_xs))) - 0.5,
      np.max(np.concatenate((o_xs, d_xs))) + 0.5
  ])
  ax.set_ylim([
      np.min(np.concatenate((o_ys, d_ys))) - 0.5,
      np.max(np.concatenate((o_ys, d_ys))) + 0.5
  ])

  if distribution is not None:
    for x, prob_of_position in distribution.items():
      point = g.return_position_of_road_section(x)
      width = 0.3 * scaling * prob_of_position
      circle = plt.Circle(point, width, color="r")
      ax.add_patch(circle)


def evolve_n_player_simultaneous_game(game, policy, graph):
  state = game.new_initial_state()
  i = 0
  while not state.is_terminal():
    i += 1
    if state.is_chance_node():
      # Sample a chance event outcome.
      outcomes_with_probs = state.chance_outcomes()
      action_list, prob_list = zip(*outcomes_with_probs)
      action = np.random.choice(action_list, p=prob_list)
      state.apply_action(action)
    elif state.is_simultaneous_node():
      # Simultaneous node: sample actions for all players.
      chosen_actions = []
      for i in range(game.num_players()):
        legal_actions = state.legal_actions(i)
        state_policy = policy(state, i)
        assert len(legal_actions) == len(state_policy), (
            f"{legal_actions} not same length than {state_policy}")
        chosen_actions.append(
            random.choices(legal_actions,
                           [state_policy[a] for a in legal_actions])[0])
      state.apply_actions(chosen_actions)
    else:
      raise ValueError(
          "State should either be simultaneous node or change node.")
    plot_network_n_player_game(graph, [
        graph.return_position_of_road_section(x)
        for x in state.get_current_vehicle_locations()
    ])
  print(f"Travel times: {[-x for x in state.returns()]}")


def evolve_n_player_sequential_game(seq_game, policy, graph, debug=False):
  state = seq_game.new_initial_state()
  while not state.is_terminal():
    legal_actions = state.legal_actions()
    if state.is_chance_node():
      # Sample a chance event outcome.
      outcomes_with_probs = state.chance_outcomes()
      action_list, prob_list = zip(*outcomes_with_probs)
      action = np.random.choice(action_list, p=prob_list)
      if debug:
        print("------------ Change node ------------")
        print(
            (f"Possible chance actions: {outcomes_with_probs}, the one taken: "
             f"{action}."))
      state.apply_action(action)
    else:
      if debug:
        print("------------ Sequential action node ------------")
        print(state.information_state_tensor())
        print(state.observation_tensor())
        print(state.information_state_string())
      if policy is not None:
        state_policy = policy(state)
        vehicle_location = [
            s.replace("'", "")
            for s in str(state).split("[")[1].split("]")[0].split(", ")
        ]
        if debug:
          print((f"Policy for player {state.current_player()} at location "
                 f"{vehicle_location[state.current_player()]}: ") +
                str([(str(graph.get_road_section_from_action_id(k)) +
                      f"with probability {v}")
                     for k, v in state_policy.items()]))
        assert set(state_policy) == set(legal_actions)
        action = random.choices(legal_actions,
                                [state_policy[a] for a in legal_actions])
        assert len(action) == 1
        action = action[0]
      else:
        action = random.choice(legal_actions)
      state.apply_action(action)
      vehicle_location = [
          s.replace("'", "")
          for s in str(state).split("[")[1].split("]")[0].split(", ")
      ]
      if debug:
        print(vehicle_location)
      plot_network_n_player_game(
          graph,
          [graph.return_position_of_road_section(x) for x in vehicle_location])
  if debug:
    print(f"Travel times: {[-x for x in state.returns()]}")


def evolve_mean_field_game(mfg_game,
                           policy,
                           graph,
                           scaling=1,
                           frequency_printing=1):
  distribution_mfg = distribution_module.DistributionPolicy(mfg_game, policy)
  root_state = mfg_game.new_initial_state()
  listing_states = [root_state]

  # plot_network_mean_field_game(graph, {origin: 1})
  i = 0
  while not listing_states[0].is_terminal() and not all(
      state._vehicle_without_legal_action for state in listing_states):  # pylint:disable=protected-access
    assert abs(sum(map(distribution_mfg.value, listing_states)) - 1) < 1e-4, (
        f"{list(map(distribution_mfg.value, listing_states))}")
    new_listing_states = []
    list_of_state_seen = set()
    # In case chance node:
    if listing_states[0].current_player() == pyspiel.PlayerId.CHANCE:
      for mfg_state in listing_states:
        for action, _ in mfg_state.chance_outcomes():
          new_mfg_state = mfg_state.child(action)
          # Do not append twice the same file.
          if str(new_mfg_state) not in list_of_state_seen:
            new_listing_states.append(new_mfg_state)
          list_of_state_seen.add(str(new_mfg_state))
      current_distribution = {}
      for mfg_state in new_listing_states:
        location = mfg_state._vehicle_location  # pylint:disable=protected-access
        if location not in current_distribution:
          current_distribution[location] = 0
        current_distribution[location] += distribution_mfg.value(mfg_state)
      plot_network_mean_field_game(graph, current_distribution, scaling=scaling)

    # In case mean field node:
    elif listing_states[0].current_player() == pyspiel.PlayerId.MEAN_FIELD:
      for mfg_state in listing_states:
        dist_to_register = mfg_state.distribution_support()

        def get_probability_for_state(str_state):
          try:
            return distribution_mfg.value_str(str_state)
          except ValueError:
            return 0

        dist = [
            get_probability_for_state(str_state)
            for str_state in dist_to_register
        ]
        new_mfg_state = mfg_state.clone()
        new_mfg_state.update_distribution(dist)
        # Do not append twice the same file.
        if str(new_mfg_state) not in list_of_state_seen:
          new_listing_states.append(new_mfg_state)
        list_of_state_seen.add(str(new_mfg_state))

    # In case action node:
    else:
      assert (listing_states[0].current_player() ==
              pyspiel.PlayerId.DEFAULT_PLAYER_ID), "The player id should be 0"
      for mfg_state in listing_states:
        for action, _ in policy.action_probabilities(mfg_state).items():
          new_mfg_state = mfg_state.child(action)
          # Do not append twice the same file.
          if str(new_mfg_state) not in list_of_state_seen:
            new_listing_states.append(new_mfg_state)
          list_of_state_seen.add(str(new_mfg_state))
      current_distribution = {}
      for mfg_state in new_listing_states:
        location = mfg_state._vehicle_location  # pylint:disable=protected-access
        if location not in current_distribution:
          current_distribution[location] = 0
        current_distribution[location] += distribution_mfg.value(mfg_state)
      assert abs(sum(current_distribution.values()) - 1) < 1e-4, (
          f"{current_distribution}")
      i += 1
      if i % frequency_printing == 0:
        plot_network_mean_field_game(
            graph, current_distribution, scaling=scaling)
    listing_states = new_listing_states


def uniform_policy_n_player(seq_game):
  return policy_module.UniformRandomPolicy(seq_game)


def first_action_policy_n_player(seq_game):
  return policy_module.FirstActionPolicy(seq_game)


def ficticious_play(seq_game, number_of_iterations, compute_metrics=False):
  xfp_solver = fictitious_play.XFPSolver(seq_game)
  tick_time = time.time()
  for _ in range(number_of_iterations):
    xfp_solver.iteration()
  timing = time.time() - tick_time
  # print('done')
  # average_policies = xfp_solver.average_policy_tables()
  tabular_policy = policy_module.TabularPolicy(seq_game)
  if compute_metrics:
    nash_conv = exploitability.nash_conv(seq_game, xfp_solver.average_policy())
    average_policy_values = expected_game_score.policy_value(
        seq_game.new_initial_state(), [tabular_policy])
    return timing, tabular_policy, nash_conv, average_policy_values
  return timing, tabular_policy


def counterfactual_regret_minimization(seq_game,
                                       number_of_iterations,
                                       compute_metrics=False):
  # freq_iteration_printing = number_of_iterations // 10
  cfr_solver = cfr.CFRSolver(seq_game)
  tick_time = time.time()
  # print("CFRSolver initialized.")
  for _ in range(number_of_iterations):
    cfr_solver.evaluate_and_update_policy()
    # if i % freq_iteration_printing == 0:
    #   print(f"Iteration {i}")
  timing = time.time() - tick_time
  # print("Finish.")
  if compute_metrics:
    nash_conv = exploitability.nash_conv(seq_game, cfr_solver.average_policy())
    return timing, cfr_solver.average_policy(), nash_conv
  return timing, cfr_solver.average_policy()


def external_sampling_monte_carlo_counterfactual_regret_minimization(
    seq_game, number_of_iterations, compute_metrics=False):
  cfr_solver = external_mccfr.ExternalSamplingSolver(
      seq_game, external_mccfr.AverageType.SIMPLE)
  tick_time = time.time()
  # print("CFRSolver initialized.")
  for _ in range(number_of_iterations):
    cfr_solver.iteration()
  timing = time.time() - tick_time
  # print("Finish.")
  if compute_metrics:
    nash_conv = exploitability.nash_conv(seq_game, cfr_solver.average_policy())
    return timing, cfr_solver.average_policy(), nash_conv
  return timing, cfr_solver.average_policy()


class NFSPPolicies(policy_module.Policy):
  """Joint policy to be evaluated."""

  def __init__(self, env, nfsp_policies, mode):
    game = env.game
    num_players = env.num_players
    player_ids = list(range(num_players))
    super().__init__(game, player_ids)
    self._policies = nfsp_policies
    self._mode = mode
    self._obs = {
        "info_state": [None] * num_players,
        "legal_actions": [None] * num_players
    }

  def action_probabilities(self, state, player_id=None):
    del player_id
    cur_player = state.current_player()
    legal_actions = state.legal_actions(cur_player)

    self._obs["current_player"] = cur_player
    self._obs["info_state"][cur_player] = (
        state.information_state_tensor(cur_player))
    self._obs["legal_actions"][cur_player] = legal_actions

    info_state = rl_environment.TimeStep(
        observations=self._obs, rewards=None, discounts=None, step_type=None)

    with self._policies[cur_player].temp_mode_as(self._mode):
      p = self._policies[cur_player].step(info_state, is_evaluation=True).probs
    prob_dict = {action: p[action] for action in legal_actions}
    return prob_dict


def neural_ficticious_self_play(seq_game,
                                num_epoch,
                                sess,
                                compute_metrics=False):
  env = rl_environment.Environment(seq_game)
  # Parameters from the game.
  num_players = env.num_players
  num_actions = env.action_spec()["num_actions"]
  info_state_size = env.observation_spec()["info_state"][0]

  # Parameters for the algorithm.
  hidden_layers_sizes = [int(l) for l in [128]]

  kwargs = {
      "replay_buffer_capacity": int(2e5),
      "reservoir_buffer_capacity": int(2e6),
      "min_buffer_size_to_learn": 1000,
      "anticipatory_param": 0.1,
      "batch_size": 128,
      "learn_every": 64,
      "rl_learning_rate": 0.01,
      "sl_learning_rate": 0.01,
      "optimizer_str": "sgd",
      "loss_str": "mse",
      "update_target_network_every": 19200,
      "discount_factor": 1.0,
      "epsilon_decay_duration": int(20e6),
      "epsilon_start": 0.06,
      "epsilon_end": 0.001,
  }

  # freq_epoch_printing = num_epoch // 10
  agents = [
      nfsp.NFSP(sess, idx, info_state_size, num_actions, hidden_layers_sizes,
                **kwargs) for idx in range(num_players)
  ]
  joint_avg_policy = NFSPPolicies(env, agents, nfsp.MODE.average_policy)

  sess.run(tf.global_variables_initializer())
  # print("TF initialized.")
  tick_time = time.time()
  for _ in range(num_epoch):
    # if ep % freq_epoch_printing == 0:
    #   print(f"Iteration {ep}")
    time_step = env.reset()
    while not time_step.last():
      player_id = time_step.observations["current_player"]
      agent_output = agents[player_id].step(time_step)
      action_list = [agent_output.action]
      time_step = env.step(action_list)

    # Episode is over, step all agents with final info state.
    for agent in agents:
      agent.step(time_step)
  timing = time.time() - tick_time
  # print("Finish.")
  if compute_metrics:
    tabular_policy = joint_avg_policy.TabularPolicy(seq_game)
    average_policy_values = expected_game_score.policy_value(
        seq_game.new_initial_state(), [tabular_policy])
    nash_conv = exploitability.nash_conv(env.game, joint_avg_policy)
    return timing, joint_avg_policy, average_policy_values, nash_conv
  return timing, joint_avg_policy


def mean_field_uniform_policy(mfg_game,
                              number_of_iterations,
                              compute_metrics=False):
  del number_of_iterations
  uniform_policy = policy_module.UniformRandomPolicy(mfg_game)
  if compute_metrics:
    distribution_mfg = distribution_module.DistributionPolicy(
        mfg_game, uniform_policy)
    policy_value_ = policy_value.PolicyValue(mfg_game, distribution_mfg,
                                             uniform_policy).value(
                                                 mfg_game.new_initial_state())
    return uniform_policy, policy_value_
  return uniform_policy


def mean_field_fictitious_play(mfg_game,
                               number_of_iterations,
                               compute_metrics=False):
  fp = mean_field_fictitious_play_module.FictitiousPlay(mfg_game)
  tick_time = time.time()
  for _ in range(number_of_iterations):
    fp.iteration()
  timing = time.time() - tick_time
  fp_policy = fp.get_policy()
  # print('learning done')
  if compute_metrics:
    distribution_mfg = distribution_module.DistributionPolicy(
        mfg_game, fp_policy)
    # print('distribution done')
    policy_value_ = policy_value.PolicyValue(mfg_game, distribution_mfg,
                                             fp_policy).value(
                                                 mfg_game.new_initial_state())
    nash_conv_fp = nash_conv_module.NashConv(mfg_game, fp_policy)
    return timing, fp_policy, nash_conv_fp, policy_value_
  return timing, fp_policy


def online_mirror_descent(mfg_game,
                          number_of_iterations,
                          compute_metrics=False,
                          return_policy=False,
                          md_p=None):
  md = md_p if md_p else mirror_descent.MirrorDescent(mfg_game)
  tick_time = time.time()
  for _ in range(number_of_iterations):
    md.iteration()
  timing = time.time() - tick_time
  md_policy = md.get_policy()
  if compute_metrics:
    distribution_mfg = distribution_module.DistributionPolicy(
        mfg_game, md_policy)
    # print('distribution done')
    policy_value_ = policy_value.PolicyValue(mfg_game, distribution_mfg,
                                             md_policy).value(
                                                 mfg_game.new_initial_state())
    nash_conv_md = nash_conv_module.NashConv(mfg_game, md_policy)
    if return_policy:
      return timing, md_policy, nash_conv_md, policy_value_, md
    return timing, md_policy, nash_conv_md, policy_value_
  return timing, md_policy


class RandomPolicyDeviation:

  def __init__(self):
    self.policy_deviation = {}

  def get_policy_deviation(self, state, player_id):
    key = (str(state), player_id)
    if key not in self.policy_deviation:
      assert player_id == state.current_player()
      action_probability = [random.random() for a in state.legal_actions()]
      self.policy_deviation[key] = [
          x / sum(action_probability) for x in action_probability
      ]
    return self.policy_deviation[key]


def get_results_n_player_sequential_game(seq_game, policy):
  state = seq_game.new_initial_state()
  while not state.is_terminal():
    legal_actions = state.legal_actions()
    if state.is_chance_node():
      outcomes_with_probs = state.chance_outcomes()
      action_list, prob_list = zip(*outcomes_with_probs)
      action = np.random.choice(action_list, p=prob_list)
    else:
      state_policy = policy(state)
      assert set(state_policy) == set(legal_actions)
      action = random.choices(legal_actions,
                              [state_policy[a] for a in legal_actions])
      assert len(action) == 1
      action = action[0]
    state.apply_action(action)
  return state.returns()


def get_list_results_n_player_game(seq_game, policy, num_sample=10):
  return [
      get_results_n_player_sequential_game(seq_game, policy)
      for _ in range(num_sample)
  ]


def get_average_results_n_player_game(seq_game, policy, num_sample=10):
  result_array = get_list_results_n_player_game(seq_game, policy, num_sample)
  return sum([sum(i) / len(i) for i in zip(*result_array)]) / len(result_array)


def get_results_n_player_simultaneous_game(game, policy):
  state = game.new_initial_state()
  i = 0
  while not state.is_terminal():
    i += 1
    if state.is_chance_node():
      # Sample a chance event outcome.
      outcomes_with_probs = state.chance_outcomes()
      action_list, prob_list = zip(*outcomes_with_probs)
      action = np.random.choice(action_list, p=prob_list)
      state.apply_action(action)
    elif state.is_simultaneous_node():
      # Simultaneous node: sample actions for all players.
      chosen_actions = []
      for i in range(game.num_players()):
        legal_actions = state.legal_actions(i)
        state_policy = policy(state, player_id=i)
        assert abs(sum([state_policy[a] for a in legal_actions]) - 1) < 1e-4
        chosen_actions.append(
            random.choices(legal_actions,
                           [state_policy[a] for a in legal_actions])[0])
      state.apply_actions(chosen_actions)
    else:
      raise ValueError(
          "State should either be simultaneous node or change node.")
  return state.returns()


def get_list_results_n_player_simulataneous_game(game, policy, num_sample=10):
  return [
      get_results_n_player_simultaneous_game(game, policy)
      for _ in range(num_sample)
  ]


def get_expected_value(seq_game, policy, num_sample, player=0):
  results = get_list_results_n_player_game(
      seq_game, policy, num_sample=num_sample)
  expected_value = sum(x[player] for x in results) / num_sample
  # num_vehicle = len(results[0])
  # error_bar = abs(sum([x[1] for x in results]) - sum(
  # [x[2] for x in results])) / num_sample_trajectories
  # expected_value_policy = sum(sum(x[i] for x in results) for i in range(
  # 1, BRAESS_NUM_VEHICLES)) / ((BRAESS_NUM_VEHICLES-1)*num_sample_trajectories)
  return expected_value


def compute_regret_policy(game,
                          policy,
                          num_random_policy_tested=10,
                          num_sample=100):
  time_tick = time.time()
  expected_value_policy = get_expected_value(game, policy, num_sample)
  worse_regret = 0
  for _ in range(num_random_policy_tested):
    noisy_n_policy = noisy_policy.NoisyPolicy(policy, player_id=0, alpha=1)
    expected_value_noise = get_expected_value(
        game, noisy_n_policy, num_sample, player=0)
    approximate_regret = expected_value_noise - expected_value_policy
    worse_regret = max(worse_regret, approximate_regret)
  return worse_regret, time.time() - time_tick


def get_expected_value_sim_game(game, policy, num_sample, player=0):
  results = get_list_results_n_player_simulataneous_game(
      game, policy, num_sample=num_sample)
  assert len(results) == num_sample
  expected_value = sum(x[player] for x in results) / num_sample
  # num_vehicle = len(results[0])
  # error_bar = abs(sum([x[1] for x in results]) - sum(
  # [x[2] for x in results])) / num_sample_trajectories
  # expected_value_policy = sum(sum(x[i] for x in results) for i in range(
  # 1, BRAESS_NUM_VEHICLES)) / ((BRAESS_NUM_VEHICLES-1)*num_sample_trajectories)
  return expected_value


def compute_regret_policy_random_noise_sim_game(game,
                                                policy,
                                                num_random_policy_tested=10,
                                                num_sample=100):
  time_tick = time.time()
  expected_value_policy = get_expected_value_sim_game(game, policy, num_sample)
  worse_regret = 0
  for _ in range(num_random_policy_tested):
    noisy_n_policy = noisy_policy.NoisyPolicy(policy, player_id=0, alpha=1)
    expected_value_noise = get_expected_value_sim_game(
        game, noisy_n_policy, num_sample, player=0)
    approximate_regret = expected_value_noise - expected_value_policy
    worse_regret = max(worse_regret, approximate_regret)
  return worse_regret, time.time() - time_tick


class PurePolicyResponse(policy_module.Policy):

  def __init__(self, game, policy, player_id):
    self.game = game
    self.player_id = player_id
    self.policy = policy

  def pure_action(self, state):
    raise NotImplementedError()

  def action_probabilities(self, state, player_id=None):
    assert player_id is not None
    if player_id == self.player_id:
      legal_actions = state.legal_actions(self.player_id)
      if not legal_actions:
        return {0: 1.0}
      if len(legal_actions) == 1:
        return {legal_actions[0]: 1.0}
      answer = {action: 0.0 for action in legal_actions}
      pure_a = self.pure_action(state)
      assert pure_a in answer
      answer[pure_a] = 1.0
      return answer
    return self.policy.action_probabilities(state, player_id)


class PathBCEResponse(PurePolicyResponse):

  def pure_action(self, state):
    location = state.get_current_vehicle_locations()[self.player_id].split(
        "->")[1]
    if location == "B":
      return state.get_game().network.get_action_id_from_movement("B", "C")
    if location == "C":
      return state.get_game().network.get_action_id_from_movement("C", "E")
    return 0


class PathBCDEResponse(PurePolicyResponse):

  def pure_action(self, state):
    location = state.get_current_vehicle_locations()[self.player_id].split(
        "->")[1]
    if location == "B":
      return state.get_game().network.get_action_id_from_movement("B", "C")
    if location == "C":
      return state.get_game().network.get_action_id_from_movement("C", "D")
    return 0


class PathBDEResponse(PurePolicyResponse):

  def pure_action(self, state):
    location = state.get_current_vehicle_locations()[self.player_id].split(
        "->")[1]
    if location == "B":
      return state.get_game().network.get_action_id_from_movement("B", "D")
    return 0


def compute_regret_policy_against_pure_policy_sim_game(game,
                                                       policy,
                                                       compute_true_value=False,
                                                       num_sample=100):
  time_tick = time.time()
  if compute_true_value:
    expected_value_policy = expected_game_score.policy_value(
        game.new_initial_state(), policy)[0]
  else:
    expected_value_policy = get_expected_value_sim_game(game, policy,
                                                        num_sample)
  worse_regret = 0
  policies = [
      PathBCEResponse(game, policy, 0),
      PathBCDEResponse(game, policy, 0),
      PathBDEResponse(game, policy, 0)
  ]
  for deviation_policy in policies:
    if compute_true_value:
      expected_value_noise = expected_game_score.policy_value(
          game.new_initial_state(), deviation_policy)[0]
    else:
      expected_value_noise = get_expected_value_sim_game(
          game, deviation_policy, num_sample, player=0)
    approximate_regret = expected_value_noise - expected_value_policy
    worse_regret = max(worse_regret, approximate_regret)
  return worse_regret, time.time() - time_tick


def online_mirror_descent_sioux_falls(mfg_game,
                                      number_of_iterations,
                                      md_p=None):
  nash_conv_dict = {}
  md = md_p if md_p else mirror_descent.MirrorDescent(mfg_game)
  tick_time = time.time()
  for i in range(number_of_iterations):
    md.iteration()
    md_policy = md.get_policy()
    nash_conv_md = nash_conv_module.NashConv(mfg_game, md_policy)
    nash_conv_dict[i] = nash_conv_md.nash_conv()
    print((f"Iteration {i}, Nash conv: {nash_conv_md.nash_conv()}, "
           "time: {time.time() - tick_time}"))
  timing = time.time() - tick_time
  md_policy = md.get_policy()
  distribution_mfg = distribution_module.DistributionPolicy(mfg_game, md_policy)
  policy_value_ = policy_value.PolicyValue(
      mfg_game, distribution_mfg, md_policy).value(mfg_game.new_initial_state())
  nash_conv_md = nash_conv_module.NashConv(mfg_game, md_policy)
  return timing, md_policy, nash_conv_md, policy_value_, md, nash_conv_dict
