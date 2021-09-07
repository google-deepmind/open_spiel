"""TODO: docstring."""

import random
import time

import matplotlib.pyplot as plt
import numpy as np

import tensorflow.compat.v1 as tf

import pyspiel

from open_spiel.python import rl_environment
from open_spiel.python import policy as policy_module
from open_spiel.python.algorithms import cfr
from open_spiel.python.algorithms import expected_game_score
from open_spiel.python.algorithms import exploitability
from open_spiel.python.algorithms import external_sampling_mccfr as external_mccfr
from open_spiel.python.algorithms import nfsp
from open_spiel.python.algorithms import fictitious_play
from open_spiel.python.games.dynamic_routing import dynamic_routing_game_utils
from open_spiel.python.games.dynamic_routing import dynamic_routing_to_mean_field_game
from open_spiel.python.games.dynamic_routing import mean_field_routing_game
from open_spiel.python.mfg.algorithms import distribution
from open_spiel.python.mfg.algorithms import fictitious_play as mean_field_fictitious_play_module
from open_spiel.python.mfg.algorithms import policy_value
from open_spiel.python.mfg.algorithms import mirror_descent
from open_spiel.python.mfg.algorithms import nash_conv

def create_games(origin, destination, num_vehicles, graph, max_time_step):
  """TODO: docstring."""
  list_of_vehicles = [
    dynamic_routing_game_utils.Vehicle(origin, destination)
    for _ in range(num_vehicles)]
  game = dynamic_routing_to_mean_field_game.DynamicRoutingToMeanFieldGame(
    network=graph, vehicles=list_of_vehicles, max_num_time_step=max_time_step)
  seq_game = pyspiel.convert_to_turn_based(game)
  od_demand = [dynamic_routing_game_utils.OriginDestinationDemand(
    origin, destination, 0, num_vehicles)]
  mfg_game = mean_field_routing_game.MeanFieldRoutingGame(
    network=graph, od_demand=od_demand, max_num_time_step=max_time_step
  )
  return game, seq_game, mfg_game


def plot_network_n_player_game(
  g: dynamic_routing_game_utils.Network, vehicle_locations = None
):
  """
  Plot the network
  Args:
    g: network to plot
  """
  fig, ax = plt.subplots()
  o_xs, o_ys, d_xs, d_ys = g.return_list_for_matplotlib_quiver()
  ax.quiver(o_xs, o_ys, np.subtract(d_xs, o_xs), np.subtract(d_ys, o_ys),
            color="b", angles='xy', scale_units='xy', scale=1)
  ax.set_xlim([np.min(np.concatenate((o_xs, d_xs)))-0.5, np.max(
    np.concatenate((o_xs, d_xs))) + 0.5])
  ax.set_ylim([np.min(np.concatenate((o_ys, d_ys)))-0.5, np.max(
    np.concatenate((o_ys, d_ys))) + 0.5])

  if vehicle_locations is not None:
    num_vehicle = len(vehicle_locations)
    dict_location = {}
    for vehicle_location in vehicle_locations:
      if vehicle_location not in dict_location:
        dict_location[vehicle_location] = 0
      dict_location[vehicle_location] += 0.3 / num_vehicle
    for point, width in dict_location.items():
      circle = plt.Circle(point, width, color='r')
      ax.add_patch(circle)


def plot_network_mean_field_game(
  g: dynamic_routing_game_utils.Network, distribution = None, scaling = 1
):
  """
  Plot the network

  Args:
    g: network to plot
  """
  fig, ax = plt.subplots()
  o_xs, o_ys, d_xs, d_ys = g.return_list_for_matplotlib_quiver()
  ax.quiver(o_xs, o_ys, np.subtract(d_xs, o_xs), np.subtract(d_ys, o_ys),
            color="b", angles='xy', scale_units='xy', scale=1)
  ax.set_xlim([np.min(np.concatenate((o_xs, d_xs)))-0.5, np.max(
    np.concatenate((o_xs, d_xs))) + 0.5])
  ax.set_ylim([np.min(np.concatenate((o_ys, d_ys)))-0.5, np.max(
    np.concatenate((o_ys, d_ys))) + 0.5])

  if distribution is not None:
    for x, prob_of_position in distribution.items():
      point = g.return_position_of_road_section(x)
      width = 0.3 * scaling * prob_of_position
      circle = plt.Circle(point, width, color='r')
      ax.add_patch(circle)


def evolve_n_player_simultaneous_game(game, policy, graph):
  """TODO: docstring."""
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
          random.choices(
            legal_actions, [state_policy[a] for a in legal_actions])[0])
      state.apply_actions(chosen_actions)
    else:
      raise ValueError(
        "State should either be simultaneous node or change node.")
    plot_network_n_player_game(
      graph,
      [graph.return_position_of_road_section(x)
       for x in state.get_current_vehicle_locations()])
  print(f"Travel times: {[-x for x in state.returns()]}")

def evolve_n_player_sequential_game(seq_game, policy, graph, debug=False):
  """TODO: docstring."""
  state = seq_game.new_initial_state()
  while not state.is_terminal():
    legal_actions = state.legal_actions()
    if state.is_chance_node():
      # Sample a chance event outcome.
      outcomes_with_probs = state.chance_outcomes()
      action_list, prob_list = zip(*outcomes_with_probs)
      action = np.random.choice(action_list, p=prob_list)
      if debug:
        print('------------ Change node ------------')
        print(f'Possible chance actions: {outcomes_with_probs}, the one taken: {action}.')
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
          for s in str(state).split('[')[1].split(']')[0].split(', ')]
        if debug:
          print(f"Policy for player {state.current_player()} at location {vehicle_location[state.current_player()]}: {[f'{graph.get_road_section_from_action_id(k)} with probability {v}' for k, v in state_policy.items()]}")
        assert set(state_policy) == set(legal_actions)
        action = random.choices(
          legal_actions, [state_policy[a] for a in legal_actions])
        assert len(action) == 1
        action = action[0]
      else:
        action = random.choice(legal_actions)
      state.apply_action(action)
      vehicle_location = [
        s.replace("'", "")
        for s in str(state).split('[')[1].split(']')[0].split(', ')]
      if debug:
        print(vehicle_location)
      plot_network_n_player_game(
        graph,
        [graph.return_position_of_road_section(x) for x in vehicle_location])
  if debug:
    print(f"Travel times: {[-x for x in state.returns()]}")


def evolve_mean_field_game(
  mfg_game, policy, graph, scaling=1, frequency_printing=1
):
  """TODO: docstring."""
  distribution_mfg = distribution.DistributionPolicy(mfg_game, policy)
  root_state = mfg_game.new_initial_state()
  listing_states = [root_state]

  # plot_network_mean_field_game(graph, {origin: 1})
  i = 0
  while not listing_states[0].is_terminal() and not all(
      state._vehicle_without_legal_action for state in listing_states):
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
          if not str(new_mfg_state) in list_of_state_seen:
            new_listing_states.append(new_mfg_state)
          list_of_state_seen.add(str(new_mfg_state))

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
        if not str(new_mfg_state) in list_of_state_seen:
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
          if not str(new_mfg_state) in list_of_state_seen:
            new_listing_states.append(new_mfg_state)
          list_of_state_seen.add(str(new_mfg_state))
      current_distribution = {}
      for mfg_state in new_listing_states:
        location = mfg_state._vehicle_location
        if location not in current_distribution:
          current_distribution[location] = 0
        current_distribution[location] += distribution_mfg.value(mfg_state)
      assert abs(sum(current_distribution.values()) - 1) < 1e-4, (
        f"{current_distribution}")
      i += 1
      if i % frequency_printing == 0:
        plot_network_mean_field_game(graph, current_distribution, scaling=scaling)
    listing_states = new_listing_states

def uniform_policy_n_player(
  seq_game
):
  """TODO: docstring."""
  return policy_module.UniformRandomPolicy(seq_game)

def ficticious_play(
  seq_game,
  number_of_iterations,
  compute_metrics=False
):
  """TODO: docstring."""
  xfp_solver = fictitious_play.XFPSolver(seq_game)
  tick_time = time.time()
  for i in range(number_of_iterations):
    xfp_solver.iteration()
  timing = time.time() - tick_time
  print('done')
  average_policies = xfp_solver.average_policy_tables()
  tabular_policy = policy_module.TabularPolicy(seq_game)
  if compute_metrics:
    nash_conv = exploitability.nash_conv(seq_game, xfp_solver.average_policy())
    average_policy_values = expected_game_score.policy_value(
      seq_game.new_initial_state(), [tabular_policy])
    return timing, tabular_policy, nash_conv, average_policy_values
  else:
    return timing, tabular_policy

def counterfactual_regret_minimization(
  seq_game,
  number_of_iterations,
  compute_metrics=False
):
  """TODO: docstring."""
  freq_iteration_printing = number_of_iterations//10
  cfr_solver = cfr.CFRSolver(seq_game)
  tick_time = time.time()
  print("CFRSolver initialized.")
  for i in range(number_of_iterations):
    cfr_solver.evaluate_and_update_policy()
    if i % freq_iteration_printing == 0:
      print(f"Iteration {i}")
  timing = time.time() - tick_time
  print("Finish.")
  if compute_metrics:
    nash_conv = exploitability.nash_conv(seq_game, cfr_solver.average_policy())
    return timing, cfr_solver.average_policy(), nash_conv
  return timing, cfr_solver.average_policy()

def external_sampling_Monte_Carlo_counterfactual_regret_minimization(
  seq_game,
  number_of_iterations,
  compute_metrics=False
):
  """TODO: docstring."""
  cfr_solver = external_mccfr.ExternalSamplingSolver(seq_game, external_mccfr.AverageType.SIMPLE)
  tick_time = time.time()
  print("CFRSolver initialized.")
  for i in range(number_of_iterations):
    cfr_solver.iteration()
  timing = time.time() - tick_time
  print("Finish.")
  if compute_metrics:
    nash_conv = exploitability.nash_conv(seq_game, cfr_solver.average_policy())
    return timing, cfr_solver.average_policy(), nash_conv
  return timing, cfr_solver.average_policy()

class NFSPPolicies(policy_module.Policy):
  """Joint policy to be evaluated."""

  def __init__(self, env, nfsp_policies, mode):
    """TODO: docstring."""
    game = env.game
    num_players = env.num_players
    player_ids = list(range(num_players))
    super(NFSPPolicies, self).__init__(game, player_ids)
    self._policies = nfsp_policies
    self._mode = mode
    self._obs = {
      "info_state": [None] * num_players,
      "legal_actions": [None] * num_players
    }

  def action_probabilities(self, state, player_id=None):
    """TODO: docstring."""
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

def neural_ficticious_self_play(
  seq_game,
  num_epoch,
  sess,
  compute_metrics=False
):
  """TODO: docstring."""
  env = rl_environment.Environment(seq_game)
  # Parameters from the game.
  num_players = env.num_players
  num_actions = env.action_spec()['num_actions']
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

  freq_epoch_printing = num_epoch//10
  agents = [nfsp.NFSP(sess, idx, info_state_size, num_actions, 
                      hidden_layers_sizes, **kwargs) 
            for idx in range(num_players)]
  joint_avg_policy = NFSPPolicies(env, agents, nfsp.MODE.average_policy)

  sess.run(tf.global_variables_initializer())
  print("TF initialized.")
  tick_time = time.time()
  for ep in range(num_epoch):
    if ep % freq_epoch_printing == 0:
      print(f"Iteration {ep}")
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
  print("Finish.")
  if compute_metrics:
    tabular_policy = joint_avg_policy.TabularPolicy(seq_game)
    average_policy_values = expected_game_score.policy_value(
      seq_game.new_initial_state(), [tabular_policy])
    nash_conv = exploitability.nash_conv(env.game, joint_avg_policy)
    return timing, joint_avg_policy, average_policy_values, nash_conv
  return timing, joint_avg_policy

def mean_field_uniform_policy(  
  mfg_game,
  number_of_iterations,
  compute_metrics=False
):
  """TODO: docstring."""
  del number_of_iterations
  uniform_policy = policy_module.UniformRandomPolicy(mfg_game)
  if compute_metrics:
    distribution_mfg = distribution.DistributionPolicy(mfg_game, uniform_policy)
    policy_value_ = policy_value.PolicyValue(
      mfg_game, distribution_mfg, uniform_policy).value(mfg_game.new_initial_state())
    return uniform_policy, policy_value_
  return uniform_policy

def mean_field_fictitious_play(
  mfg_game,
  number_of_iterations,
  compute_metrics=False
):
  """TODO: docstring."""
  fp = mean_field_fictitious_play_module.FictitiousPlay(mfg_game)
  tick_time = time.time()
  for i in range(number_of_iterations):
    fp.iteration()
  timing = time.time() - tick_time
  fp_policy = fp.get_policy()
  print('learning done')
  if compute_metrics:
    distribution_mfg = distribution.DistributionPolicy(mfg_game, fp_policy)
    print('distribution done')
    policy_value_ = policy_value.PolicyValue(
      mfg_game, distribution_mfg, fp_policy).value(mfg_game.new_initial_state())
    nash_conv_fp = nash_conv.NashConv(mfg_game, fp_policy)
    return timing, fp_policy, nash_conv_fp, policy_value_
  return timing, fp_policy

def online_mirror_descent(
  mfg_game,
  number_of_iterations,
  compute_metrics=False
):
  md = mirror_descent.MirrorDescent(mfg_game)
  tick_time = time.time()
  for _ in range(number_of_iterations):
    md.iteration()
  timing = time.time() - tick_time
  md_policy = md.get_policy()
  if compute_metrics:
    distribution_mfg = distribution.DistributionPolicy(mfg_game, md_policy)
    print('distribution done')
    policy_value_ = policy_value.PolicyValue(
      mfg_game, distribution_mfg, md_policy).value(mfg_game.new_initial_state())
    nash_conv_md = nash_conv.NashConv(mfg_game, md_policy)
    return timing, md_policy, nash_conv_md, policy_value_
  return timing, md_policy

class RandomPolicyDeviation:
  """TODO: docstring."""

  def __init__(self):
    """TODO: docstring."""
    self.policy_deviation = {}

  def get_policy_deviation(self, state, player_id):
    """TODO: docstring."""
    key = (str(state), player_id)
    if key not in self.policy_deviation:
      assert player_id == state.current_player()
      action_probability = [random.random() for a in state.legal_actions()]
      self.policy_deviation[key] = [
        x/sum(action_probability) for x in action_probability]
    return self.policy_deviation[key]

def get_results_n_player_sequential_game(seq_game, policy):
  """TODO: docstring."""
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
      action = random.choices(
        legal_actions, [state_policy[a] for a in legal_actions])
      assert len(action) == 1
      action = action[0]
    state.apply_action(action)
  return state.returns()

def get_average_results_n_player_game(seq_game, policy, num_sample=10):
  """TODO: docstring."""
  result_array = [get_results_n_player_sequential_game(
    seq_game, policy) for _ in range(num_sample)]
  return sum([sum(i)/len(i) for i in zip(*result_array)])/len(result_array)

def get_results_n_player_sequential_game_with_random_noise(
  seq_game, policy, random_policy_deviation, player_id = 0
):
  """TODO: docstring."""
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
      if state.current_player() == player_id:
        action_probability = random_policy_deviation.get_policy_deviation(
          state, player_id)
      else:
        action_probability = [state_policy[a] for a in legal_actions]
      action = random.choices(legal_actions, action_probability)
      assert len(action) == 1
      action = action[0]
    state.apply_action(action)
  return state.returns()

def get_list_results_n_player_game_with_random_noise(
  seq_game, policy, random_policy_deviation, player_id=0, num_sample=10
):
  """TODO: docstring."""
  return [
    get_results_n_player_sequential_game_with_random_noise(
      seq_game, policy, random_policy_deviation, player_id)
    for _ in range(num_sample)]


def get_average_results_n_player_game_with_random_noise(
  seq_game, policy, random_policy_deviation, player_id=0, num_sample=10
):
  """TODO: docstring."""
  result_array = get_list_results_n_player_game_with_random_noise(
    seq_game, policy, random_policy_deviation, player_id, num_sample)
  return [sum(i)/len(i) for i in zip(*result_array)]

def get_average_regret_with_random_noise(
  seq_game, policy, random_policy_deviation, player_id=0, num_sample=10
):
  """TODO: docstring."""
  # fix the random noise in the policy
  result_array = get_average_results_n_player_game_with_random_noise(
    seq_game, policy, random_policy_deviation, player_id, num_sample)
  player_value = result_array[player_id]
  other_values = [
    result_array[i] for i in range(len(result_array)) if i != player_id]
  other_average_value = sum(other_values) / len(other_values)
  other_max_deviation = max(other_values) - min(other_values)
  gain_with_randomly_changing_policy = other_average_value - player_value

  print(f"Gain by adding fixed random noise to policy: {gain_with_randomly_changing_policy}")
  print(f"Maximum deviation in return due to stochasticity in players returns {other_max_deviation}")
  return gain_with_randomly_changing_policy, other_max_deviation
