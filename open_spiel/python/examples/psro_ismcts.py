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


"""Example use of ISMCTS as a best response oracle in PSRO.

PSRO-ISMCTS demonstrates that online planning algorithms are compatiable with the population-based PSRO paradigm.
It potentially fits for large-scale perfect-information games.
On kuhn poker nashconv decreases from 1.0 to ~0.15 in 50 PSRO iterations.
"""

import time

from absl import app
from absl import flags

import pyspiel

from open_spiel.python.algorithms import policy_aggregator
from open_spiel.python.algorithms.mcts import RandomRolloutEvaluator
from open_spiel.python.algorithms.psro_v2 import optimization_oracle
from open_spiel.python.algorithms.psro_v2 import utils
from open_spiel.python.algorithms.psro_v2 import psro_v2
from open_spiel.python.algorithms import best_response as pyspiel_best_response
from open_spiel.python.policy import Policy


import copy
from enum import Enum

import numpy as np


FLAGS = flags.FLAGS

# ISMCTS related
flags.DEFINE_float("uct_c", 2, "uct_c for ISMCTS")
flags.DEFINE_integer("rollout_count", 1, "rollout count for ISMCTS")
flags.DEFINE_integer("max_simulations", 100,
                     "max iterations for ISMCTS simulations phase")
flags.DEFINE_integer("seed", 17, "random seed")

# PSRO related
flags.DEFINE_integer("psro_sims_per_entry", 1,
                     "simulation numbers for meta-game entry")
flags.DEFINE_integer("psro_iterations", 100, "number of PSRO iterations")


# ------------------adapted from C++ IS-MCTS implementations-------------

UNLIMITED_NUM_WORLD_SAMPLES = -1
UNEXPANDED_VISIT_COUNT = -1
TIE_TOLERANCE = 1e-5


class ISMCTSFinalPolicyType(Enum):
  NORMALIZED_VISITED_COUNT = 1
  MAX_VISIT_COUNT = 2
  MAX_VALUE = 3


class ChildSelectionPolicy(Enum):
  UCT = 1
  PUCT = 2


class ChildInfo(object):
  def __init__(self, visits, return_sum, prior):
    self.visits = visits
    self.return_sum = return_sum
    self.prior = prior

  def value(self):
    return self.return_sum / self.visits


class ISMCTSNode(object):
  def __init__(self):
    self.child_info = {}
    self.total_visits = 0
    self.prior_map = {}


class ISMCTSBot(pyspiel.Bot):
  def __init__(self, game, evaluator, uct_c, max_simulations,
               max_world_samples=UNLIMITED_NUM_WORLD_SAMPLES,
               random_state=None,
               final_policy_type=ISMCTSFinalPolicyType.NORMALIZED_VISITED_COUNT,
               use_observation_string=False,
               allow_inconsistent_action_sets=False,
               child_selection_policy=ChildSelectionPolicy.PUCT):

    pyspiel.Bot.__init__(self)
    self._game = game
    self._evaluator = evaluator
    self._uct_c = uct_c
    self._max_simulations = max_simulations
    self._max_world_samples = max_world_samples
    self._final_policy_type = final_policy_type
    self._use_observation_string = use_observation_string
    self._allow_inconsistent_action_sets = allow_inconsistent_action_sets
    self._nodes = {}
    self._node_pool = []
    self._root_samples = []
    self._random_state = random_state or np.random.RandomState()
    self._child_selection_policy = child_selection_policy
    self._resampler_cb = None

  def random_number(self):
    return self._random_state.uniform()

  def reset(self):
    self._nodes = {}
    self._node_pool = []
    self._root_samples = []

  def get_state_key(self, state):
    if self._use_observation_string:
      return state.current_player(), state.observation_string()
    else:
      return state.current_player(), state.information_state_string()

  def run_search(self, state):
    self.reset()
    assert state.get_game().get_type().dynamics == pyspiel.GameType.Dynamics.SEQUENTIAL
    assert state.get_game().get_type(
    ).information == pyspiel.GameType.Information.IMPERFECT_INFORMATION

    legal_actions = state.legal_actions()
    if len(legal_actions) == 1:
      return [(legal_actions[0], 1.0)]

    self._root_node = self.create_new_node(state)

    assert self._root_node

    root_infostate_key = self.get_state_key(state)

    for sim in range(self._max_simulations):
      # how to sample a pyspiel.state from another pyspiel.state?
      sampled_root_state = self.sample_root_state(state)
      assert root_infostate_key == self.get_state_key(sampled_root_state)
      assert sampled_root_state
      self.run_simulation(sampled_root_state)

    if self._allow_inconsistent_action_sets:  # when this happens?
      legal_actions = state.legal_actions()
      temp_node = self.filter_illegals(self._root_node, legal_actions)
      assert temp_node.total_visits > 0
      return self.get_final_policy(state, temp_node)
    else:
      return self.get_final_policy(state, self._root_node)

  def step(self, state):
    action_list, prob_list = zip(*self.run_search(state))
    return self._random_state.choice(action_list, p=prob_list)

  def get_policy(self, state):
    return self.run_search(state)

  def step_with_policy(self, state):
    policy = self.get_policy(state)
    action_list, prob_list = zip(*policy)
    sampled_action = self._random_state.choice(action_list, p=prob_list)
    return policy, sampled_action

  def get_final_policy(self, state, node):
    assert node
    if self._final_policy_type == ISMCTSFinalPolicyType.NORMALIZED_VISITED_COUNT:
      assert node.total_visits > 0
      total_visits = node.total_visits
      policy = [(action, child.visits/total_visits)
                for action, child in node.child_info.items()]
    elif self._final_policy_type == ISMCTSFinalPolicyType.MAX_VISIT_COUNT:
      assert node.total_visits > 0
      max_visits = -float('inf')
      count = 0
      for action, child in node.child_info.items():
        if child.visits == max_visits:
          count += 1
        elif child.visits > max_visits:
          max_visits = child.visits
          count = 1
      policy = [(action, 1./count if child.visits == max_visits else 0.0)]
    elif self._final_policy_type == ISMCTSFinalPolicyType.MAX_VALUE:
      assert node.total_visits > 0
      max_value = -float('inf')
      count = 0
      for action, child in node.child_info.items():
        if child.value() == max_value:
          count += 1
        elif child.value() > max_value:
          max_value = child.value()
          count = 1
      policy = [(action, 1./count if child.value() == max_value else 0.0)]

    policy_size = len(policy)
    legal_actions = state.legal_actions()
    if policy_size < len(legal_actions):   # do we really need this step?
      for action in legal_actions:
        if action not in node.child_info:
          policy.append((action, 0.0))
    return policy

  def sample_root_state(self, state):
    if self._max_world_samples == UNLIMITED_NUM_WORLD_SAMPLES:
      return self.resample_from_infostate(state)
    elif len(self._root_samples) < self._max_world_samples:
      self._root_samples.append(self.resample_from_infostate(state))
      return self._root_samples[-1].clone()
    elif len(self._root_samples) == self._max_world_samples:
      idx = self._random_state.randint(len(self._root_samples))
      return self._root_samples[idx].clone()
    else:
      raise pyspiel.SpielError(
          "Case not handled (badly set max_world_samples..?)")

  def resample_from_infostate(self, state):
    if self._resampler_cb:
      return self._resampler_cb(state, state.current_player())
    else:
      return state.resample_from_infostate(state.current_player(), pyspiel.UniformProbabilitySampler(0., 1.))

  def create_new_node(self, state):
    infostate_key = self.get_state_key(state)
    self._node_pool.append(ISMCTSNode())
    node = self._node_pool[-1]
    self._nodes[infostate_key] = node
    node.total_visits = UNEXPANDED_VISIT_COUNT
    return node

  def set_resampler(self, cb):
    self._resampler_cb = cb

  def lookup_node(self, state):
    if self.get_state_key(state) in self._nodes:
      return self._nodes[self.get_state_key(state)]
    return None

  def lookup_or_create_node(self, state):
    node = self.lookup_node(state)
    if node:
      return node
    return self.create_new_node(state)

  def filter_illeals(self, node, legal_actions):
    new_node = copy.deepcopy(node)
    for action, child in node.child_info.items():
      if action not in legal_actions:
        new_node.total_visits -= child.visits
        del new_node.child_info[action]
    return new_node

  def expand_if_necessary(self, node, action):
    if action not in node.child_info:
      node.child_info[action] = ChildInfo(0.0, 0.0, node.prior_map[action])

  def select_action_tree_policy(self, node, legal_actions):
    if self._allow_inconsistent_action_sets:
      temp_node = self.filter_illegals(node, legal_actions)
      if temp_node.total_visits == 0:
        action = legal_actions[self._random_state.randint(
            len(legal_actions))]  # prior?
        self.expand_if_necessary(node, action)
        return action
      else:
        return self.select_action(temp_node)
    else:
      return self.select_action(node)

  def select_action(self, node):
    candidates = []
    max_value = -float('inf')
    for action, child in node.child_info.items():
      assert child.visits > 0

      action_value = child.value()
      if self._child_selection_policy == ChildSelectionPolicy.UCT:
        action_value += self._uct_c * \
            np.sqrt(np.log(node.total_visits)/child.visits)
      elif self._child_selection_policy == ChildSelectionPolicy.PUCT:
        action_value += self._uct_c * child.prior * \
            np.sqrt(node.total_visits)/(1 + child.visits)
      else:
        raise pyspiel.SpielError("Child selection policy unrecognized.")
      if action_value > max_value + TIE_TOLERANCE:
        candidates = [action]
        max_value = action_value
      elif action_value > max_value - TIE_TOLERANCE and action_value < max_value + TIE_TOLERANCE:
        candidates.append(action)
        max_value = action_value

    assert len(candidates) >= 1
    return candidates[self._random_state.randint(len(candidates))]

  def check_expand(self, node, legal_actions):
    if not self._allow_inconsistent_action_sets and len(node.child_info) == len(legal_actions):
      return pyspiel.INVALID_ACTION
    legal_actions_copy = copy.deepcopy(legal_actions)
    self._random_state.shuffle(legal_actions_copy)
    for action in legal_actions_copy:
      if action not in node.child_info:
        return action
    return pyspiel.INVALID_ACTION

  def run_simulation(self, state):
    if state.is_terminal():
      return state.returns()
    elif state.is_chance_node():
      action_list, prob_list = zip(*state.chance_outcomes())
      chance_action = self._random_state.choice(action_list, p=prob_list)
      state.apply_action(chance_action)
      return self.run_simulation(state)
    legal_actions = state.legal_actions()
    cur_player = state.current_player()
    node = self.lookup_or_create_node(state)

    assert node

    if node.total_visits == UNEXPANDED_VISIT_COUNT:
      node.total_visits = 0
      for action, prob in self._evaluator.prior(state):
        node.prior_map[action] = prob
      return self._evaluator.evaluate(state)
    else:
      chosen_action = self.check_expand(
          node, legal_actions)  # add one children at a time?
      if chosen_action != pyspiel.INVALID_ACTION:
        # check if all actions have been expanded, if not, select one?, if yes, ucb?
        self.expand_if_necessary(node, chosen_action)
      else:
        chosen_action = self.select_action_tree_policy(node, legal_actions)

      assert chosen_action != pyspiel.INVALID_ACTION

      node.total_visits += 1
      node.child_info[chosen_action].visits += 1
      state.apply_action(chosen_action)
      returns = self.run_simulation(state)
      node.child_info[chosen_action].return_sum += returns[cur_player]
      return returns


# ------------------end of python adaption----------------------------

class AgainstMixtureISMCTSBot(ISMCTSBot):
  """An ISMCTS bot that conducts planning against a fixed mixture of opponents.

  Args:
    player_id: the player id that this bot corresponds to in a game
    player_policies: if is_joint, then a list of lists each containing num_players policies, 
    representing joint-policy profiles. Otherwise a list of lists each containing individual policies for each num_players
    weights: lists of arrays. it corresponds to the mixture of player_policies
    is_joint: a boolean indicating whether player_policies are represented as joint profile or not
    epsilon: a trembling-hand parameters dealing with zero-reaching cases
  """

  def __init__(self,
               player_id,
               player_policies,
               weights,
               is_joint=False,
               epsilon=1e-40,
               **ismcts_base_params):
    ISMCTSBot.__init__(self, **ismcts_base_params)

    self._player_id = player_id
    self._player_policies = player_policies
    self._weights = weights
    self._is_joint = is_joint
    self._epsilon = epsilon

  def get_root_state_distributions(self, state):
    """Computes correct posterior (state, opponent-sampling weights) distribution.

      Using BFS search to compute the probabilities.
      Args:
        state: a game state
      Returns:
        a list of (state_p, weights, player_weights)
        all state_p have the same infostate of the input state
        weights are the corresponding un-normalized sampling weights
        player_weights are the corresponding posterior sampling weights for this state_p 
    """

    player = state.current_player()
    infostate_action_map = {}
    working_state = self._game.new_initial_state()
    history = state.history()
    for action in history:
      if working_state.current_player() == player:
        infostate_action_map[working_state.information_state_string()] = action
      working_state.apply_action(action)
    infostate_action_map[state.information_state_string()
                         ] = pyspiel.INVALID_ACTION

    queue = [(self._game.new_initial_state(),
              1.0, copy.deepcopy(self._weights))]
    final_distribution = []

    while len(queue):
      i = 0
      while i < len(queue):
        cur_state = queue[i][0].clone()
        if cur_state.is_chance_node():
          for action, prob in cur_state.chance_outcomes():
            queue.append(
                (cur_state.child(action), queue[i][1] * prob, copy.deepcopy(queue[i][2])))
        elif cur_state.current_player() != player:
          cur_player = cur_state.current_player()
          for action in cur_state.legal_actions():
            prob_sum = 0
            next_prob = copy.deepcopy(queue[i][2])
            if self._is_joint:
              for j in range(len(self._player_policies)):
                prob_sum += queue[i][2][j] * \
                    self._player_policies[j][cur_player].action_probabilities(cur_state)[
                    action]
                next_prob[j] *= self._player_policies[j][cur_player].action_probabilities(cur_state)[
                    action]
            else:
              for j in range(len(self._player_policies[cur_player])):
                prob_sum += queue[i][2][cur_player][j] * \
                    self._player_policies[cur_player][j].action_probabilities(cur_state)[
                    action]
                next_prob[cur_player][j] *= self._player_policies[cur_player][j].action_probabilities(cur_state)[
                    action]
            queue.append((cur_state.child(action), prob_sum, next_prob))
        elif cur_state.current_player() == player:
          cur_infostate_str = cur_state.information_state_string()
          if cur_infostate_str == state.information_state_string():
            final_distribution.append(queue[i])
          else:
            if cur_infostate_str in infostate_action_map:
              queue.append((cur_state.child(
                  infostate_action_map[cur_infostate_str]), queue[i][1], queue[i][2]))
        queue[i] = queue[-1]
        queue.pop()
        i += 1
    return final_distribution

  def sample_root_state_and_policies(self, state):
    """Sample a (state, opponent-mixture) from the current infostate"""
    final_distribution = self.get_root_state_distributions(state)
    state_probs = np.array([elem[1]
                           for elem in final_distribution]) + self._epsilon
    state_probs = state_probs/np.sum(state_probs)
    #print(final_distribution, state_probs)
    chosen_idx = self._random_state.choice(
        range(len(final_distribution)), p=state_probs)
    root_elem = final_distribution[chosen_idx]

    if self._is_joint:
      norm_prob = np.array(root_elem[2]) + self._epsilon
      norm_prob = norm_prob/np.sum(norm_prob)
      policy_idx = self._random_state.choice(
          len(self._player_policies), p=norm_prob)
      cur_policies = self._player_policies[policy_idx]
    else:
      norm_probs = [
          np.array(weights) + self._epsilon for weights in root_elem[2]]
      norm_probs = [weights/np.sum(weights) for weights in norm_probs]
      cur_policies = [self._random_state.choice(
          self._player_policies[n], p=norm_probs[n]) for n in range(len(norm_probs))]

    return root_elem[0].clone(), cur_policies

  def run_search(self, state):
    self.reset()
    assert state.get_game().get_type().dynamics == pyspiel.GameType.Dynamics.SEQUENTIAL
    assert state.get_game().get_type(
    ).information == pyspiel.GameType.Information.IMPERFECT_INFORMATION

    legal_actions = state.legal_actions()
    if len(legal_actions) == 1:
      return [(legal_actions[0], 1.0)]

    self._root_node = self.create_new_node(state)

    assert self._root_node

    root_infostate_key = self.get_state_key(state)

    for sim in range(self._max_simulations):
      sampled_root_state, sampled_player_policies = self.sample_root_state_and_policies(
          state)  # how to sample a pyspiel.state from another pyspiel.state?

      assert root_infostate_key == self.get_state_key(sampled_root_state)
      assert sampled_root_state
      self.run_simulation(sampled_root_state, sampled_player_policies)

    if self._allow_inconsistent_action_sets:  # when this happens?
      legal_actions = state.legal_actions()
      temp_node = self.filter_illegals(self._root_node, legal_actions)
      assert temp_node.total_visits > 0
      return self.get_final_policy(state, temp_node)
    else:
      return self.get_final_policy(state, self._root_node)

  def run_simulation(self, state, player_policies):
    if state.is_terminal():
      return state.returns()
    elif state.current_player() != self._player_id:
      if state.is_chance_node():
        action_list, prob_list = zip(*state.chance_outcomes())
      else:
        cur_player = state.current_player()
        action_list, prob_list = zip(
            *(player_policies[cur_player].action_probabilities(state)).items())
      chosen_action = self._random_state.choice(action_list, p=prob_list)
      state.apply_action(chosen_action)
      return self.run_simulation(state, player_policies)

    legal_actions = state.legal_actions()
    cur_player = state.current_player()
    node = self.lookup_or_create_node(state)

    assert node

    if node.total_visits == UNEXPANDED_VISIT_COUNT:
      node.total_visits = 0
      for action, prob in self._evaluator.prior(state):
        node.prior_map[action] = prob
      return self._evaluator.evaluate(state)
    else:
      chosen_action = self.check_expand(
          node, legal_actions)  # add one children at a time?
      if chosen_action != pyspiel.INVALID_ACTION:
        # check if all actions have been expanded, if not, select one?, if yes, ucb?
        self.expand_if_necessary(node, chosen_action)
      else:
        chosen_action = self.select_action_tree_policy(node, legal_actions)

      assert chosen_action != pyspiel.INVALID_ACTION

      node.total_visits += 1
      node.child_info[chosen_action].visits += 1
      state.apply_action(chosen_action)
      returns = self.run_simulation(state, player_policies)
      node.child_info[chosen_action].return_sum += returns[cur_player]
      return returns


class WrappedISMCTSTabularPolicy(Policy):
  """A policy class wrapping and caching ISMCTS search results."""

  def __init__(self, game, ismcts_agent):
    self._game = game
    self._ismcts_agent = ismcts_agent
    self._cache = {}

  def action_probabilities(self, state, player_id=None):
    if state.information_state_string() in self._cache:
      return self._cache[state.information_state_string()]
    action_and_probs = self._ismcts_agent.step_with_policy(state)[0]
    self._cache[state.information_state_string()] = {
        action: prob for action, prob in action_and_probs}
    return self._cache[state.information_state_string()]


class ISMCTSBROracle(optimization_oracle.AbstractOracle):
  """A BR oracle class that generates ISMCTS-based BR policies"""

  def __init__(self, **oracle_specific_execution_kwargs):
    self._oracle_specific_execution_kwargs = oracle_specific_execution_kwargs

  def __call__(self, game, training_parameters, strategy_sampler, using_joint_strategies, **oracle_specific_execution_kwargs):
    new_policies = []
    for player_parameters in training_parameters:
      player_policies = []
      for params in player_parameters:
        current_player = params['current_player']
        total_policies = params['total_policies']
        probabilities_of_playing_policies = params['probabilities_of_playing_policies']
        if using_joint_strategies:
          ismcts_agent = AgainstMixtureISMCTSBot(game=game, player_id=current_player,
                                                 player_policies=utils.marginal_to_joint(
                                                     utils.marginal_to_joint(total_policies)),
                                                 weights=probabilities_of_playing_policies.reshape(
                                                     -1),
                                                 is_joint=True, **self._oracle_specific_execution_kwargs)
        else:
          ismcts_agent = AgainstMixtureISMCTSBot(game=game, player_id=current_player,
                                                 player_policies=total_policies,
                                                 weights=probabilities_of_playing_policies,
                                                 is_joint=False, **self._oracle_specific_execution_kwargs)
        player_policies.append(WrappedISMCTSTabularPolicy(game, ismcts_agent))
      new_policies.append(player_policies)
    return new_policies


def _state_values(state, num_players, policy):
  """Value of a state for every player given a policy."""
  if state.is_terminal():
    return np.array(state.returns())
  else:
    p_action = (
        state.chance_outcomes() if state.is_chance_node() else
        policy.action_probabilities(state).items())
    return sum(prob * _state_values(state.child(action), num_players, policy)
               for action, prob in p_action)


def nash_conv(game, policy, use_cpp_br=False):
  root_state = game.new_initial_state()
  if use_cpp_br:
    best_response_values = np.array([
        pyspiel_best_response.CPPBestResponsePolicy(
            game, best_responder, policy).value(root_state)
        for best_responder in range(game.num_players())
    ])
  else:
    best_response_values = np.array([
        pyspiel_best_response.BestResponsePolicy(
            game, best_responder, policy).value(root_state)
        for best_responder in range(game.num_players())
    ])
  on_policy_values = _state_values(root_state, game.num_players(), policy)
  player_improvements = best_response_values - on_policy_values
  nash_conv_ = sum(player_improvements)
  return nash_conv_, best_response_values, on_policy_values, player_improvements


def main(_):
  rng = np.random.RandomState(FLAGS.seed)
  evaluator = RandomRolloutEvaluator(FLAGS.rollout_count, rng)
  psro_sims_per_entry = 1

  game = pyspiel.load_game("kuhn_poker")

  ismcts_br = ISMCTSBROracle(
      uct_c=FLAGS.uct_c, max_simulations=FLAGS.max_simulations, evaluator=evaluator)

  psro_solver = psro_v2.PSROSolver(game, ismcts_br, sims_per_entry=FLAGS.psro_sims_per_entry,
                                   training_strategy_selector='probabilistic',
                                   meta_strategy_method='nash',
                                   sample_from_marginals=True)


  for it in range(FLAGS.psro_iterations):
    start_time = time.time()
    meta_game = psro_solver.get_meta_game()
    meta_probabilities = psro_solver.get_meta_strategies()
    print("------------------iter {}--------------------------------".format(it))
    print("meta game matrix for row player:")
    print(meta_game[0])
    print("meta probabilities:")
    print(meta_probabilities)
    policies = psro_solver.get_policies()
    aggregator = policy_aggregator.PolicyAggregator(game)
    aggr_policies = aggregator.aggregate(
        range(game.num_players()), policies, meta_probabilities)
    exploitabilities, br_values, on_policy_values, expl_per_player = nash_conv(
        game, aggr_policies)
    print("Nash Conv:")
    print(exploitabilities)
    print("exploitabilities")
    print(expl_per_player)
    print("original values")
    print(on_policy_values)
    print("BR values")
    print(br_values)
    tt = (time.time()-start_time)/60
    print("spent {} mins".format(tt))
    psro_solver.iteration()


if __name__ == "__main__":
  app.run(main)
