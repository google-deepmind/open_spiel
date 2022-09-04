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
"""An implementation of Information Set Monte Carlo Tree Search (IS-MCTS).

See Cowling, Powley, and Whitehouse 2011.
https://ieeexplore.ieee.org/document/6203567
"""

import copy
import enum
import numpy as np
import pyspiel

UNLIMITED_NUM_WORLD_SAMPLES = -1
UNEXPANDED_VISIT_COUNT = -1
TIE_TOLERANCE = 1e-5


class ISMCTSFinalPolicyType(enum.Enum):
  """A enumeration class for final ISMCTS policy type."""
  NORMALIZED_VISITED_COUNT = 1
  MAX_VISIT_COUNT = 2
  MAX_VALUE = 3


class ChildSelectionPolicy(enum.Enum):
  """A enumeration class for children selection in ISMCTS."""
  UCT = 1
  PUCT = 2


class ChildInfo(object):
  """Child node information for the search tree."""

  def __init__(self, visits, return_sum, prior):
    self.visits = visits
    self.return_sum = return_sum
    self.prior = prior

  def value(self):
    return self.return_sum / self.visits


class ISMCTSNode(object):
  """Node data structure for the search tree."""

  def __init__(self):
    self.child_info = {}
    self.total_visits = 0
    self.prior_map = {}


class ISMCTSBot(pyspiel.Bot):
  """Adapted from the C++ implementation."""

  def __init__(self,
               game,
               evaluator,
               uct_c,
               max_simulations,
               max_world_samples=UNLIMITED_NUM_WORLD_SAMPLES,
               random_state=None,
               final_policy_type=ISMCTSFinalPolicyType.MAX_VISIT_COUNT,
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
    assert state.get_game().get_type(
    ).dynamics == pyspiel.GameType.Dynamics.SEQUENTIAL
    assert state.get_game().get_type(
    ).information == pyspiel.GameType.Information.IMPERFECT_INFORMATION

    legal_actions = state.legal_actions()
    if len(legal_actions) == 1:
      return [(legal_actions[0], 1.0)]

    self._root_node = self.create_new_node(state)

    assert self._root_node

    root_infostate_key = self.get_state_key(state)

    for _ in range(self._max_simulations):
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
      policy = [(action, child.visits / total_visits)
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
      policy = [(action, 1. / count if child.visits == max_visits else 0.0)
                for action, child in node.child_info.items()]
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
      policy = [(action, 1. / count if child.value() == max_value else 0.0)
                for action, child in node.child_info.items()]

    policy_size = len(policy)
    legal_actions = state.legal_actions()
    if policy_size < len(legal_actions):  # do we really need this step?
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
          'Case not handled (badly set max_world_samples..?)')

  def resample_from_infostate(self, state):
    if self._resampler_cb:
      return self._resampler_cb(state, state.current_player())
    else:
      return state.resample_from_infostate(
          state.current_player(), pyspiel.UniformProbabilitySampler(0., 1.))

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

  def filter_illegals(self, node, legal_actions):
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
        action_value += (self._uct_c *
                         np.sqrt(np.log(node.total_visits)/child.visits))
      elif self._child_selection_policy == ChildSelectionPolicy.PUCT:
        action_value += (self._uct_c * child.prior *
                         np.sqrt(node.total_visits)/(1 + child.visits))
      else:
        raise pyspiel.SpielError('Child selection policy unrecognized.')
      if action_value > max_value + TIE_TOLERANCE:
        candidates = [action]
        max_value = action_value
      elif (action_value > max_value - TIE_TOLERANCE and
            action_value < max_value + TIE_TOLERANCE):
        candidates.append(action)
        max_value = action_value

    assert len(candidates) >= 1
    return candidates[self._random_state.randint(len(candidates))]

  def check_expand(self, node, legal_actions):
    if not self._allow_inconsistent_action_sets and len(
        node.child_info) == len(legal_actions):
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
        # check if all actions have been expanded, if not, select one?
        # if yes, ucb?
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
