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

"""Computes the distribution of a policy."""
import collections

from typing import Dict, List, Tuple
from open_spiel.python import policy as policy_module
from open_spiel.python.mfg import distribution as distribution_module
import pyspiel


def type_from_states(states):
  """Get node type of a list of states and assert they are the same."""
  types = [state.get_type() for state in states]
  assert len(set(types)) == 1, f"types: {types}"
  return types[0]


def state_to_str(state):
  # TODO(author15): Consider switching to
  # state.mean_field_population(). For now, this does not matter in
  # practice since games don't have different observation strings for
  # different player IDs.
  return state.observation_string(pyspiel.PlayerId.DEFAULT_PLAYER_ID)


def forward_actions(
    current_states: List[pyspiel.State], distribution: Dict[str, float],
    actions_and_probs_fn) -> Tuple[List[pyspiel.State], Dict[str, float]]:
  """Applies one action to each current state.

  Args:
    current_states: The states to apply actions on.
    distribution: Current distribution.
    actions_and_probs_fn: Function that maps one state to the corresponding list
      of (action, proba). For decision nodes, this should be the policy, and for
      chance nodes, this should be chance outcomes.

  Returns:
    A pair:
      - new_states: List of new states after applying one action on
        each input state.
      - new_distribution: Probabilities for each of these states.
  """
  new_states = []
  new_distribution = collections.defaultdict(float)
  for state in current_states:
    state_str = state_to_str(state)
    for action, prob in actions_and_probs_fn(state):
      new_state = state.child(action)
      new_state_str = state_to_str(new_state)
      if new_state_str not in new_distribution:
        new_states.append(new_state)
      new_distribution[new_state_str] += prob * distribution[state_str]
  return new_states, new_distribution


def one_forward_step(current_states: List[pyspiel.State],
                     distribution: Dict[str, float],
                     policy: policy_module.Policy):
  """Performs one step of the forward equation.

  Namely, this takes as input a list of current state, the current
  distribution, and performs one step of the forward equation, using
  actions coming from the policy or from the chance node
  probabilities, or propagating the distribution to the MFG nodes.

  Args:
    current_states: The states to perform the forward step on. All states are
      assumed to be of the same type.
    distribution: Current distribution.
    policy: Policy that will be used if states

  Returns:
    A pair:
      - new_states: List of new states after applying one step of the
        forward equation (either performing one action or doing one
        distribution update).
      - new_distribution: Probabilities for each of these states.
  """
  state_types = type_from_states(current_states)
  if state_types == pyspiel.StateType.CHANCE:
    return forward_actions(current_states, distribution,
                           lambda state: state.chance_outcomes())

  if state_types == pyspiel.StateType.MEAN_FIELD:
    new_states = []
    new_distribution = {}
    for state in current_states:
      dist = [
          # We need to default to 0, since the support requested by
          # the state in `state.distribution_support()` might have
          # states that we might not have reached yet. A probability
          # of 0. should be given for them.
          distribution.get(str_state, 0.)
          for str_state in state.distribution_support()
      ]
      new_state = state.clone()
      new_state.update_distribution(dist)
      new_state_str = state_to_str(new_state)
      if new_state_str not in new_distribution:
        new_states.append(new_state)
        new_distribution[new_state_str] = 0.0
      new_distribution[new_state_str] += distribution.get(
          state_to_str(state), 0)
    return new_states, new_distribution

  if state_types == pyspiel.StateType.DECISION:
    return forward_actions(
        current_states, distribution,
        lambda state: policy.action_probabilities(state).items())

  raise ValueError(
      f"Unpexpected state_stypes: {state_types}, states: {current_states}")


def check_distribution_sum(distribution: Dict[str, float], expected_sum: int):
  """Sanity check that the distribution sums to a given value."""
  sum_state_probabilities = sum(distribution.values())
  assert abs(sum_state_probabilities - expected_sum) < 1e-4, (
      "Sum of probabilities of all possible states should be the number of "
      f"population, it is {sum_state_probabilities}.")


class DistributionPolicy(distribution_module.Distribution):
  """Computes the distribution of a specified strategy."""

  def __init__(self, game: pyspiel.Game, policy: policy_module.Policy,
               root_state: pyspiel.State = None):
    """Initializes the distribution calculation.

    Args:
      game: The game to analyze.
      policy: The policy we compute the distribution of.
      root_state: The state of the game at which to start analysis. If `None`,
        the game root states are used.
    """
    super().__init__(game)
    self._policy = policy
    if root_state is None:
      self._root_states = game.new_initial_states()
    else:
      self._root_states = [root_state]
    self.distribution = None
    self.evaluate()

  def evaluate(self):
    """Evaluate the distribution over states of self._policy."""
    # List of all game states that have a non-zero probability at the current
    # timestep and player ID.
    current_states = self._root_states.copy()
    # Distribution at the current timestep. Maps state strings to
    # floats. For each group of states for a given population, these
    # floats represent a probability distribution.
    current_distribution = {state_to_str(state): 1
                            for state in current_states}
    # List of all distributions computed so far.
    all_distributions = [current_distribution]

    while type_from_states(current_states) != pyspiel.StateType.TERMINAL:
      new_states, new_distribution = one_forward_step(current_states,
                                                      current_distribution,
                                                      self._policy)
      check_distribution_sum(new_distribution, self.game.num_players())
      current_distribution = new_distribution
      current_states = new_states
      all_distributions.append(new_distribution)

    # Merge all per-timestep distributions into `self.distribution`.
    self.distribution = {}
    for dist in all_distributions:
      for state_str, prob in dist.items():
        if state_str in self.distribution:
          raise ValueError(
              f"{state_str} has already been seen in distribution.")
        self.distribution[state_str] = prob

  def value(self, state):
    return self.value_str(state_to_str(state))

  def value_str(self, state_str, default_value=None):
    """Return probability of the state encoded by state_str.

    Args:
      state_str: string description of the state. This should be created
        using observation_string.
      default_value: in case the state has not been seen by the distribution, to
        avoid raising a value error the default value is returned if it is not
        None.

    Returns:
      state_probability: probability to be in the state descripbed by
        state_str.

    Raises:
      ValueError: if the state has not been seen by the distribution and no
        default value has been passed to the method.
    """
    if default_value is None:
      try:
        return self.distribution[state_str]
      except KeyError as e:
        raise ValueError(
            f"Distribution not computed for state {state_str}") from e
    return self.distribution.get(state_str, default_value)
