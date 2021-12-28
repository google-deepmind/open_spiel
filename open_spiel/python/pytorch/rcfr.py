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

"""Regression counterfactual regret minimization (RCFR) [Waugh et al., 2015; Morrill, 2016].

In contrast to (tabular) counterfactual regret minimization (CFR)
[Zinkevich et al., 2007], RCFR replaces the table of regrets that generate the
current policy profile with a profile of regression models. The average
policy is still tracked exactly with a full game-size table. The exploitability
of the average policy in zero-sum games decreases as the model accuracy and
the number of iterations increase [Waugh et al., 2015; Morrill, 2016]. As long
as the regression model errors decrease across iterations, the average policy
converges toward a Nash equilibrium in zero-sum games.

# References

Dustin Morrill. Using Regret Estimation to Solve Games Compactly.
    M.Sc. thesis, Computing Science Department, University of Alberta,
    Apr 1, 2016, Edmonton Alberta, Canada.
Kevin Waugh, Dustin Morrill, J. Andrew Bagnell, and Michael Bowling.
    Solving Games with Functional Regret Estimation. At the Twenty-Ninth AAAI
    Conference on Artificial Intelligence, January 25-29, 2015, Austin Texas,
    USA. Pages 2138-2145.
Martin Zinkevich, Michael Johanson, Michael Bowling, and Carmelo Piccione.
    Regret Minimization in Games with Incomplete Information.
    At Advances in Neural Information Processing Systems 20 (NeurIPS). 2007.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def tensor_to_matrix(tensor):
  """Converts `tensor` to a matrix (a rank-2 tensor) or raises an exception.

  Args:
    tensor: The tensor to convert.

  Returns:
    A PyTorch matrix (rank-2 `torch.Tensor`).

  Raises:
    ValueError: If `tensor` cannot be trivially converted to a matrix, i.e.
      `tensor` has a rank > 2.
  """
  tensor = torch.Tensor(tensor)
  rank = tensor.ndim
  # rank = len(list(tensor.shape))
  if rank > 2:
    raise ValueError(
        ("Tensor {} cannot be converted into a matrix as it is rank "
         "{} > 2.").format(tensor, rank))
  elif rank < 2:
    num_columns = 1 if rank == 0 else tensor.shape[0]
    tensor = torch.reshape(tensor, [1, num_columns])
  return tensor


def with_one_hot_action_features(state_features, legal_actions,
                                 num_distinct_actions):
  """Constructs features for each sequence by extending state features.

  Sequences features are constructed by concatenating one-hot features
  indicating each action to the information state features and stacking them.

  Args:
    state_features: The features for the information state alone. Must be a
      `torch.Tensor` with a rank less than or equal to (if batched) 2.
    legal_actions: The list of legal actions in this state. Determines the
      number of rows in the returned feature matrix.
    num_distinct_actions: The number of globally distinct actions in the game.
      Determines the length of the action feature vector concatenated onto the
      state features.

  Returns:
    A `torch.Tensor` feature matrix with one row for each sequence and # state
    features plus `num_distinct_actions`-columns.

  Raises:
    ValueError: If `state_features` has a rank > 2.
  """
  state_features = tensor_to_matrix(state_features)
  with_action_features = []
  for action in legal_actions:
    action_features = F.one_hot(
        torch.tensor([action]), num_classes=num_distinct_actions)
    all_features = torch.cat([state_features, action_features], axis=1)
    with_action_features.append(all_features)
  return torch.cat(with_action_features, axis=0)


def sequence_features(state, num_distinct_actions):
  """The sequence features at `state`.

  Features are constructed by concatenating `state`'s normalized feature
  vector with one-hot vectors indicating each action (see
  `with_one_hot_action_features`).

  Args:
    state: An OpenSpiel `State`.
    num_distinct_actions: The number of globally distinct actions in `state`'s
      game.

  Returns:
    A `torch.Tensor` feature matrix with one row for each sequence.
  """
  return with_one_hot_action_features(state.information_state_tensor(),
                                      state.legal_actions(),
                                      num_distinct_actions)


def num_features(game):
  """Returns the number of features returned by `sequence_features`.

  Args:
    game: An OpenSpiel `Game`.
  """
  return game.information_state_tensor_size() + game.num_distinct_actions()


class RootStateWrapper(object):
  """Analyzes the subgame at a given root state.

  It enumerates features for each player sequence, creates a mapping between
  information states to sequence index offsets, and caches terminal values
  in a dictionary with history string keys.

  Properties:
    root: An OpenSpiel `State`.
    sequence_features: A `list` of sequence feature matrices, one for each
      player. This list uses depth-first, information state-major ordering, so
      sequences are grouped by information state. I.e. the first legal action
      in the first state has index 0, the second action in the same information
      state has index 1, the third action will have index 3, and so on.
      Sequences in the next information state descendant of the first action
      will begin indexing its sequences at the number of legal actions in the
      ancestor information state.
    num_player_sequences: The number of sequences for each player.
    info_state_to_sequence_idx: A `dict` mapping each information state string
      to the `sequence_features` index of the first sequence in the
      corresponding information state.
    terminal_values: A `dict` mapping history strings to terminal values for
      each player.
  """

  def __init__(self, state):
    self.root = state
    self._num_distinct_actions = len(state.legal_actions_mask(0))

    self.sequence_features = [[] for _ in range(state.num_players())]
    self.num_player_sequences = [0] * state.num_players()
    self.info_state_to_sequence_idx = {}
    self.terminal_values = {}
    self._walk_descendants(state)
    self.sequence_features = [
        torch.cat(rows, axis=0) for rows in self.sequence_features
    ]

  def _walk_descendants(self, state):
    """Records information about `state` and its descendants."""
    if state.is_terminal():
      self.terminal_values[state.history_str()] = np.array(state.returns())
      return

    elif state.is_chance_node():
      for action, _ in state.chance_outcomes():
        self._walk_descendants(state.child(action))
      return

    player = state.current_player()
    info_state = state.information_state_string(player)
    actions = state.legal_actions()

    if info_state not in self.info_state_to_sequence_idx:
      n = self.num_player_sequences[player]
      self.info_state_to_sequence_idx[info_state] = n
      self.sequence_features[player].append(
          sequence_features(state, self._num_distinct_actions))
      self.num_player_sequences[player] += len(actions)

    for action in actions:
      self._walk_descendants(state.child(action))

  def sequence_weights_to_policy(self, sequence_weights, state):
    """Returns a behavioral policy at `state` from sequence weights.

    Args:
      sequence_weights: An array of non-negative weights, one for each of
        `state.current_player()`'s sequences in `state`'s game.
      state: An OpenSpiel `State` that represents an information state in an
        alternating-move game.

    Returns:
      A `np.array<double>` probability distribution representing the policy in
      `state` encoded by `sequence_weights`. Weights corresponding to actions
      in `state` are normalized by their sum.

    Raises:
      ValueError: If there are too few sequence weights at `state`.
    """
    info_state = state.information_state_string()
    sequence_offset = self.info_state_to_sequence_idx[info_state]
    actions = state.legal_actions()

    sequence_idx_end = sequence_offset + len(actions)
    weights = sequence_weights[sequence_offset:sequence_idx_end]

    if len(weights) < len(actions):
      raise ValueError(
          ("Invalid policy: Policy {player} at sequence offset "
           "{sequence_offset} has only {policy_len} elements but there "
           "are {num_actions} legal actions.").format(
               player=state.current_player(),
               sequence_offset=sequence_offset,
               policy_len=len(weights),
               num_actions=len(actions)))
    return normalized_by_sum(weights)

  def sequence_weights_to_policy_fn(self, player_sequence_weights):
    """Returns a policy function based on sequence weights for each player.

    Args:
      player_sequence_weights: A list of weight arrays, one for each player.
        Each array should have a weight for each of that player's sequences in
        `state`'s game.

    Returns:
      A `State` -> `np.array<double>` function. The output of this function is
        a probability distribution that represents the policy at the given
        `State` encoded by `player_sequence_weights` according to
        `sequence_weights_to_policy`.
    """

    def policy_fn(state):
      player = state.current_player()
      return self.sequence_weights_to_policy(player_sequence_weights[player],
                                             state)

    return policy_fn

  def sequence_weights_to_tabular_profile(self, player_sequence_weights):
    """Returns the tabular profile-form of `player_sequence_weights`."""
    return sequence_weights_to_tabular_profile(
        self.root, self.sequence_weights_to_policy_fn(player_sequence_weights))

  def counterfactual_regrets_and_reach_weights(self, regret_player,
                                               reach_weight_player,
                                               *sequence_weights):
    """Returns counterfactual regrets and reach weights as a tuple.

    Args:
      regret_player: The player for whom counterfactual regrets are computed.
      reach_weight_player: The player for whom reach weights are computed.
      *sequence_weights: A list of non-negative sequence weights for each player
        determining the policy profile. Behavioral policies are generated by
        normalizing sequence weights corresponding to actions in each
        information state by their sum.

    Returns:
      The counterfactual regrets and reach weights as an `np.array`-`np.array`
        tuple.

    Raises:
      ValueError: If there are too few sequence weights at any information state
        for any player.
    """
    num_players = len(sequence_weights)
    regrets = np.zeros(self.num_player_sequences[regret_player])
    reach_weights = np.zeros(self.num_player_sequences[reach_weight_player])

    def _walk_descendants(state, reach_probabilities, chance_reach_probability):
      """Compute `state`'s counterfactual regrets and reach weights.

      Args:
        state: An OpenSpiel `State`.
        reach_probabilities: The probability that each player plays to reach
          `state`'s history.
        chance_reach_probability: The probability that all chance outcomes in
          `state`'s history occur.

      Returns:
        The counterfactual value of `state`'s history.
      Raises:
        ValueError if there are too few sequence weights at any information
        state for any player.
      """

      if state.is_terminal():
        player_reach = (
            np.prod(reach_probabilities[:regret_player]) *
            np.prod(reach_probabilities[regret_player + 1:]))

        counterfactual_reach_prob = player_reach * chance_reach_probability
        u = self.terminal_values[state.history_str()]
        return u[regret_player] * counterfactual_reach_prob

      elif state.is_chance_node():
        v = 0.0
        for action, action_prob in state.chance_outcomes():
          v += _walk_descendants(
              state.child(action), reach_probabilities,
              chance_reach_probability * action_prob)
        return v

      player = state.current_player()
      info_state = state.information_state_string(player)
      sequence_idx_offset = self.info_state_to_sequence_idx[info_state]
      actions = state.legal_actions(player)

      sequence_idx_end = sequence_idx_offset + len(actions)
      my_sequence_weights = sequence_weights[player][
          sequence_idx_offset:sequence_idx_end]

      if len(my_sequence_weights) < len(actions):
        raise ValueError(
            ("Invalid policy: Policy {player} at sequence offset "
             "{sequence_idx_offset} has only {policy_len} elements but there "
             "are {num_actions} legal actions.").format(
                 player=player,
                 sequence_idx_offset=sequence_idx_offset,
                 policy_len=len(my_sequence_weights),
                 num_actions=len(actions)))

      policy = normalized_by_sum(my_sequence_weights)
      action_values = np.zeros(len(actions))
      state_value = 0.0

      is_reach_weight_player_node = player == reach_weight_player
      is_regret_player_node = player == regret_player

      reach_prob = reach_probabilities[player]
      for action_idx, action in enumerate(actions):
        action_prob = policy[action_idx]
        next_reach_prob = reach_prob * action_prob

        if is_reach_weight_player_node:
          reach_weight_player_plays_down_this_line = next_reach_prob > 0
          if not reach_weight_player_plays_down_this_line:
            continue
          sequence_idx = sequence_idx_offset + action_idx
          reach_weights[sequence_idx] += next_reach_prob

        reach_probabilities[player] = next_reach_prob

        action_value = _walk_descendants(
            state.child(action), reach_probabilities, chance_reach_probability)

        if is_regret_player_node:
          state_value = state_value + action_prob * action_value
        else:
          state_value = state_value + action_value
        action_values[action_idx] = action_value

      reach_probabilities[player] = reach_prob

      if is_regret_player_node:
        regrets[sequence_idx_offset:sequence_idx_end] += (
            action_values - state_value)
      return state_value

    # End of _walk_descendants

    _walk_descendants(self.root, np.ones(num_players), 1.0)
    return regrets, reach_weights


def normalized_by_sum(v, axis=0, mutate=False):
  """Divides each element of `v` along `axis` by the sum of `v` along `axis`.

  Assumes `v` is non-negative. Sets of `v` elements along `axis` that sum to
  zero are normalized to `1 / v.shape[axis]` (a uniform distribution).

  Args:
    v: Non-negative array of values.
    axis: An integer axis.
    mutate: Whether or not to store the result in `v`.

  Returns:
    The normalized array.
  """
  v = np.asarray(v)
  denominator = v.sum(axis=axis, keepdims=True)
  denominator_is_zero = denominator == 0

  # Every element of `denominator_is_zero` that is true corresponds to a
  # set of elements in `v` along `axis` that are all zero. By setting these
  # denominators to `v.shape[axis]` and adding 1 to each of the corresponding
  # elements in `v`, these elements are normalized to `1 / v.shape[axis]`
  # (a uniform distribution).
  denominator += v.shape[axis] * denominator_is_zero
  if mutate:
    v += denominator_is_zero
    v /= denominator
  else:
    v = (v + denominator_is_zero) / denominator
  return v


def relu(v):
  """Returns the element-wise maximum between `v` and 0."""
  return np.maximum(v, 0)


def _descendant_states(state, depth_limit, depth, include_terminals,
                       include_chance_states):
  """Recursive descendant state generator.

  Decision states are always yielded.

  Args:
    state: The current state.
    depth_limit: The descendant depth limit. Zero will ensure only
      `initial_state` is generated and negative numbers specify the absence of a
      limit.
    depth: The current descendant depth.
    include_terminals: Whether or not to include terminal states.
    include_chance_states: Whether or not to include chance states.

  Yields:
    `State`, a state that is `initial_state` or one of its descendants.
  """
  if state.is_terminal():
    if include_terminals:
      yield state
    return

  if depth > depth_limit >= 0:
    return

  if not state.is_chance_node() or include_chance_states:
    yield state

  for action in state.legal_actions():
    state_for_search = state.child(action)
    for substate in _descendant_states(state_for_search, depth_limit, depth + 1,
                                       include_terminals,
                                       include_chance_states):
      yield substate


def all_states(initial_state,
               depth_limit=-1,
               include_terminals=False,
               include_chance_states=False):
  """Generates states from `initial_state`.

  Generates the set of states that includes only the `initial_state` and its
  descendants that satisfy the inclusion criteria specified by the remaining
  parameters. Decision states are always included.

  Args:
    initial_state: The initial state from which to generate states.
    depth_limit: The descendant depth limit. Zero will ensure only
      `initial_state` is generated and negative numbers specify the absence of a
      limit. Defaults to no limit.
    include_terminals: Whether or not to include terminal states. Defaults to
      `False`.
    include_chance_states: Whether or not to include chance states. Defaults to
      `False`.

  Returns:
    A generator that yields the `initial_state` and its descendants that
    satisfy the inclusion criteria specified by the remaining parameters.
  """
  return _descendant_states(
      state=initial_state,
      depth_limit=depth_limit,
      depth=0,
      include_terminals=include_terminals,
      include_chance_states=include_chance_states)


def sequence_weights_to_tabular_profile(root, policy_fn):
  """Returns the `dict` of `list`s of action-prob pairs-form of `policy_fn`."""
  tabular_policy = {}
  players = list(range(root.num_players()))
  for state in all_states(root):
    for player in players:
      legal_actions = state.legal_actions(player)
      if len(legal_actions) < 1:
        continue
      info_state = state.information_state_string(player)
      if info_state in tabular_policy:
        continue
      my_policy = policy_fn(state)
      tabular_policy[info_state] = list(zip(legal_actions, my_policy))
  return tabular_policy


def feedforward_evaluate(layers,
                         x,
                         use_skip_connections=False,
                         hidden_are_factored=False,
                         hidden_activation=nn.ReLU):
  """Evaluates `layers` as a feedforward neural network on `x`.

  Args:
    layers: The neural network layers (`torch.Tensor` -> `torch.Tensor`
      callables).
    x: The array-like input to evaluate. Must be trivially convertible to a
      matrix (tensor rank <= 2).
    use_skip_connections: Whether or not to use skip connections between layers.
      If the layer input has too few features to be added to the layer output,
      then the end of input is padded with zeros. If it has too many features,
      then the input is truncated.
    hidden_are_factored: Whether or not hidden logical layers are factored into
      two separate linear transformations stored as adjacent elements of
      `layers`.
    hidden_activation: the activation function following the hidden layers.

  Returns:
    The `torch.Tensor` evaluation result.

  Raises:
    ValueError: If `x` has a rank greater than 2.
  """
  x = tensor_to_matrix(x)
  i = 0
  while i < len(layers) - 1:
    if isinstance(layers[i], hidden_activation):
      x = layers[i](x)
      i += 1
      continue
    y = layers[i](x)
    i += 1
    if hidden_are_factored:
      y = layers[i](y)
      i += 1
    if use_skip_connections:
      my_num_features = x.shape[1]
      padding = y.shape[1] - my_num_features
      if padding > 0:
        zeros = torch.zeros([x.shape[0], padding])
        x = torch.cat([x, zeros], axis=1)
      elif padding < 0:
        x = x[0:x.shape[0], 0:y.shape[1]]
      y = x + y
    x = y
  return layers[-1](x)


class DeepRcfrModel(nn.Module):
  """A flexible deep feedforward RCFR model class.

  Properties:
    layers: The `torch.keras.Layer` layers describing this  model.
  """

  def __init__(self,
               game,
               num_hidden_units,
               num_hidden_layers=1,
               num_hidden_factors=0,
               hidden_activation=nn.ReLU,
               use_skip_connections=False,
               regularizer=None):
    """Creates a new `DeepRcfrModel.

    Args:
      game: The OpenSpiel game being solved.
      num_hidden_units: The number of units in each hidden layer.
      num_hidden_layers: The number of hidden layers. Defaults to 1.
      num_hidden_factors: The number of hidden factors or the matrix rank of the
        layer. If greater than zero, hidden layers will be split into two
        separate linear transformations, the first with
        `num_hidden_factors`-columns and the second with
        `num_hidden_units`-columns. The result is that the logical hidden layer
        is a rank-`num_hidden_units` matrix instead of a rank-`num_hidden_units`
        matrix. When `num_hidden_units < num_hidden_units`, this is effectively
        implements weight sharing. Defaults to 0.
      hidden_activation: The activation function to apply over hidden layers.
        Defaults to `torch.nn.ReLU`.
      use_skip_connections: Whether or not to apply skip connections (layer
        output = layer(x) + x) on hidden layers. Zero padding or truncation is
        used to match the number of columns on layer inputs and outputs.
      regularizer: A regularizer to apply to each layer. Defaults to `None`.
    """
    super(DeepRcfrModel, self).__init__()
    self._use_skip_connections = use_skip_connections
    self._hidden_are_factored = num_hidden_factors > 0
    self._hidden_activation = hidden_activation
    input_rank = game.information_state_tensor_shape(
    )[0] + game.new_initial_state().num_distinct_actions()

    self.layers = []
    for _ in range(num_hidden_layers):
      if self._hidden_are_factored:
        self.layers.append(nn.Linear(input_rank, num_hidden_factors, bias=True))

      self.layers.append(
          nn.Linear(
              num_hidden_factors if self._hidden_are_factored else input_rank,
              num_hidden_units,
              bias=True))
      if hidden_activation:
        self.layers.append(hidden_activation())

    self.layers.append(nn.Linear(num_hidden_units, 1, bias=True))

    self.layers = nn.ModuleList(self.layers)
    # Construct variables for all layers by exercising the network.
    x = torch.zeros([1, num_features(game)])
    for layer in self.layers:
      x = layer(x)

  def __call__(self, x):
    """Evaluates this model on `x`."""
    return feedforward_evaluate(
        layers=self.layers,
        x=x,
        use_skip_connections=self._use_skip_connections,
        hidden_are_factored=self._hidden_are_factored,
        hidden_activation=self._hidden_activation)


class _RcfrSolver(object):
  """An abstract RCFR solver class.

  Requires that subclasses implement `evaluate_and_update_policy`.
  """

  def __init__(self, game, models, truncate_negative=False):
    """Creates a new `_RcfrSolver`.

    Args:
      game: An OpenSpiel `Game`.
      models: Current policy models (optimizable array-like -> `torch.Tensor`
        callables) for both players.
      truncate_negative: Whether or not to truncate negative (approximate)
        cumulative regrets to zero to implement RCFR+. Defaults to `False`.
    """
    self._game = game
    self._models = models
    self._truncate_negative = truncate_negative
    self._root_wrapper = RootStateWrapper(game.new_initial_state())

    self._cumulative_seq_probs = [
        np.zeros(n) for n in self._root_wrapper.num_player_sequences
    ]

  def _sequence_weights(self, player=None):
    """Returns regret-like weights for each sequence as an `np.array`.

    Negative weights are truncated to zero.

    Args:
      player: The player to compute weights for, or both if `player` is `None`.
        Defaults to `None`.
    """
    if player is None:
      return [
          self._sequence_weights(player)
          for player in range(self._game.num_players())
      ]
    else:
      tensor = F.relu(
          torch.squeeze(self._models[player](
              self._root_wrapper.sequence_features[player])))
      return tensor.detach().numpy()

  def evaluate_and_update_policy(self, train_fn):
    """Performs a single step of policy evaluation and policy improvement.

    Args:
      train_fn: A (model, `torch.data.Dataset`) function that trains the given
        regression model to accurately reproduce the x to y mapping given x-y
        data.

    Raises:
      NotImplementedError: If not overridden by child class.
    """
    raise NotImplementedError()

  def current_policy(self):
    """Returns the current policy profile.

    Returns:
      A `dict<info state, list<Action, probability>>` that maps info state
      strings to `Action`-probability pairs describing each player's policy.
    """
    return self._root_wrapper.sequence_weights_to_tabular_profile(
        self._sequence_weights())

  def average_policy(self):
    """Returns the average of all policies iterated.

    This average policy converges toward a Nash policy as the number of
    iterations increases as long as the regret prediction error decreases
    continually [Morrill, 2016].

    The policy is computed using the accumulated policy probabilities computed
    using `evaluate_and_update_policy`.

    Returns:
      A `dict<info state, list<Action, probability>>` that maps info state
      strings to (Action, probability) pairs describing each player's policy.
    """
    return self._root_wrapper.sequence_weights_to_tabular_profile(
        self._cumulative_seq_probs)

  def _previous_player(self, player):
    """The previous player in the turn ordering."""
    return player - 1 if player > 0 else self._game.num_players() - 1

  def _average_policy_update_player(self, regret_player):
    """The player for whom the average policy should be updated."""
    return self._previous_player(regret_player)


class RcfrSolver(_RcfrSolver):
  """RCFR with an effectively infinite regret data buffer.

  Exact or bootstrapped cumulative regrets are stored as if an infinitely
  large data buffer. The average strategy is updated and stored in a full
  game-size table. Reproduces the RCFR versions used in experiments by
  Waugh et al. [2015] and Morrill [2016] except that this class does not
  restrict the user to regression tree models.
  """

  def __init__(self, game, models, bootstrap=None, truncate_negative=False):
    self._bootstrap = bootstrap
    super(RcfrSolver, self).__init__(
        game, models, truncate_negative=truncate_negative)

    self._regret_targets = [
        np.zeros(n) for n in self._root_wrapper.num_player_sequences
    ]

  def evaluate_and_update_policy(self, train_fn):
    """Performs a single step of policy evaluation and policy improvement.

    Args:
      train_fn: A (model, `torch.data.Dataset`) function that trains the given
        regression model to accurately reproduce the x to y mapping given x-y
        data.
    """
    sequence_weights = self._sequence_weights()
    player_seq_features = self._root_wrapper.sequence_features
    for regret_player in range(self._game.num_players()):
      seq_prob_player = self._average_policy_update_player(regret_player)

      regrets, seq_probs = (
          self._root_wrapper.counterfactual_regrets_and_reach_weights(
              regret_player, seq_prob_player, *sequence_weights))

      if self._bootstrap:
        self._regret_targets[regret_player][:] = sequence_weights[regret_player]
      if self._truncate_negative:
        regrets = np.maximum(-relu(self._regret_targets[regret_player]),
                             regrets)

      self._regret_targets[regret_player] += regrets
      self._cumulative_seq_probs[seq_prob_player] += seq_probs

      targets = torch.unsqueeze(
          torch.Tensor(self._regret_targets[regret_player]), axis=1)
      data = torch.utils.data.TensorDataset(player_seq_features[regret_player],
                                            targets)

      regret_player_model = self._models[regret_player]
      train_fn(regret_player_model, data)
      sequence_weights[regret_player] = self._sequence_weights(regret_player)


class ReservoirBuffer(object):
  """A generic reservoir buffer data structure.

  After every insertion, its contents represents a `size`-size uniform
  random sample from the stream of candidates that have been encountered.
  """

  def __init__(self, size):
    self.size = size
    self.num_elements = 0
    self._buffer = np.full([size], None, dtype=object)
    self._num_candidates = 0

  @property
  def buffer(self):
    return self._buffer[:self.num_elements]

  def insert(self, candidate):
    """Consider this `candidate` for inclusion in this sampling buffer."""
    self._num_candidates += 1
    if self.num_elements < self.size:
      self._buffer[self.num_elements] = candidate
      self.num_elements += 1
      return
    idx = np.random.choice(self._num_candidates)
    if idx < self.size:
      self._buffer[idx] = candidate

  def insert_all(self, candidates):
    """Consider all `candidates` for inclusion in this sampling buffer."""
    for candidate in candidates:
      self.insert(candidate)

  def num_available_spaces(self):
    """The number of freely available spaces in this buffer."""
    return self.size - self.num_elements


class ReservoirRcfrSolver(_RcfrSolver):
  """RCFR with a reservoir buffer for storing regret data.

  The average strategy is updated and stored in a full game-size table.
  """

  def __init__(self, game, models, buffer_size, truncate_negative=False):
    self._buffer_size = buffer_size
    super(ReservoirRcfrSolver, self).__init__(
        game, models, truncate_negative=truncate_negative)
    self._reservoirs = [
        ReservoirBuffer(self._buffer_size) for _ in range(game.num_players())
    ]

  def evaluate_and_update_policy(self, train_fn):
    """Performs a single step of policy evaluation and policy improvement.

    Args:
      train_fn: A (model, `torch.data.Dataset`) function that trains the given
        regression model to accurately reproduce the x to y mapping given x-y
        data.
    """
    sequence_weights = self._sequence_weights()
    player_seq_features = self._root_wrapper.sequence_features
    for regret_player in range(self._game.num_players()):
      seq_prob_player = self._average_policy_update_player(regret_player)

      regrets, seq_probs = (
          self._root_wrapper.counterfactual_regrets_and_reach_weights(
              regret_player, seq_prob_player, *sequence_weights))

      if self._truncate_negative:
        regrets = np.maximum(-relu(sequence_weights[regret_player]), regrets)

      next_data = list(
          zip(player_seq_features[regret_player],
              torch.unsqueeze(torch.Tensor(regrets), axis=1)))

      self._reservoirs[regret_player].insert_all(next_data)

      self._cumulative_seq_probs[seq_prob_player] += seq_probs

      my_buffer = list(
          torch.stack(a) for a in zip(*self._reservoirs[regret_player].buffer))

      data = torch.utils.data.TensorDataset(*my_buffer)

      regret_player_model = self._models[regret_player]
      train_fn(regret_player_model, data)
      sequence_weights[regret_player] = self._sequence_weights(regret_player)
