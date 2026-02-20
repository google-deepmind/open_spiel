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

from typing import Callable

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from open_spiel.python.pytorch.deep_cfr import ReservoirBuffer

# pylint: disable=g-bare-generic


def num_features(game) -> int:
  """Returns a number of features used for regression.

  Args:
    game: An OpenSpiel's `Game`.

  Returns:
    int: number columns in the feature matrix.
  """
  return game.information_state_tensor_size() + game.num_distinct_actions()


def sequence_features(
    state_features: list[str | int],
    legal_actions: list[int],
    num_distinct_actions: int,
) -> torch.Tensor:
  """Constructs features for each sequence by extending state features.

  Sequences features are constructed by concatenating one-hot features
  indicating each action to the information state features and stacking them.

  Args:
    state_features: The features of the information state.
    legal_actions: The list of legal actions available in the state. Determines
      the number of rows in the returned feature matrix.
    num_distinct_actions: The number of globally distinct actions in the game.
      Determines the length of the action feature vector concatenated onto the
      state features.

  Returns:
    A `torch.Tensor` feature matrix with one row for each sequence and # state
    features plus `num_distinct_actions`-columns.
  """
  state_features = torch.as_tensor(state_features)
  state_features = state_features[None].repeat(len(legal_actions), 1)
  action_features = F.one_hot(
      torch.LongTensor(legal_actions), num_distinct_actions
  )
  return torch.concatenate([state_features, action_features], dim=-1)


class RootStateWrapper:
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
    num_features: The number of features used for regression.
    info_state_to_sequence_idx: A `dict` mapping each information state string
      to the `sequence_features` index of the first sequence in the
      corresponding information state.
    terminal_values: A `dict` mapping history strings to terminal values for
      each player.
  """

  def __init__(self, state, game):
    self.root = state
    self._num_distinct_actions = game.num_distinct_actions()
    self.num_features = num_features(game)

    self.sequence_features = [[] for _ in range(state.num_players())]
    self.num_player_sequences = np.zeros(state.num_players(), dtype=np.int32)
    self.info_state_to_sequence_idx = {}
    self.terminal_values = {}

    def _traverse_tree(state) -> None:
      """Records information about `state` and its descendants."""
      if state.is_terminal():
        self.terminal_values[state.history_str()] = np.array(state.returns())
        return

      elif state.is_chance_node():
        for action, _ in state.chance_outcomes():
          _traverse_tree(state.child(action))
        return

      player = state.current_player()
      info_state = state.information_state_string(player)
      actions = state.legal_actions()

      if info_state not in self.info_state_to_sequence_idx:
        self.info_state_to_sequence_idx[info_state] = self.num_player_sequences[
            player
        ]
        self.sequence_features[player].append(
            sequence_features(
                state.information_state_tensor(),
                state.legal_actions(),
                self._num_distinct_actions,
            )
        )
        self.num_player_sequences[player] += len(actions)

      for action in actions:
        _traverse_tree(state.child(action))

    _traverse_tree(state)
    self.sequence_features = [
        torch.cat(rows, axis=0) for rows in self.sequence_features
    ]

  def sequence_weights_to_policy(
      self, sequence_weights: list[np.ndarray], state
  ):
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
          (
              "Invalid policy: Policy {player} at sequence offset "
              "{sequence_offset} has only {policy_len} elements but there "
              "are {num_actions} legal actions."
          ).format(
              player=state.current_player(),
              sequence_offset=sequence_offset,
              policy_len=len(weights),
              num_actions=len(actions),
          )
      )

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
      return self.sequence_weights_to_policy(
          player_sequence_weights[player], state
      )

    return policy_fn

  def sequence_weights_to_tabular_profile(self, player_sequence_weights):
    """Returns the tabular profile-form of `player_sequence_weights`."""
    return sequence_weights_to_tabular_profile(
        self.root, self.sequence_weights_to_policy_fn(player_sequence_weights)
    )

  def counterfactual_regrets_and_reach_weights(
      self, regret_player, reach_weight_player, *sequence_weights
  ):
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

    def _traverse_and_compute_regret(
        state, reach_probabilities, chance_reach_probability
    ):
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
        player_reach = np.prod(reach_probabilities[:regret_player]) * np.prod(
            reach_probabilities[regret_player + 1 :]
        )

        counterfactual_reach_prob = player_reach * chance_reach_probability
        u = self.terminal_values[state.history_str()]
        return u[regret_player] * counterfactual_reach_prob

      elif state.is_chance_node():
        v = 0.0
        for action, action_prob in state.chance_outcomes():
          v += _traverse_and_compute_regret(
              state.child(action),
              reach_probabilities,
              chance_reach_probability * action_prob,
          )
        return v

      player = state.current_player()
      info_state = state.information_state_string(player)
      sequence_idx_offset = self.info_state_to_sequence_idx[info_state]
      actions = state.legal_actions(player)

      sequence_idx_end = sequence_idx_offset + len(actions)
      my_sequence_weights = sequence_weights[player][
          sequence_idx_offset:sequence_idx_end
      ]

      if len(my_sequence_weights) < len(actions):
        raise ValueError(
            (
                "Invalid policy: Policy {player} at sequence offset"
                " {sequence_idx_offset} has only {policy_len} elements but"
                " there are {num_actions} legal actions."
            ).format(
                player=player,
                sequence_idx_offset=sequence_idx_offset,
                policy_len=len(my_sequence_weights),
                num_actions=len(actions),
            )
        )

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

        action_value = _traverse_and_compute_regret(
            state.child(action), reach_probabilities, chance_reach_probability
        )

        if is_regret_player_node:
          state_value = state_value + action_prob * action_value
        else:
          state_value = state_value + action_value
        action_values[action_idx] = action_value

      reach_probabilities[player] = reach_prob

      if is_regret_player_node:
        regrets[sequence_idx_offset:sequence_idx_end] += (
            action_values - state_value
        )
      return state_value

    _traverse_and_compute_regret(self.root, np.ones(num_players), 1.0)
    return regrets, reach_weights


def normalized_by_sum(v: list, axis: int = 0) -> np.ndarray:
  """Divides each element of `v` along `axis` by the sum of `v` along `axis`."""
  v = np.asarray(v)
  s = v.sum(axis=axis, keepdims=True)
  return np.where(s == 0, 1.0 / v.shape[axis], v / np.where(s == 0, 1.0, s))


def relu(v: np.ndarray) -> np.ndarray:
  """Returns the element-wise maximum between `v` and 0."""
  return np.maximum(v, 0)


def _descendant_states(
    state,
    depth_limit: int,
    depth: int,
    include_terminals: bool,
    include_chance_states: bool,
):
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
    for substate in _descendant_states(
        state_for_search,
        depth_limit,
        depth + 1,
        include_terminals,
        include_chance_states,
    ):
      yield substate


def all_states(
    initial_state,
    depth_limit: int = -1,
    include_terminals: bool = False,
    include_chance_states: bool = False,
):
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
      include_chance_states=include_chance_states,
  )


def sequence_weights_to_tabular_profile(root, policy_fn) -> dict:
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


class ResidualMLPBlock(nn.Module):
  """A residual MLP block."""

  def __init__(
      self,
      input_size: int,
      output_size: int,
      num_hidden_factors: int = 0,
      hidden_activation: nn.Module = nn.ReLU(),
  ) -> None:
    super().__init__()
    self._activation = hidden_activation
    self._gate_layer = (
        (nn.Linear(num_hidden_factors, output_size))
        if num_hidden_factors > 0
        else None
    )
    self._layer = nn.Sequential(
        nn.Linear(
            input_size,
            output_size if self._gate_layer is None else num_hidden_factors,
        ),
        self._activation if self._gate_layer else nn.Identity(),
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    residual = x.clone()
    x = self._layer(x)
    if self._gate_layer:
      x = self._gate_layer(x)
    x += residual
    return self._activation(x)


class DeepRcfrModel(nn.Module):
  """A flexible deep feedforward RCFR model class.

  Properties:
    layers: The `torch.nn.Sequential` layers describing this  model.
  """

  def __init__(
      self,
      game,
      num_hidden_units: int,
      num_hidden_layers: int = 1,
      num_hidden_factors: int = 0,
      hidden_activation: nn.Module = nn.ReLU(),
      use_skip_connections: bool = False,
  ) -> None:
    """Creates a new `DeepRcfrModel.

    Args:
      game: The OpenSpiel game being solved.
      num_hidden_units: The number of units in each hidden layer.
      num_hidden_layers: The number of hidden layers. Defaults to 1.
      num_hidden_factors: The number of hidden factors or the matrix rank of the
        layer. If greater than zero, hidden layers will be split into two
        separate linear transformations. Defaults to 0
      hidden_activation: The activation function to apply over hidden layers.
        Defaults to `torch.nn.ReLU`.
      use_skip_connections: Whether or not to apply skip connections (layer
        output = layer(x) + x) on hidden layers. Zero padding or truncation is
        used to match the number of columns on layer inputs and outputs.
    """
    super().__init__()

    input_size = num_features(game)
    layers_ = [
        nn.Sequential(
            nn.Linear(input_size, num_hidden_units), hidden_activation
        )
    ]

    layers_.extend([
        (
            ResidualMLPBlock(  # pylint: disable=g-long-ternary
                num_hidden_units,
                num_hidden_units,
                num_hidden_factors,
                hidden_activation,
            )
            if use_skip_connections
            else nn.Sequential(
                nn.Linear(num_hidden_units, num_hidden_units), hidden_activation
            )
        )
        for _ in range(num_hidden_layers)
    ])

    layers_.append(nn.Linear(num_hidden_units, 1))

    self.layers = nn.Sequential(*layers_)

  def __call__(self, x: torch.Tensor) -> torch.Tensor:
    """Evaluates this model on `x`."""
    return self.layers(x).squeeze(-1)


class _RcfrSolver(object):
  """An abstract RCFR solver class.

  Requires that subclasses implement `evaluate_and_update_policy`.
  """

  def __init__(
      self, game, models: list[DeepRcfrModel], truncate_negative: bool = False
  ) -> None:
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
    self._root_wrapper = RootStateWrapper(game.new_initial_state(), game)

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
          self._models[player](
              self._root_wrapper.sequence_features[player][None]
          )
      )
      return tensor.detach().numpy().squeeze(0)

  def evaluate_and_update_policy(self, train_fn: Callable):
    """Performs a single step of policy evaluation and policy improvement.

    Args:
      train_fn: A (model, `torch.data.Dataset`) function that trains the given
        regression model to accurately reproduce the x to y mapping given x-y
        data.

    Raises:
      NotImplementedError: If not overridden by child class.
    """
    raise NotImplementedError()

  def current_policy(self) -> dict:
    """Returns the current policy profile.

    Returns:
      A `dict<info state, list<Action, probability>>` that maps info state
      strings to `Action`-probability pairs describing each player's policy.
    """
    return self._root_wrapper.sequence_weights_to_tabular_profile(
        self._sequence_weights()
    )

  def average_policy(self) -> dict:
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
        self._cumulative_seq_probs
    )

  def _previous_player(self, player: int) -> int:
    """The previous player in the turn ordering."""
    return player - 1 if player > 0 else self._game.num_players() - 1

  def _average_policy_update_player(self, regret_player: int) -> int:
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

  def __init__(
      self,
      game,
      models: list[DeepRcfrModel],
      bootstrap=None,
      truncate_negative=False,
  ):
    self._bootstrap = bootstrap
    super(RcfrSolver, self).__init__(
        game, models, truncate_negative=truncate_negative
    )

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
              regret_player, seq_prob_player, *sequence_weights
          )
      )

      if self._bootstrap:
        self._regret_targets[regret_player] = sequence_weights[regret_player]
      if self._truncate_negative:
        regrets = np.maximum(
            -relu(self._regret_targets[regret_player]), regrets
        )

      self._regret_targets[regret_player] += regrets
      self._cumulative_seq_probs[seq_prob_player] += seq_probs

      targets = torch.FloatTensor(self._regret_targets[regret_player])
      data = torch.utils.data.TensorDataset(
          player_seq_features[regret_player], targets
      )

      regret_player_model = self._models[regret_player]
      train_fn(regret_player_model, data)
      sequence_weights[regret_player] = self._sequence_weights(regret_player)


class ReservoirRcfrSolver(_RcfrSolver):
  """RCFR with a reservoir buffer for storing regret data.

  The average strategy is updated and stored in a full game-size table.
  """

  def __init__(
      self,
      game,
      models: list[DeepRcfrModel],
      buffer_size: int,
      truncate_negative: bool = False,
  ):
    self._buffer_size = buffer_size
    super(ReservoirRcfrSolver, self).__init__(
        game, models, truncate_negative=truncate_negative
    )
    self._reservoirs = [None for _ in range(game.num_players())]

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
              regret_player, seq_prob_player, *sequence_weights
          )
      )

      if self._truncate_negative:
        regrets = np.maximum(-relu(sequence_weights[regret_player]), regrets)

      next_data = list(
          zip(player_seq_features[regret_player].numpy(), np.array(regrets))
      )

      for data in next_data:
        if self._reservoirs[regret_player] is None:
          self._reservoirs[regret_player] = ReservoirBuffer.init(
              self._buffer_size, data
          )
        assert self._reservoirs[regret_player] is not None
        self._reservoirs[regret_player].append(data)

      self._cumulative_seq_probs[seq_prob_player] += seq_probs

      X, y = [], []  # pylint: disable=invalid-name
      for _x, _y in zip(*self._reservoirs[regret_player].experience):
        X.append(torch.from_numpy(_x))
        y.append(torch.tensor(_y))

      data = torch.utils.data.TensorDataset(torch.stack(X), torch.stack(y))

      regret_player_model = self._models[regret_player]
      train_fn(regret_player_model, data)
      sequence_weights[regret_player] = self._sequence_weights(regret_player)
