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
"""Neural Replicator Dynamics [Omidshafiei et al, 2019].

A policy gradient-like extension to replicator dynamics and the hedge algorithm
that incorporates function approximation.

# References

Shayegan Omidshafiei, Daniel Hennes, Dustin Morrill, Remi Munos,
  Julien Perolat, Marc Lanctot, Audrunas Gruslys, Jean-Baptiste Lespiau,
  Karl Tuyls. Neural Replicator Dynamics. https://arxiv.org/abs/1906.00190.
  2019.
"""

from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from open_spiel.python.pytorch import rcfr


def thresholded(
  logits: torch.Tensor, regrets: torch.Tensor, threshold: float = 2.0
) -> torch.Tensor:
  """Zeros out `regrets` where `logits` are too negative or too large."""
  can_decrease = torch.gt(logits, -threshold).float()
  can_increase = torch.lt(logits, threshold).float()
  regrets_negative = -F.relu(-regrets)
  regrets_positive = F.relu(regrets)
  return can_decrease * regrets_negative + can_increase * regrets_positive


class DeepNeurdModel(nn.Module):
  """A flexible deep feedforward NeuRD model class.

  Properties:
    layers: The `torch.nn.Linear` layers describing this model.
  """

  def __init__(
    self,
    game,
    num_hidden_units: int,
    num_hidden_layers: int = 1,
    num_hidden_factors: int = 0,
    hidden_activation: nn.Module = nn.ReLU(),
    use_skip_connections: bool = False,
    autoencode: bool = False,
  ) -> None:
    """Creates a new `DeepNeurdModel.

    Args:
      game: The OpenSpiel `Game` being solved.
      num_hidden_units: The number of units in each hidden layer.
      num_hidden_layers: The number of hidden layers. Defaults to 1.
      num_hidden_factors: The number of hidden factors or the matrix rank of the
        layer. If greater than zero, hidden layers will be split into two
        separate linear transformations. Defaults to 0
      hidden_activation: The activation function to apply over hidden layers.
        Defaults to `torch.nn.Relu`.
      use_skip_connections: Whether or not to apply skip connections (layer
        output = layer(x) + x) on hidden layers. Zero padding or truncation is
        used to match the number of columns on layer inputs and outputs.
      autoencode: Whether or not to output a reconstruction of the inputs upon
        being called. Defaults to `False`.
    """
    super().__init__()
    self._autoencode = autoencode

    input_size = rcfr.num_features(game)
    layers_ = [
      nn.Sequential(nn.Linear(input_size, num_hidden_units), hidden_activation)
    ]

    layers_.extend(
      [
        (
          rcfr.ResidualMLPBlock(
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
      ]
    )

    layers_.append(
      nn.Linear(
        num_hidden_units,
        1 + self._autoencode * rcfr.num_features(game),
      )
    )

    self.layers = nn.Sequential(*layers_)

  def forward(self, x: torch.Tensor, autoencode: bool) -> torch.Tensor:
    """Calls the model for an input tensor x.

    Args:
      x (torch.Tensor): Model input.
      training: Whether or not this is being called during training. If
        `training` and the constructor argument `autoencode` was `True`, then
        the output will contain the estimated regrets concatenated with a
        reconstruction of the input, otherwise only regrets will be returned.
        Defaults to `False`.

    Returns:
      The `torch.Tensor` resulting from evaluating this model on `x`. If
        `autoencode` and the constructor argument `autoencode` was `True`, then
        it will contain the estimated regrets concatenated with a
        reconstruction of the input, otherwise only regrets will be returned.
    """
    y = self.layers(x)
    return y if (autoencode and self._autoencode) else y[:, 0]


def train(
  model: DeepNeurdModel,
  data: torch.utils.data.Dataset,
  batch_size: int,
  step_size: float = 1.0,
  threshold: float = 2.0,
  autoencoder_loss: Callable = None,
) -> None:
  """Train NeuRD `model` on `data`."""
  data = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)
  optimiser = torch.optim.SGD(model.parameters(), lr=step_size)

  for x, regrets in data:
    optimiser.zero_grad()

    outputs = model(x, autoencode=autoencoder_loss is not None)
    logits = outputs[:, 0]
    logits = logits - torch.mean(logits)

    regrets = thresholded(logits, regrets, threshold=threshold).detach()
    utility = F.cross_entropy(logits, regrets)

    if autoencoder_loss is not None:
      utility = utility + autoencoder_loss(x, outputs[:, 1:])

    utility.backward()
    optimiser.step()


class CounterfactualNeurdSolver(object):
  """All-actions, strong NeuRD on counterfactual regrets.

  No regularization bonus is applied, so the current policy likely will not
  converge. The average policy profile is updated and stored in a full
  game-size table and may converge to an approximate Nash equilibrium in
  two-player, zero-sum games.
  """

  def __init__(self, game, models):
    """Creates a new `CounterfactualNeurdSolver`.

    Args:
      game: An OpenSpiel `Game`.
      models: Current policy models (optimizable array-like -> `torch.Tensor`
        callables) for both players.
    """
    self._game = game
    self._models = models
    self._root_wrapper = rcfr.RootStateWrapper(game.new_initial_state(), game)

    self._cumulative_seq_probs = [
      np.zeros(n) for n in self._root_wrapper.num_player_sequences
    ]

  def _sequence_weights(self, player=None):
    """Returns exponentiated weights for each sequence as an `np.array`."""
    if player is None:
      return [
        self._sequence_weights(player)
        for player in range(self._game.num_players())
      ]
    else:
      tensor = self._models[player](
        self._root_wrapper.sequence_features[player][None], autoencode=False
      )
      tensor = tensor - torch.max(tensor, dim=-1)[0]
      tensor = torch.exp(tensor)
      return tensor.detach().numpy().squeeze(0)

  def current_policy(self):
    """Returns the current policy profile.

    Returns:
      A `dict<info state, list<Action, probability>>` that maps info state
      strings to `Action`-probability pairs describing each player's policy.
    """
    return self._root_wrapper.sequence_weights_to_tabular_profile(
      self._sequence_weights()
    )

  def average_policy(self):
    """Returns the average of all policies iterated.

    The policy is computed using the accumulated policy probabilities computed
    using `evaluate_and_update_policy`.

    Returns:
      A `dict<info state, list<Action, probability>>` that maps info state
      strings to (Action, probability) pairs describing each player's policy.
    """
    return self._root_wrapper.sequence_weights_to_tabular_profile(
      self._cumulative_seq_probs
    )

  def _previous_player(self, player):
    """The previous player in the turn ordering."""
    return player - 1 if player > 0 else self._game.num_players() - 1

  def _average_policy_update_player(self, regret_player):
    """The player for whom the average policy should be updated."""
    return self._previous_player(regret_player)

  def evaluate_and_update_policy(self, train_fn: Callable):
    """Performs a single step of policy evaluation and policy improvement.

    Args:
      train_fn: A (model, `torch.utils.data.TensorDataset`) function that trains
        the given regression model to accurately reproduce the x to y mapping
        given x-y data.
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

      self._cumulative_seq_probs[seq_prob_player] += seq_probs
      targets = torch.Tensor(regrets)
      data = torch.utils.data.TensorDataset(
        player_seq_features[regret_player], targets
      )

      regret_player_model = self._models[regret_player]
      train_fn(regret_player_model, data)
      sequence_weights[regret_player] = self._sequence_weights(regret_player)
