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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow.compat.v1 as tf

from open_spiel.python.algorithms import rcfr

# Temporarily disable TF2 behavior while the code is not updated.
tf.disable_v2_behavior()


def thresholded(logits, regrets, threshold=2.0):
  """Zeros out `regrets` where `logits` are too negative or too large."""
  can_decrease = tf.cast(tf.greater(logits, -threshold), tf.float32)
  can_increase = tf.cast(tf.less(logits, threshold), tf.float32)
  regrets_negative = tf.minimum(regrets, 0.0)
  regrets_positive = tf.maximum(regrets, 0.0)
  return can_decrease * regrets_negative + can_increase * regrets_positive


@tf.function
def train(model,
          data,
          batch_size,
          step_size=1.0,
          threshold=2.0,
          random_shuffle_size=None,
          autoencoder_loss=None):
  """Train NeuRD `model` on `data`."""
  if random_shuffle_size is None:
    random_shuffle_size = 10 * batch_size
  data = data.shuffle(random_shuffle_size)
  data = data.batch(batch_size)
  data = data.repeat(1)

  for x, regrets in data:
    with tf.GradientTape() as tape:
      output = model(x, training=True)
      logits = output[:, :1]
      logits = logits - tf.reduce_mean(logits, keepdims=True)

      regrets = tf.stop_gradient(
          thresholded(logits, regrets, threshold=threshold))
      utility = tf.reduce_mean(logits * regrets)

      if autoencoder_loss is not None:
        utility = utility - autoencoder_loss(x, output[:, 1:])

    grad = tape.gradient(utility, model.trainable_variables)

    for i, var in enumerate(model.trainable_variables):
      var.assign_add(step_size * grad[i])


class DeepNeurdModel(object):
  """A flexible deep feedforward NeuRD model class.

  Properties:
    layers: The `tf.keras.Layer` layers describing this  model.
    trainable_variables: The trainable `tf.Variable`s in this model's `layers`.
    losses: This model's layer specific losses (e.g. regularizers).
  """

  def __init__(self,
               game,
               num_hidden_units,
               num_hidden_layers=1,
               num_hidden_factors=0,
               hidden_activation=tf.nn.relu,
               use_skip_connections=False,
               regularizer=None,
               autoencode=False):
    """Creates a new `DeepNeurdModel.

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
        Defaults to `tf.nn.relu`.
      use_skip_connections: Whether or not to apply skip connections (layer
        output = layer(x) + x) on hidden layers. Zero padding or truncation is
        used to match the number of columns on layer inputs and outputs.
      regularizer: A regularizer to apply to each layer. Defaults to `None`.
      autoencode: Whether or not to output a reconstruction of the inputs upon
        being called. Defaults to `False`.
    """

    self._autoencode = autoencode
    self._use_skip_connections = use_skip_connections
    self._hidden_are_factored = num_hidden_factors > 0

    self.layers = []
    for _ in range(num_hidden_layers):
      if self._hidden_are_factored:
        self.layers.append(
            tf.keras.layers.Dense(
                num_hidden_factors,
                use_bias=True,
                kernel_regularizer=regularizer))

      self.layers.append(
          tf.keras.layers.Dense(
              num_hidden_units,
              use_bias=True,
              activation=hidden_activation,
              kernel_regularizer=regularizer))

    self.layers.append(
        tf.keras.layers.Dense(
            1 + self._autoencode * rcfr.num_features(game),
            use_bias=True,
            kernel_regularizer=regularizer))

    # Construct variables for all layers by exercising the network.
    x = tf.zeros([1, rcfr.num_features(game)])
    for layer in self.layers:
      x = layer(x)

    self.trainable_variables = sum(
        [layer.trainable_variables for layer in self.layers], [])
    self.losses = sum([layer.losses for layer in self.layers], [])

  def __call__(self, x, training=False):
    """Evaluates this model on x.

    Args:
      x: Model input.
      training: Whether or not this is being called during training. If
        `training` and the constructor argument `autoencode` was `True`, then
        the output will contain the estimated regrets concatenated with a
        reconstruction of the input, otherwise only regrets will be returned.
        Defaults to `False`.

    Returns:
      The `tf.Tensor` resulting from evaluating this model on `x`. If
        `training` and the constructor argument `autoencode` was `True`, then
        it will contain the estimated regrets concatenated with a
        reconstruction of the input, otherwise only regrets will be returned.
    """
    y = rcfr.feedforward_evaluate(
        layers=self.layers,
        x=x,
        use_skip_connections=self._use_skip_connections,
        hidden_are_factored=self._hidden_are_factored)
    return y if training else y[:, :1]


class CounterfactualNeurdSolver(object):
  """All-actions, strong NeuRD on counterfactual regrets.

  No regularization bonus is applied, so the current policy likely will not
  converge. The average policy profile is updated and stored in a full
  game-size table and may converge to an approximate Nash equilibrium in
  two-player, zero-sum games.
  """

  def __init__(self, game, models, session=None):
    """Creates a new `CounterfactualNeurdSolver`.

    Args:
      game: An OpenSpiel `Game`.
      models: Current policy models (optimizable array-like -> `tf.Tensor`
        callables) for both players.
      session: A TensorFlow `Session` to convert sequence weights from
        `tf.Tensor`s produced by `models` to `np.array`s. If `None`, it is
        assumed that eager mode is enabled. Defaults to `None`.
    """
    self._game = game
    self._models = models
    self._root_wrapper = rcfr.RootStateWrapper(game.new_initial_state())
    self._session = session

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
      tensor = tf.squeeze(self._models[player](
          self._root_wrapper.sequence_features[player]))
      tensor = tensor - tf.reduce_max(tensor, keepdims=True)
      tensor = tf.math.exp(tensor)
      return tensor.numpy() if self._session is None else self._session(tensor)

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

  def evaluate_and_update_policy(self, train_fn):
    """Performs a single step of policy evaluation and policy improvement.

    Args:
      train_fn: A (model, `tf.data.Dataset`) function that trains the given
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

      self._cumulative_seq_probs[seq_prob_player] += seq_probs

      targets = tf.expand_dims(regrets.astype('float32'), axis=1)
      data = tf.data.Dataset.from_tensor_slices(
          (player_seq_features[regret_player], targets))

      regret_player_model = self._models[regret_player]
      train_fn(regret_player_model, data)
      sequence_weights[regret_player] = self._sequence_weights(regret_player)
