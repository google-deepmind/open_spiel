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

# Lint as: python3
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
"""An AlphaZero style model with a policy and value head."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
from typing import Sequence

import numpy as np
import tensorflow.compat.v1 as tf


class TrainInput(collections.namedtuple(
    "TrainInput", "observation legals_mask policy value")):
  """Inputs for training the Model."""

  @staticmethod
  def stack(train_inputs):
    observation, legals_mask, policy, value = zip(*train_inputs)
    return TrainInput(
        np.array(observation, dtype=np.float32),
        np.array(legals_mask, dtype=np.bool),
        np.array(policy),
        np.expand_dims(value, 1))


class Losses(collections.namedtuple("Losses", "policy value l2")):
  """Losses from a training step."""

  @property
  def total(self):
    return self.policy + self.value + self.l2

  def __str__(self):
    return ("Losses(total: {:.3f}, policy: {:.3f}, value: {:.3f}, "
            "l2: {:.3f})").format(self.total, self.policy, self.value, self.l2)

  def __add__(self, other):
    return Losses(self.policy + other.policy,
                  self.value + other.value,
                  self.l2 + other.l2)

  def __truediv__(self, n):
    return Losses(self.policy / n, self.value / n, self.l2 / n)


class Model(object):
  """A wrapper around a keras model, and optimizer."""

  def __init__(self, keras_model, l2_regularization, learning_rate, device):
    """A wrapper around a keras model, and optimizer.

    Args:
      keras_model: a Keras Model object.
      l2_regularization: the amount of l2 regularization to use during training.
      learning_rate: a learning rate for the adam optimizer.
      device: The device used to run the keras_model during evaluation and
        training. Possible values are 'cpu', 'gpu', or a tf.device(...) object.
    """
    if device == "gpu":
      if not tf.test.is_gpu_available():
        raise ValueError("GPU support is unavailable.")
      self._device = tf.device("gpu:0")
    elif device == "cpu":
      self._device = tf.device("cpu:0")
    else:
      self._device = device
    self._keras_model = keras_model
    self._optimizer = tf.train.AdamOptimizer(learning_rate)
    self._l2_regularization = l2_regularization

  def inference(self, obs, mask):
    with self._device:
      value, _, policy_softmax = self._keras_model([
          np.array(obs, dtype=np.float32), np.array(mask, dtype=np.bool)])
    return value, policy_softmax

  def update(self, train_inputs: Sequence[TrainInput]):
    """Run an update step."""
    batch = TrainInput.stack(train_inputs)

    with self._device:
      with tf.GradientTape() as tape:
        values, policy_logits, _ = self._keras_model(
            [batch.observation, batch.legals_mask], training=True)
        loss_value = tf.losses.mean_squared_error(
            values, tf.stop_gradient(batch.value))

        # The policy_logits applied the mask already, and the targets only
        # contain valid policies, ie are also masked.
        loss_policy = tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=policy_logits, labels=tf.stop_gradient(batch.policy))
        loss_policy = tf.reduce_mean(loss_policy)

        loss_l2 = 0
        for weights in self._keras_model.trainable_variables:
          loss_l2 += self._l2_regularization * tf.nn.l2_loss(weights)
        loss = loss_policy + loss_value + loss_l2

      grads = tape.gradient(loss, self._keras_model.trainable_variables)

      self._optimizer.apply_gradients(
          zip(grads, self._keras_model.trainable_variables),
          global_step=tf.train.get_or_create_global_step())

    return Losses(policy=float(loss_policy), value=float(loss_value),
                  l2=float(loss_l2))


def cascade(x, fns):
  for fn in fns:
    x = fn(x)
  return x


def keras_resnet(input_shape,
                 num_actions,
                 num_residual_blocks=19,
                 num_filters=256,
                 value_head_hidden_size=256,
                 activation="relu",
                 data_format="channels_last"):
  """A ResNet implementation following AlphaGo Zero.

  This ResNet implementation copies as closely as possible the
  description found in the Methods section of the AlphaGo Zero Nature paper.
  It is mentioned in the AlphaZero Science paper supplementary material that
  "AlphaZero uses the same network architecture as AlphaGo Zero". Note that
  this implementation only supports flat policy distributions.

  Arguments:
    input_shape: A tuple of 3 integers specifying the non-batch dimensions of
      input tensor shape.
    num_actions: The determines the output size of the policy head.
    num_residual_blocks: The number of residual blocks. Can be 0.
    num_filters: the number of convolution filters to use in the residual blocks
    value_head_hidden_size: number of hidden units in the value head dense layer
    activation: the activation function to use in the net. Does not affect the
      final tanh activation in the value head.
    data_format: Can take values 'channels_first' or 'channels_last' (default).
      Which input dimension to interpret as the channel dimension. The input
      is (1, channel, width, height) with (1, width, height, channel)
  Returns:
    A keras Model with a single input and two outputs (value head, policy head).
    The policy is a flat distribution over actions.
  """
  def residual_layer(inputs, num_filters, kernel_size):
    return cascade(inputs, [
        tf.keras.layers.Conv2D(num_filters, kernel_size, padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation(activation),
        tf.keras.layers.Conv2D(num_filters, kernel_size, padding="same"),
        tf.keras.layers.BatchNormalization(axis=-1),
        lambda x: tf.keras.layers.add([x, inputs]),
        tf.keras.layers.Activation(activation),
    ])

  def resnet_body(inputs, num_filters, kernel_size):
    x = cascade(inputs, [
        tf.keras.layers.Conv2D(num_filters, kernel_size, padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation(activation),
    ])
    for _ in range(num_residual_blocks):
      x = residual_layer(x, num_filters, kernel_size)
    return x

  def resnet_value_head(inputs, hidden_size):
    return cascade(inputs, [
        tf.keras.layers.Conv2D(filters=1, kernel_size=1),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation(activation),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(hidden_size, activation),
        tf.keras.layers.Dense(1, activation="tanh", name="value"),
    ])

  def resnet_policy_head(inputs, num_classes):
    return cascade(inputs, [
        tf.keras.layers.Conv2D(filters=2, kernel_size=1),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation(activation),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(num_classes, name="policy"),
    ])

  input_size = int(np.prod(input_shape))
  inputs = tf.keras.Input(shape=input_size, dtype="float32", name="input")
  mask = tf.keras.Input(shape=num_actions, dtype="bool", name="mask")
  torso = tf.keras.layers.Reshape(input_shape)(inputs)
  # Note: Keras with TensorFlow 1.15 does not support the data_format arg on CPU
  # for convolutions. Hence why this transpose is needed.
  if data_format == "channels_first":
    torso = tf.keras.backend.permute_dimensions(torso, (0, 2, 3, 1))
  torso = resnet_body(torso, num_filters, 3)
  value_head = resnet_value_head(torso, value_head_hidden_size)
  policy_logits = resnet_policy_head(torso, num_actions)
  policy_logits = tf.where(mask, policy_logits,
                           -1e32 * tf.ones_like(policy_logits))
  policy_softmax = tf.keras.layers.Softmax()(policy_logits)
  return tf.keras.Model(inputs=[inputs, mask],
                        outputs=[value_head, policy_logits, policy_softmax])


def keras_mlp(input_shape,
              num_actions,
              num_layers=2,
              num_hidden=128,
              activation="relu"):
  """A simple MLP implementation with both a value and policy head.

  Arguments:
    input_shape: A tuple of 3 integers specifying the non-batch dimensions of
      input tensor shape.
    num_actions: The determines the output size of the policy head.
    num_layers: The number of dense layers before the policy and value heads.
    num_hidden: the number of hidden units in the dense layers.
    activation: the activation function to use in the net. Does not affect the
      final tanh activation in the value head.

  Returns:
    A keras Model with a single input and two outputs (value head, policy head).
    The policy is a flat distribution over actions.
  """
  input_size = int(np.prod(input_shape))
  inputs = tf.keras.Input(shape=input_size, dtype="float32", name="input")
  mask = tf.keras.Input(shape=num_actions, dtype="bool", name="mask")
  torso = inputs
  for _ in range(num_layers):
    torso = tf.keras.layers.Dense(num_hidden, activation=activation)(torso)
  policy_logits = tf.keras.layers.Dense(num_actions, name="policy")(torso)
  policy_logits = tf.where(mask, policy_logits,
                           -1e32 * tf.ones_like(policy_logits))
  policy_softmax = tf.keras.layers.Softmax()(policy_logits)
  value = tf.keras.layers.Dense(1, activation="tanh", name="value")(torso)
  return tf.keras.Model(inputs=[inputs, mask],
                        outputs=[value, policy_logits, policy_softmax])
