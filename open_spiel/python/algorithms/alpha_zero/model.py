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

"""An AlphaZero style model with a policy and value head."""

import collections
import functools
import os
from typing import Sequence

import numpy as np
import tensorflow.compat.v1 as tf


def cascade(x, fns):
  for fn in fns:
    x = fn(x)
  return x

tfkl = tf.keras.layers
conv_2d = functools.partial(tfkl.Conv2D, padding="same")


def batch_norm(training, updates, name):
  """A batch norm layer.

  Args:
    training: A placeholder of whether this is done in training or not.
    updates: A list to be extended with this layer's updates.
    name: Name of the layer.

  Returns:
    A function to apply to the previous layer.
  """
  bn = tfkl.BatchNormalization(name=name)
  def batch_norm_layer(x):
    # This emits a warning that training is a placeholder instead of a concrete
    # bool, but seems to work anyway.
    applied = bn(x, training)
    updates.extend(bn.updates)
    return applied
  return batch_norm_layer


def residual_layer(inputs, num_filters, kernel_size, training, updates, name):
  return cascade(inputs, [
      conv_2d(num_filters, kernel_size, name=f"{name}_res_conv1"),
      batch_norm(training, updates, f"{name}_res_batch_norm1"),
      tfkl.Activation("relu"),
      conv_2d(num_filters, kernel_size, name=f"{name}_res_conv2"),
      batch_norm(training, updates, f"{name}_res_batch_norm2"),
      lambda x: tfkl.add([x, inputs]),
      tfkl.Activation("relu"),
  ])


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
  """An AlphaZero style model with a policy and value head.

  This supports three types of models: mlp, conv2d and resnet.

  All models have a shared torso stack with two output heads: policy and value.
  They have same meaning as in the AlphaGo Zero and AlphaZero papers. The resnet
  model copies the one in that paper when set with width 256 and depth 20. The
  conv2d model is the same as the resnet except uses a conv+batchnorm+relu
  instead of the res blocks. The mlp model uses dense layers instead of conv,
  and drops batch norm.

  Links to relevant articles/papers:
    https://deepmind.com/blog/article/alphago-zero-starting-scratch has an open
      access link to the AlphaGo Zero nature paper.
    https://deepmind.com/blog/article/alphazero-shedding-new-light-grand-games-chess-shogi-and-go
      has an open access link to the AlphaZero science paper.

  All are parameterized by their input (observation) shape and output size
  (number of actions), though the conv2d and resnet might only work with games
  that have spatial data (ie 3 non-batch dimensions, eg: connect four would
  work, but not poker).

  The depth is the number of blocks in the torso, where the definition of a
  block varies by model. For a resnet it's a resblock which is two conv2ds,
  batch norms and relus, and an addition. For conv2d it's a conv2d, a batch norm
  and a relu. For mlp it's a dense plus relu.

  The width is the number of filters for any conv2d and the number of hidden
  units for any dense layer.

  Note that this uses an explicit graph so that it can be used for inference
  and training from C++. It seems to also be 20%+ faster than using eager mode,
  at least for the unit test.
  """

  valid_model_types = ["mlp", "conv2d", "resnet"]

  def __init__(self, session, saver, path):
    """Init a model. Use build_model, from_checkpoint or from_graph instead."""
    self._session = session
    self._saver = saver
    self._path = path

    def get_var(name):
      return self._session.graph.get_tensor_by_name(name + ":0")

    self._input = get_var("input")
    self._legals_mask = get_var("legals_mask")
    self._training = get_var("training")
    self._value_out = get_var("value_out")
    self._policy_softmax = get_var("policy_softmax")
    self._policy_loss = get_var("policy_loss")
    self._value_loss = get_var("value_loss")
    self._l2_reg_loss = get_var("l2_reg_loss")
    self._policy_targets = get_var("policy_targets")
    self._value_targets = get_var("value_targets")
    self._train = self._session.graph.get_operation_by_name("train")

  @classmethod
  def build_model(cls, model_type, input_shape, output_size, nn_width, nn_depth,
                  weight_decay, learning_rate, path):
    """Build a model with the specified params."""
    if model_type not in cls.valid_model_types:
      raise ValueError(f"Invalid model type: {model_type}, "
                       f"expected one of: {cls.valid_model_types}")

    # The order of creating the graph, init, saver, and session is important!
    # https://stackoverflow.com/a/40788998
    g = tf.Graph()  # Allow multiple independent models and graphs.
    with g.as_default():
      cls._define_graph(model_type, input_shape, output_size, nn_width,
                        nn_depth, weight_decay, learning_rate)
      init = tf.variables_initializer(tf.global_variables(),
                                      name="init_all_vars_op")
      with tf.device("/cpu:0"):  # Saver only works on CPU.
        saver = tf.train.Saver(
            max_to_keep=10000, sharded=False, name="saver")
    session = tf.Session(graph=g)
    session.__enter__()
    session.run(init)
    return cls(session, saver, path)

  @classmethod
  def from_checkpoint(cls, checkpoint, path=None):
    """Load a model from a checkpoint."""
    model = cls.from_graph(checkpoint, path)
    model.load_checkpoint(checkpoint)
    return model

  @classmethod
  def from_graph(cls, metagraph, path=None):
    """Load only the model from a graph or checkpoint."""
    if not os.path.exists(metagraph):
      metagraph += ".meta"
    if not path:
      path = os.path.dirname(metagraph)
    g = tf.Graph()  # Allow multiple independent models and graphs.
    with g.as_default():
      saver = tf.train.import_meta_graph(metagraph)
    session = tf.Session(graph=g)
    session.__enter__()
    session.run("init_all_vars_op")
    return cls(session, saver, path)

  def __del__(self):
    if hasattr(self, "_session") and self._session:
      self._session.close()

  @staticmethod
  def _define_graph(model_type, input_shape, output_size,
                    nn_width, nn_depth, weight_decay, learning_rate):
    """Define the model graph."""
    # Inference inputs
    input_size = int(np.prod(input_shape))
    observations = tf.placeholder(tf.float32, [None, input_size], name="input")
    legals_mask = tf.placeholder(tf.bool, [None, output_size],
                                 name="legals_mask")
    training = tf.placeholder(tf.bool, name="training")

    bn_updates = []

    # Main torso of the network
    if model_type == "mlp":
      torso = observations  # Ignore the input shape, treat it as a flat array.
      for i in range(nn_depth):
        torso = cascade(torso, [
            tfkl.Dense(nn_width, name=f"torso_{i}_dense"),
            tfkl.Activation("relu"),
        ])
    elif model_type == "conv2d":
      torso = tfkl.Reshape(input_shape)(observations)
      for i in range(nn_depth):
        torso = cascade(torso, [
            conv_2d(nn_width, 3, name=f"torso_{i}_conv"),
            batch_norm(training, bn_updates, f"torso_{i}_batch_norm"),
            tfkl.Activation("relu"),
        ])
    elif model_type == "resnet":
      torso = cascade(observations, [
          tfkl.Reshape(input_shape),
          conv_2d(nn_width, 3, name="torso_in_conv"),
          batch_norm(training, bn_updates, "torso_in_batch_norm"),
          tfkl.Activation("relu"),
      ])
      for i in range(nn_depth):
        torso = residual_layer(torso, nn_width, 3, training, bn_updates,
                               f"torso_{i}")
    else:
      raise ValueError("Unknown model type.")

    # The policy head
    if model_type == "mlp":
      policy_head = cascade(torso, [
          tfkl.Dense(nn_width, name="policy_dense"),
          tfkl.Activation("relu"),
      ])
    else:
      policy_head = cascade(torso, [
          conv_2d(filters=2, kernel_size=1, name="policy_conv"),
          batch_norm(training, bn_updates, "policy_batch_norm"),
          tfkl.Activation("relu"),
          tfkl.Flatten(),
      ])
    policy_logits = tfkl.Dense(output_size, name="policy")(policy_head)
    policy_logits = tf.where(legals_mask, policy_logits,
                             -1e32 * tf.ones_like(policy_logits))
    unused_policy_softmax = tf.identity(tfkl.Softmax()(policy_logits),
                                        name="policy_softmax")
    policy_targets = tf.placeholder(
        shape=[None, output_size], dtype=tf.float32, name="policy_targets")
    policy_loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=policy_logits, labels=policy_targets),
        name="policy_loss")

    # The value head
    if model_type == "mlp":
      value_head = torso  # Nothing specific before the shared value head.
    else:
      value_head = cascade(torso, [
          conv_2d(filters=1, kernel_size=1, name="value_conv"),
          batch_norm(training, bn_updates, "value_batch_norm"),
          tfkl.Activation("relu"),
          tfkl.Flatten(),
      ])
    value_out = cascade(value_head, [
        tfkl.Dense(nn_width, name="value_dense"),
        tfkl.Activation("relu"),
        tfkl.Dense(1, name="value"),
        tfkl.Activation("tanh"),
    ])
    # Need the identity to name the single value output from the dense layer.
    value_out = tf.identity(value_out, name="value_out")
    value_targets = tf.placeholder(
        shape=[None, 1], dtype=tf.float32, name="value_targets")
    value_loss = tf.identity(tf.losses.mean_squared_error(
        value_out, value_targets), name="value_loss")

    l2_reg_loss = tf.add_n([
        weight_decay * tf.nn.l2_loss(var)
        for var in tf.trainable_variables()
        if "/bias:" not in var.name
    ], name="l2_reg_loss")

    total_loss = policy_loss + value_loss + l2_reg_loss
    optimizer = tf.train.AdamOptimizer(learning_rate)
    with tf.control_dependencies(bn_updates):
      unused_train = optimizer.minimize(total_loss, name="train")

  @property
  def num_trainable_variables(self):
    return sum(np.prod(v.shape) for v in tf.trainable_variables())

  def print_trainable_variables(self):
    for v in tf.trainable_variables():
      print("{}: {}".format(v.name, v.shape))

  def write_graph(self, filename):
    full_path = os.path.join(self._path, filename)
    tf.train.export_meta_graph(
        graph_def=self._session.graph_def, saver_def=self._saver.saver_def,
        filename=full_path, as_text=False)
    return full_path

  def inference(self, observation, legals_mask):
    return self._session.run(
        [self._value_out, self._policy_softmax],
        feed_dict={self._input: np.array(observation, dtype=np.float32),
                   self._legals_mask: np.array(legals_mask, dtype=np.bool),
                   self._training: False})

  def update(self, train_inputs: Sequence[TrainInput]):
    """Runs a training step."""
    batch = TrainInput.stack(train_inputs)

    # Run a training step and get the losses.
    _, policy_loss, value_loss, l2_reg_loss = self._session.run(
        [self._train, self._policy_loss, self._value_loss, self._l2_reg_loss],
        feed_dict={self._input: batch.observation,
                   self._legals_mask: batch.legals_mask,
                   self._policy_targets: batch.policy,
                   self._value_targets: batch.value,
                   self._training: True})

    return Losses(policy_loss, value_loss, l2_reg_loss)

  def save_checkpoint(self, step):
    return self._saver.save(
        self._session,
        os.path.join(self._path, "checkpoint"),
        global_step=step)

  def load_checkpoint(self, path):
    return self._saver.restore(self._session, path)
