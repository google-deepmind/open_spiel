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

"""Simple network classes for Tensorflow based on tf.Module."""

import math
import tensorflow.compat.v1 as tf

# Temporarily disable TF2 behavior until code is updated.
tf.disable_v2_behavior()

# This code is based directly on the TF docs:
# https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/Module


class Linear(tf.Module):
  """A simple linear module.

  Always includes biases and only supports ReLU activations.
  """

  def __init__(self, in_size, out_size, activate_relu=True, name=None):
    """Creates a linear layer.

    Args:
      in_size: (int) number of inputs
      out_size: (int) number of outputs
      activate_relu: (bool) whether to include a ReLU activation layer
      name: (string): the name to give to this layer
    """

    super(Linear, self).__init__(name=name)
    self._activate_relu = activate_relu
    # Weight initialization inspired by Sonnet's Linear layer,
    # which cites https://arxiv.org/abs/1502.03167v3
    stddev = 1.0 / math.sqrt(in_size)
    self._weights = tf.Variable(
        tf.random.truncated_normal([in_size, out_size], mean=0.0,
                                   stddev=stddev),
        name="weights")
    self._bias = tf.Variable(tf.zeros([out_size]), name="bias")

  def __call__(self, tensor):
    y = tf.matmul(tensor, self._weights) + self._bias
    return tf.nn.relu(y) if self._activate_relu else y


class Sequential(tf.Module):
  """A simple sequential module.

  Always includes biases and only supports ReLU activations.
  """

  def __init__(self, layers, name=None):
    """Creates a model from successively applying layers.

    Args:
      layers: Iterable[tf.Module] that can be applied.
      name: (string): the name to give to this layer
    """

    super(Sequential, self).__init__(name=name)
    self._layers = layers

  def __call__(self, tensor):
    for layer in self._layers:
      tensor = layer(tensor)
    return tensor


class MLP(tf.Module):
  """A simple dense network built from linear layers above."""

  def __init__(self,
               input_size,
               hidden_sizes,
               output_size,
               activate_final=False,
               name=None):
    """Create the MLP.

    Args:
      input_size: (int) number of inputs
      hidden_sizes: (list) sizes (number of units) of each hidden layer
      output_size: (int) number of outputs
      activate_final: (bool) should final layer should include a ReLU
      name: (string): the name to give to this network
    """

    super(MLP, self).__init__(name=name)
    self._layers = []
    with self.name_scope:
      # Hidden layers
      for size in hidden_sizes:
        self._layers.append(Linear(in_size=input_size, out_size=size))
        input_size = size
      # Output layer
      self._layers.append(
          Linear(
              in_size=input_size,
              out_size=output_size,
              activate_relu=activate_final))

  @tf.Module.with_name_scope
  def __call__(self, x):
    for layer in self._layers:
      x = layer(x)
    return x


class MLPTorso(tf.Module):
  """A specialized half-MLP module when constructing multiple heads.

  Note that every layer includes a ReLU non-linearity activation.
  """

  def __init__(self, input_size, hidden_sizes, name=None):
    super(MLPTorso, self).__init__(name=name)
    self._layers = []
    with self.name_scope:
      for size in hidden_sizes:
        self._layers.append(Linear(in_size=input_size, out_size=size))
        input_size = size

  @tf.Module.with_name_scope
  def __call__(self, x):
    for layer in self._layers:
      x = layer(x)
    return x
