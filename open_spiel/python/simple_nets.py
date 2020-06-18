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

"""Simple network classes for Tensorflow based on tf.Module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import tensorflow.compat.v1 as tf

# Temporarily disable TF2 behavior until code is updated.
tf.disable_v2_behavior()

# This code is based directly on the TF docs:
# https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/Module

class Dense(tf.Module):
  def __init__(self, in_size, out_size, activate_relu=True, name=None):
    super(Dense, self).__init__(name=name)
    self._activate_relu = activate_relu
    # Weight initialization inspired by Sonnet's Linear layer, 
    # which cites https://arxiv.org/abs/1502.03167v3
    stddev = 1 / math.sqrt(in_size)
    self._weights = tf.Variable(
        tf.random.truncated_normal([in_size, out_size],
                                   mean=0.0, stddev=stddev),
        name="weights")
    self._bias = tf.Variable(tf.zeros([out_size]), name="bias")

  def __call__(self, tensor):
    y = tf.matmul(tensor, self._weights) + self._bias
    return tf.nn.relu(y) if self._activate_relu else y


class MLP(tf.Module):
  def __init__(self, input_size, hidden_sizes, output_size,
               activate_final=False, name=None):
    super(MLP, self).__init__(name=name)    
    self._layers = []
    with self.name_scope:
      # Hidden layers
      for size in hidden_sizes:
        self._layers.append(Dense(in_size=input_size, out_size=size))
        input_size = size
      # Output layer
      self._layers.append(Dense(in_size=input_size, out_size=output_size,
                                activate_relu=activate_final))

  @tf.Module.with_name_scope
  def __call__(self, x):
    for layer in self._layers:
      x = layer(x)
    return x
