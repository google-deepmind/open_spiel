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

# Lint as: python3
"""Tests for open_spiel.python.algorithms.masked_softmax."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import tensorflow.compat.v1 as tf

from open_spiel.python.algorithms import masked_softmax

# Temporarily disable TF2 behavior until the code is updated.
tf.disable_v2_behavior()


exp = math.exp  # For shorter lines

_BATCH_INPUTS = np.asarray([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0],
    [7.0, 8.0, 9.0],
    [10.0, 11.0, 12.0],
    [13.0, 14.0, 15.0],
    [16.0, 17.0, 18.0],
])
_BATCH_MASK = np.asarray([
    [1.0, 1.0, 1.0],
    [1.0, 0.0, 1.0],
    [0.0, 1.0, 1.0],
    [0.0, 0.0, 1.0],
    [0.0, 1.0, 0.0],
    [1.0, 0.0, 0.0],
])
total_row_0 = exp(1) + exp(2) + exp(3)
total_row_1 = exp(4) + exp(6)
total_row_2 = exp(8) + exp(9)
# pyformat: disable
_BATCH_EXPECTED = np.asarray([
    [exp(1) / total_row_0, exp(2) / total_row_0, exp(3) / total_row_0],
    [exp(4) / total_row_1, 0, exp(6) / total_row_1],
    [0, exp(8) / total_row_2, exp(9) / total_row_2],
    [0, 0, 1],
    [0, 1, 0],
    [1, 0, 0],
])
# pyformat: enable

# The following provides a 2-batch set of time-sequence policies.
# [B, T, num_actions] = 2, 3, 3
_B_T_LOGITS = np.asarray([[
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0],
    [7.0, 8.0, 9.0],
], [
    [10.0, 11.0, 12.0],
    [13.0, 14.0, 15.0],
    [16.0, 17.0, 18.0],
]])
_B_T_MASK = np.asarray([[
    [1.0, 1.0, 1.0],
    [1.0, 0.0, 1.0],
    [0.0, 1.0, 1.0],
], [
    [0.0, 0.0, 1.0],
    [0.0, 1.0, 0.0],
    [1.0, 0.0, 0.0],
]])
_B_T_EXPECTED = np.asarray([[
    [exp(1) / total_row_0,
     exp(2) / total_row_0,
     exp(3) / total_row_0],
    [exp(4) / total_row_1, 0, exp(6) / total_row_1],
    [0, exp(8) / total_row_2, exp(9) / total_row_2],
], [
    [0, 0, 1],
    [0, 1, 0],
    [1, 0, 0],
]])
array = np.asarray
# We test over all the above examples.
_ALL_TESTS_INPUTS = [
    # Non-batch inputs
    (array([1., 1.]), array([1., 1.]), array([.5, .5])),
    (array([1., 1.]), array([0., 1.]), array([0., 1.])),
    (array([1., 1.]), array([1., 0.]), array([1., 0.])),
    (array([1., 1., 1]), array([1., 1., 0]), array([.5, .5, 0.])),
    # Batch-inputs
    (_BATCH_INPUTS, _BATCH_MASK, _BATCH_EXPECTED),
    # Batch-time inputs
    (_B_T_LOGITS, _B_T_MASK, _B_T_EXPECTED),
]


class MaskedSoftmaxTest(parameterized.TestCase, absltest.TestCase):

  @parameterized.parameters(_ALL_TESTS_INPUTS)
  def test_np_masked_softmax(self, logits, legal_actions, expected):
    np.testing.assert_array_almost_equal(
        expected, masked_softmax.np_masked_softmax(logits, legal_actions))

  @parameterized.parameters(_ALL_TESTS_INPUTS)
  def test_tf_masked_softmax(self, np_logits, np_legal_actions, expected):
    logits = tf.Variable(np_logits, tf.float32)
    mask = tf.Variable(np_legal_actions, tf.float32)

    policy = masked_softmax.tf_masked_softmax(logits, mask)

    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      np_policy = sess.run(policy)

    np.testing.assert_array_almost_equal(expected, np_policy)

  def test_masked_softmax_on_all_invalid_moves(self):
    # If all actions are illegal, the behavior is undefined (it can be nan
    # or can be 0. We add this test to document this behavior and know if we
    # change it.
    np_logits = np.asarray([[
        [1.0, 1.0, 1.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ]])
    logits = tf.Variable(np_logits, tf.float32)
    np_mask = np.asarray([[
        [1.0, 1.0, 1.0],
        [1.0, 0.0, 1.0],
        [0.0, 0.0, 0.0],
    ]])
    mask = tf.Variable(np_mask, tf.float32)

    expected = np.asarray([[
        [1 / 3, 1 / 3, 1 / 3],
        [1 / 2, 0.0, 1 / 2],
        [np.nan, np.nan, np.nan],
    ]])

    policy = masked_softmax.tf_masked_softmax(logits, mask)

    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      np_policy = sess.run(policy)
    np.testing.assert_array_almost_equal(expected, np_policy)

    # Numpy behaves similarly.
    np.testing.assert_array_almost_equal(
        expected, masked_softmax.np_masked_softmax(np_logits, np_mask))


if __name__ == '__main__':
  absltest.main()
