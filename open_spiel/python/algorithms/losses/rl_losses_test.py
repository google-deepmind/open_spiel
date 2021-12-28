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

"""Tests for open_spiel.python.algorithms.losses.rl_losses."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np
import tensorflow.compat.v1 as tf

from open_spiel.python.algorithms.losses import rl_losses

# Temporarily disable v2 behavior until code is updated.
tf.disable_v2_behavior()


class RLLossesTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.named_parameters(('no_entropy_cost', 0.),
                                  ('with_entropy_cost', 1.))
  def test_batch_qpg_loss_with_entropy_cost(self, entropy_cost):
    batch_qpg_loss = rl_losses.BatchQPGLoss(entropy_cost=entropy_cost)
    q_values = tf.constant([[0., -1., 1.], [1., -1., 0]], dtype=tf.float32)
    policy_logits = tf.constant([[1., 1., 1.], [1., 1., 4.]], dtype=tf.float32)
    total_loss = batch_qpg_loss.loss(policy_logits, q_values)
    # Compute expected quantities.
    expected_policy_entropy = (1.0986 + 0.3665) / 2
    # baseline = \sum_a pi_a * Q_a = 0.
    # -\sum_a pi_a * (Q_a - baseline)
    expected_policy_loss = (0.0 + 0.0) / 2
    expected_total_loss = (
        expected_policy_loss + entropy_cost * expected_policy_entropy)
    with self.session() as sess:
      np.testing.assert_allclose(
          sess.run(total_loss), expected_total_loss, atol=1e-4)

  @parameterized.named_parameters(('no_entropy_cost', 0.),
                                  ('with_entropy_cost', 1.))
  def test_batch_rm_loss_with_entropy_cost(self, entropy_cost):
    batch_rpg_loss = rl_losses.BatchRMLoss(entropy_cost=entropy_cost)
    q_values = tf.constant([[0., -1., 1.], [1., -1., 0]], dtype=tf.float32)
    policy_logits = tf.constant([[1., 1., 1.], [1., 1., 4.]], dtype=tf.float32)
    total_loss = batch_rpg_loss.loss(policy_logits, q_values)
    # Compute expected quantities.
    expected_policy_entropy = (1.0986 + 0.3665) / 2
    # baseline = \sum_a pi_a * Q_a = 0.
    # -\sum_a pi_a * relu(Q_a - baseline)
    # negative sign as it's a loss term and loss needs to be minimized.
    expected_policy_loss = -(.3333 + .0452) / 2
    expected_total_loss = (
        expected_policy_loss + entropy_cost * expected_policy_entropy)
    with self.session() as sess:
      np.testing.assert_allclose(
          sess.run(total_loss), expected_total_loss, atol=1e-4)

  @parameterized.named_parameters(('no_entropy_cost', 0.),
                                  ('with_entropy_cost', 1.))
  def test_batch_rpg_loss_with_entropy_cost(self, entropy_cost):
    batch_rpg_loss = rl_losses.BatchRPGLoss(entropy_cost=entropy_cost)
    q_values = tf.constant([[0., -1., 1.], [1., -1., 0]], dtype=tf.float32)
    policy_logits = tf.constant([[1., 1., 1.], [1., 1., 4.]], dtype=tf.float32)
    total_loss = batch_rpg_loss.loss(policy_logits, q_values)
    # Compute expected quantities.
    expected_policy_entropy = (1.0986 + 0.3665) / 2
    # baseline = \sum_a pi_a * Q_a = 0.
    # \sum_a relu(Q_a - baseline)
    expected_policy_loss = (1.0 + 1.0) / 2
    expected_total_loss = (
        expected_policy_loss + entropy_cost * expected_policy_entropy)
    with self.session() as sess:
      np.testing.assert_allclose(
          sess.run(total_loss), expected_total_loss, atol=1e-4)

  @parameterized.named_parameters(('no_entropy_cost', 0.),
                                  ('with_entropy_cost', 1.))
  def test_batch_a2c_loss_with_entropy_cost(self, entropy_cost):
    batch_a2c_loss = rl_losses.BatchA2CLoss(entropy_cost=entropy_cost)
    policy_logits = tf.constant([[1., 1., 1.], [1., 1., 4.]], dtype=tf.float32)
    baseline = tf.constant([1. / 3, 0.5], dtype=tf.float32)
    actions = tf.constant([1, 2], dtype=tf.int32)
    returns = tf.constant([0., 1.], dtype=tf.float32)
    total_loss = batch_a2c_loss.loss(policy_logits, baseline, actions, returns)
    # Compute expected quantities.
    # advantages = returns - baseline = [-1./3, 0.5]
    # cross_entropy = [-log(e^1./3 * e^1), -log(e^4/(e^4+ e + e))]
    # = [1.0986, 0.09492]
    # policy_loss = cross_entropy * advantages = [-0.3662, 0.04746]
    expected_policy_entropy = (1.0986 + 0.3665) / 2
    expected_policy_loss = (-0.3662 + 0.04746) / 2
    expected_total_loss = (
        expected_policy_loss + entropy_cost * expected_policy_entropy)
    with self.session() as sess:
      np.testing.assert_allclose(
          sess.run(total_loss), expected_total_loss, atol=1e-4)


if __name__ == '__main__':
  tf.test.main()
