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

"""Reinforcement learning loss functions.

All the loss functions implemented here compute the loss for the policy (actor).
The critic loss functions are typically regression loss are omitted for their
simplicity.

For the batch QPG, RM and RPG loss, please refer to the paper:
https://papers.nips.cc/paper/7602-actor-critic-policy-optimization-in-partially-observable-multiagent-environments.pdf

The BatchA2C loss uses code from the `TRFL` library:
https://github.com/deepmind/trfl/blob/master/trfl/discrete_policy_gradient_ops.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn.functional as F


def _assert_rank_and_shape_compatibility(tensors, rank):
  if not tensors:
    raise ValueError("List of tensors cannot be empty")

  tmp_shape = tensors[0].shape
  for tensor in tensors:
    if tensor.ndim != rank:
      raise ValueError("Shape %s must have rank %d" % (tensor.ndim, rank))
    if tensor.shape != tmp_shape:
      raise ValueError("Shapes %s and %s are not compatible" %
                       (tensor.shape, tmp_shape))


def compute_baseline(policy, action_values):
  # V = pi * Q, backprop through pi but not Q.
  return torch.sum(torch.mul(policy, action_values.detach()), dim=1)


def compute_regrets(policy_logits, action_values):
  """Compute regrets using pi and Q."""
  # Compute regret.
  policy = F.softmax(policy_logits, dim=1)
  # Avoid computing gradients for action_values.
  action_values = action_values.detach()

  baseline = compute_baseline(policy, action_values)

  regrets = torch.sum(
      F.relu(action_values - torch.unsqueeze(baseline, 1)), dim=1)

  return regrets


def compute_advantages(policy_logits, action_values, use_relu=False):
  """Compute advantages using pi and Q."""
  # Compute advantage.
  policy = F.softmax(policy_logits, dim=1)
  # Avoid computing gradients for action_values.
  action_values = action_values.detach()

  baseline = compute_baseline(policy, action_values)

  advantages = action_values - torch.unsqueeze(baseline, 1)
  if use_relu:
    advantages = F.relu(advantages)

  # Compute advantage weighted by policy.
  policy_advantages = -torch.mul(policy, advantages.detach())
  return torch.sum(policy_advantages, dim=1)


def compute_a2c_loss(policy_logits, actions, advantages):
  cross_entropy = F.cross_entropy(policy_logits, actions, reduction="none")
  advantages = advantages.detach()
  if advantages.ndim != cross_entropy.ndim:
    raise ValueError("Shapes %s and %s are not compatible" %
                     (advantages.ndim, cross_entropy.ndim))
  return torch.mul(cross_entropy, advantages)


def compute_entropy(policy_logits):
  return torch.sum(
      -F.softmax(policy_logits, dim=1) * F.log_softmax(policy_logits, dim=1),
      dim=-1)


class BatchQPGLoss(object):
  """Defines the batch QPG loss op."""

  def __init__(self, entropy_cost=None, name="batch_qpg_loss"):
    self._entropy_cost = entropy_cost
    self._name = name

  def loss(self, policy_logits, action_values):
    """Constructs a PyTorch Crierion that computes the QPG loss for batches.

    Args:
      policy_logits: `B x A` tensor corresponding to policy logits.
      action_values: `B x A` tensor corresponding to Q-values.

    Returns:
      loss: A 0-D `float` tensor corresponding the loss.
    """
    _assert_rank_and_shape_compatibility([policy_logits, action_values], 2)
    advantages = compute_advantages(policy_logits, action_values)
    _assert_rank_and_shape_compatibility([advantages], 1)
    total_adv = torch.mean(advantages, dim=0)

    total_loss = total_adv
    if self._entropy_cost:
      policy_entropy = torch.mean(compute_entropy(policy_logits))
      entropy_loss = torch.mul(float(self._entropy_cost), policy_entropy)
      total_loss = torch.add(total_loss, entropy_loss)

    return total_loss


class BatchRMLoss(object):
  """Defines the batch RM loss op."""

  def __init__(self, entropy_cost=None, name="batch_rm_loss"):
    self._entropy_cost = entropy_cost
    self._name = name

  def loss(self, policy_logits, action_values):
    """Constructs a PyTorch Crierion that computes the RM loss for batches.

    Args:
      policy_logits: `B x A` tensor corresponding to policy logits.
      action_values: `B x A` tensor corresponding to Q-values.

    Returns:
      loss: A 0-D `float` tensor corresponding the loss.
    """
    _assert_rank_and_shape_compatibility([policy_logits, action_values], 2)
    advantages = compute_advantages(policy_logits, action_values, use_relu=True)
    _assert_rank_and_shape_compatibility([advantages], 1)
    total_adv = torch.mean(advantages, dim=0)

    total_loss = total_adv
    if self._entropy_cost:
      policy_entropy = torch.mean(compute_entropy(policy_logits))
      entropy_loss = torch.mul(float(self._entropy_cost), policy_entropy)
      total_loss = torch.add(total_loss, entropy_loss)

    return total_loss


class BatchRPGLoss(object):
  """Defines the batch RPG loss op."""

  def __init__(self, entropy_cost=None, name="batch_rpg_loss"):
    self._entropy_cost = entropy_cost
    self._name = name

  def loss(self, policy_logits, action_values):
    """Constructs a PyTorch Crierion that computes the RPG loss for batches.

    Args:
      policy_logits: `B x A` tensor corresponding to policy logits.
      action_values: `B x A` tensor corresponding to Q-values.

    Returns:
      loss: A 0-D `float` tensor corresponding the loss.
    """
    _assert_rank_and_shape_compatibility([policy_logits, action_values], 2)
    regrets = compute_regrets(policy_logits, action_values)
    _assert_rank_and_shape_compatibility([regrets], 1)
    total_regret = torch.mean(regrets, dim=0)

    total_loss = total_regret
    if self._entropy_cost:
      policy_entropy = torch.mean(compute_entropy(policy_logits))
      entropy_loss = torch.mul(float(self._entropy_cost), policy_entropy)
      total_loss = torch.add(total_loss, entropy_loss)

    return total_loss


class BatchA2CLoss(object):
  """Defines the batch A2C loss op."""

  def __init__(self, entropy_cost=None, name="batch_a2c_loss"):
    self._entropy_cost = entropy_cost
    self._name = name

  def loss(self, policy_logits, baseline, actions, returns):
    """Constructs a PyTorch Crierion that computes the A2C loss for batches.

    Args:
      policy_logits: `B x A` tensor corresponding to policy logits.
      baseline: `B` tensor corresponding to baseline (V-values).
      actions: `B` tensor corresponding to actions taken.
      returns: `B` tensor corresponds to returns accumulated.

    Returns:
      loss: A 0-D `float` tensor corresponding the loss.
    """
    _assert_rank_and_shape_compatibility([policy_logits], 2)
    _assert_rank_and_shape_compatibility([baseline, actions, returns], 1)
    advantages = returns - baseline

    policy_loss = compute_a2c_loss(policy_logits, actions, advantages)
    total_loss = torch.mean(policy_loss, dim=0)
    if self._entropy_cost:
      policy_entropy = torch.mean(compute_entropy(policy_logits))
      entropy_loss = torch.mul(float(self._entropy_cost), policy_entropy)
      total_loss = torch.add(total_loss, entropy_loss)

    return total_loss
