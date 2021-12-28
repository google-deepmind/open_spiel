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
"""Various masked_softmax implementations, both in numpy and tensorflow."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow.compat.v1 as tf

# Temporarily disable TF2 behavior until the code is updated.
tf.disable_v2_behavior()


def tf_masked_softmax(logits, legal_actions_mask):
  """Returns the softmax over the valid actions defined by `legal_actions_mask`.

  Args:
    logits: A tensor [..., num_actions] (e.g. [num_actions] or [B, num_actions])
      representing the logits to mask.
    legal_actions_mask: The legal action mask, same shape as logits. 1 means
      it's a legal action, 0 means it's illegal. If can be a tensorflow or numpy
      tensor.
  """
  # This will raise a warning as we are taking the log of 0, which sets the 0
  # values to -inf. However, this is fine, as we then apply tf.exp, which sets
  # tf.exp(-inf) to 0. e.g. if we have logits [5, 3, 1], with legal_mask
  # [0, 1, 1], then masked_logits == [-inf, 3, 1], so we subtract the max to
  # get [-inf, 0, -2], and apply tf.exp to get [0, 1, e^-2].
  legal_actions_mask = tf.cast(legal_actions_mask, dtype=logits.dtype)
  masked_logits = logits + tf.log(legal_actions_mask)
  max_logit = tf.reduce_max(masked_logits, axis=-1, keepdims=True)
  exp_logit = tf.exp(masked_logits - max_logit)
  return exp_logit / tf.reduce_sum(exp_logit, axis=-1, keepdims=True)


def np_masked_softmax(logits, legal_actions_mask):
  """Returns the softmax over the valid actions defined by `legal_actions_mask`.

  Args:
    logits: A tensor [..., num_actions] (e.g. [num_actions] or [B, num_actions])
      representing the logits to mask.
    legal_actions_mask: The legal action mask, same shape as logits. 1 means
      it's a legal action, 0 means it's illegal.
  """
  masked_logits = logits + np.log(legal_actions_mask)
  max_logit = np.amax(masked_logits, axis=-1, keepdims=True)
  exp_logit = np.exp(masked_logits - max_logit)
  return exp_logit / np.sum(exp_logit, axis=-1, keepdims=True)
