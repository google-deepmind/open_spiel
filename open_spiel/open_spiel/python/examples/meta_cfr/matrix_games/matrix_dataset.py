# Copyright 2022 DeepMind Technologies Limited
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

"""Dataset for structured payoff matrices."""

from absl import flags
import numpy as np

FLAGS = flags.FLAGS


class Dataset:
  """Dataset class."""

  def __init__(self, base_matrix, num_training_batches, minval, maxval):
    self._base_matrix = base_matrix
    self._num_training_batches = num_training_batches
    self._minval, self._maxval = minval, maxval
    # to overfit
    self._new_matrix = np.copy(self._base_matrix)

  def get_training_batch(self):
    """Get training data."""
    while True:
      if not FLAGS.single_problem:
        random_vec = np.random.randint(
            low=self._minval, high=self._maxval, size=FLAGS.batch_size)
        self._new_matrix = np.copy(self._base_matrix)
        for i in range(FLAGS.batch_size):
          self._new_matrix[self._new_matrix > 0] += random_vec[i]
          self._new_matrix[self._new_matrix < 0] -= random_vec[i]
      yield self._new_matrix

  def get_eval_batch(self):
    """Get eval dataset."""

    if not FLAGS.single_problem:
      random_vec = np.random.randint(
          low=self._minval, high=self._maxval, size=FLAGS.batch_size)
      self._new_matrix = np.copy(self._base_matrix)
      for i in range(FLAGS.batch_size):
        self._new_matrix[self._new_matrix > 0] += random_vec[i]
        self._new_matrix[self._new_matrix < 0] -= random_vec[i]
    return self._new_matrix
