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
