"""Dataset generation for meta-CFR algorithm."""

from typing import List, Tuple

import numpy as np

from open_spiel.python.examples.meta_cfr.sequential_games.typing import InfostateNode


class Dataset:
  """Dataset class to generate data for training meta-CFR model."""

  def __init__(self, train_dataset: List[Tuple[List[List[float]],
                                               InfostateNode]],
               batch_size: int):
    self._train_dataset = np.array(train_dataset)
    self._size = self._train_dataset.shape[0]
    self._batch_size = batch_size

  def get_batch(self):
    while True:
      np.random.shuffle(self._train_dataset)
      idx_sample = np.random.choice(self._size, self._batch_size)
      next_batch = self._train_dataset[idx_sample, :]
      yield next_batch
