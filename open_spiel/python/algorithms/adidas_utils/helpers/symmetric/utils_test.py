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

"""Tests for open_spiel.python.algorithms.adidas_utils.helpers.symmetric.utils."""

from absl import logging  # pylint:disable=unused-import
from absl.testing import absltest
from absl.testing import parameterized

import numpy as np

from open_spiel.python.algorithms.adidas_utils.helpers.symmetric import utils


class UtilsTest(parameterized.TestCase):

  def test_symmetrize_tensor(self, trials=100, seed=1234):
    random = np.random.RandomState(seed)

    successes = []
    for _ in range(trials):
      pt = random.rand(3, 2, 2, 2)

      pt_sym_man = np.zeros_like(pt)
      for p in range(3):
        for i in range(2):
          for j in range(2):
            for k in range(2):
              if p == 0:
                # read: if player 0 plays i and its two opponents play j and k
                # this should return the same payoff as when
                # player 1 plays i and its two opponents play j and k
                # player 2 plays i and its two opponents play j and k
                # solution is to add up all these payoffs and replace with avg
                pt_sym_man[p, i, j, k] = (pt[0, i, j, k] + pt[0, i, k, j] +
                                          pt[1, j, i, k] + pt[1, k, i, j] +
                                          pt[2, j, k, i] + pt[2, k, j, i]) / 6.
              elif p == 1:
                # same rationale, but with player 1 playing j
                pt_sym_man[p, i, j, k] = (pt[0, j, i, k] + pt[0, j, k, i] +
                                          pt[1, i, j, k] + pt[1, k, j, i] +
                                          pt[2, i, k, j] + pt[2, k, i, j]) / 6.
              else:
                # same rationale, but with player 2 playing k
                pt_sym_man[p, i, j, k] = (pt[0, k, i, j] + pt[0, k, j, i] +
                                          pt[1, i, k, j] + pt[1, j, k, i] +
                                          pt[2, i, j, k] + pt[2, j, i, k]) / 6.
      pt_sym = utils.sym(pt)

      successes += [np.allclose(pt_sym, pt_sym_man)]

    perc = 100 * np.mean(successes)
    logging.info('symmetrizing success rate out of %d is %f', trials, perc)
    self.assertGreaterEqual(
        perc, 100., 'symmetrizing failed')


if __name__ == '__main__':
  absltest.main()
