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

"""Tests for adidas."""

from absl.testing import absltest

import numpy as np

from open_spiel.python.algorithms import adidas

from open_spiel.python.algorithms.adidas_utils.games.big import ElFarol
from open_spiel.python.algorithms.adidas_utils.games.small import MatrixGame
from open_spiel.python.algorithms.adidas_utils.solvers.symmetric import qre_anneal as qre_anneal_sym


class AdidasTest(absltest.TestCase):

  def test_adidas_on_prisoners_dilemma(self):
    """Tests ADIDAS on a 2-player prisoner's dilemma game."""
    # pylint:disable=bad-whitespace
    pt_r = np.array([[-1, -3],
                     [0,  -2]])
    # pylint:enable=bad-whitespace
    # shift tensor to ensure positivity required if run adidas w/ Tsallis entrpy
    pt_r -= pt_r.min()
    pt_c = pt_r.T  # symmetric game
    pt = np.stack((pt_r, pt_c), axis=0).astype(float)
    pt /= pt.max()  # arbitrary design choice to upper bound entries to 1
    game = MatrixGame(pt, seed=0)
    # for games with more than 2 players, see adidas_utils/games/big.py
    solver = qre_anneal_sym.Solver(temperature=100,
                                   proj_grad=False, euclidean=True,
                                   lrs=(1e-4, 1e-4), exp_thresh=0.01,
                                   rnd_init=True, seed=0)
    # note we set rnd_init to True which initializes adidas' initial
    # approximation to nash to a random point on the simplex. if rnd_init is
    # False, adidas is initialized to uniform which is the Nash equilibrium
    # of the prisoner's dilemma, in which case adidas trivially solves this
    # game in 0 iterations.
    lle = adidas.ADIDAS(seed=0)
    lle.approximate_nash(game, solver, sym=True, num_iterations=1,
                         num_samples=1, num_eval_samples=int(1e5),
                         approx_eval=True, exact_eval=True,
                         avg_trajectory=False)
    self.assertLess(lle.results['exps_exact'][-1], 0.2)

  def test_adidas_on_elfarol(self):
    """Test ADIDAS on a 10-player, symmetric El Farol bar game."""
    game = ElFarol(n=10, c=0.7)
    solver = qre_anneal_sym.Solver(temperature=100,
                                   proj_grad=False, euclidean=False,
                                   lrs=(1e-4, 1e-2), exp_thresh=0.01,
                                   seed=0)
    lle = adidas.ADIDAS(seed=0)
    lle.approximate_nash(game, solver, sym=True, num_iterations=1,
                         num_samples=np.inf, num_eval_samples=int(1e5),
                         approx_eval=True, exact_eval=True,
                         avg_trajectory=False)
    self.assertLess(lle.results['exps_exact'][-1], 0.5)


if __name__ == '__main__':
  absltest.main()
