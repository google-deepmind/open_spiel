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
"""Tests for open_spiel.python.algorithms.adidas_utils.nonsymmetric."""

import itertools

from absl import logging  # pylint:disable=unused-import
from absl.testing import absltest
from absl.testing import parameterized

import numpy as np

from scipy.spatial.distance import cosine

from open_spiel.python.algorithms.adidas_utils.helpers import misc

from open_spiel.python.algorithms.adidas_utils.solvers.nonsymmetric import ate
from open_spiel.python.algorithms.adidas_utils.solvers.nonsymmetric import ped
from open_spiel.python.algorithms.adidas_utils.solvers.nonsymmetric import qre


class ExploitabilityDescentTest(parameterized.TestCase):

  @staticmethod
  def numerical_gradient(fun, x, eps=np.sqrt(np.finfo(float).eps)):
    fun_0 = fun(x)
    num_grad = [np.zeros_like(xi) for xi in x]
    x_plus_dx = [np.copy(xi) for xi in x]
    for i in range(len(x)):
      for j in range(len(x[i])):
        x_plus_dx[i][j] = x[i][j] + eps
        num_grad[i][j] = (fun(x_plus_dx) - fun_0) / eps
        x_plus_dx[i][j] = x[i][j]
    return num_grad

  @staticmethod
  def prep_params(dist, pt, num_params):
    params = [dist]
    if num_params > 1:
      num_players = len(dist)
      nabla = [misc.pt_reduce(pt[i], dist, [i]) for i in range(num_players)]
      params += [nabla]  # policy_gradient
    return tuple(params)

  @parameterized.named_parameters(
      ("PED", (ped, False)),
      ("ATE_p=1", (ate, 1., False)),
      ("ATE_p=0.5", (ate, 0.5, False)),
      ("ATE_p=0.1", (ate, 0.1, False)),
      ("ATE_p=0", (ate, 0., False)),
      ("QRE_t=0.0", (qre, 0.0, False)),
      ("QRE_t=0.1", (qre, 0.1, False))
      )
  def test_exploitability_gradient_on_nonsymmetric_three_player_matrix_games(
      self, solver_tuple, trials=100, max_num_strats=3, atol=1e-1, rtol=1e-1,
      seed=1234):
    num_players = 3
    solver = solver_tuple[0].Solver(*solver_tuple[1:])

    random = np.random.RandomState(seed)

    successes = []
    for _ in range(trials):
      num_strats = random.randint(low=2, high=max_num_strats + 1,
                                  size=num_players)
      num_strats = tuple([int(ns) for ns in num_strats])
      payoff_tensor = random.rand(num_players, *num_strats)

      num_params = len(solver.init_vars(num_strats, num_players))
      dirichlet_alpha = [np.ones(num_strats_i) for num_strats_i in num_strats]
      dist = [random.dirichlet(alpha_i) for alpha_i in dirichlet_alpha]
      params = self.prep_params(dist, payoff_tensor, num_params)

      payoff_matrices = {}
      for pi, pj in itertools.combinations(range(num_players), 2):
        key = (pi, pj)
        pt_i = misc.pt_reduce(payoff_tensor[pi], dist, [pi, pj])
        pt_j = misc.pt_reduce(payoff_tensor[pj], dist, [pi, pj])
        payoff_matrices[key] = np.stack((pt_i, pt_j), axis=0)
      grad = solver.compute_gradients(params, payoff_matrices)[0][0]
      grad = np.concatenate(grad) / float(num_players)

      exp = lambda x: solver.exploitability(x, payoff_tensor)  # pylint: disable=cell-var-from-loop
      num_grad = np.concatenate(self.numerical_gradient(exp, dist))

      successes += [np.logical_and(np.allclose(grad, num_grad, rtol, atol),
                                   cosine(grad, num_grad) <= atol)]

    perc = 100 * np.mean(successes)
    logging.info("gradient accuracy success rate out of %d is %f", trials, perc)
    self.assertGreaterEqual(
        perc, 95., "exploitability gradient accuracy is too poor")


if __name__ == "__main__":
  absltest.main()
