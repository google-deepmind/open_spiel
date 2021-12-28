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

"""Solving matrix games with LP solver."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from open_spiel.python.algorithms import lp_solver
import pyspiel


def main(_):
  # lp_solver.solve_zero_sum_matrix_game(pyspiel.load_matrix_game("matrix_mp"))
  # lp_solver.solve_zero_sum_matrix_game(pyspiel.load_matrix_game("matrix_rps"))
  p0_sol, p1_sol, p0_sol_val, p1_sol_val = lp_solver.solve_zero_sum_matrix_game(
      pyspiel.create_matrix_game(
          [[0.0, -0.25, 0.5], [0.25, 0.0, -0.05], [-0.5, 0.05, 0.0]],
          [[0.0, 0.25, -0.5], [-0.25, 0.0, 0.05], [0.5, -0.05, 0.0]]))
  print("p0 val = {}, policy = {}".format(p0_sol_val, p0_sol))
  print("p1 val = {}, policy = {}".format(p1_sol_val, p1_sol))

  payoff_matrix = [[1., 1., 1.], [2., 0., 1.], [0., 2., 2.]]
  mixture = lp_solver.is_dominated(
      0, payoff_matrix, 0, lp_solver.DOMINANCE_WEAK, return_mixture=True)
  print("mixture strategy : {}".format(mixture))
  print("payoff vector    : {}".format(mixture.dot(payoff_matrix)))


if __name__ == "__main__":
  app.run(main)
