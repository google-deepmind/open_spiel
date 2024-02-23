# Copyright 2023 DeepMind Technologies Limited
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

"""Methods to compute the core based on linear programming.

This file contains methods to compute the core using LPs referred to in
Yan & Procaccia '21: https://ojs.aaai.org/index.php/AAAI/article/view/16721
"""

import itertools

from typing import Any, Callable, List, Tuple

import cvxpy as cp
import numpy as np

from open_spiel.python.coalitional_games import coalitional_game


ConstraintsSamplingFuncType = Callable[
    [coalitional_game.CoalitionalGame, cp.Variable, cp.Variable, List[Any]],
    Any,
]


def add_all_constraints(
    game: coalitional_game.CoalitionalGame,
    x: cp.Variable,
    e: cp.Variable,
    constraints: List[Any]):
  # \sum x_i + e >= v(S), for all subsets S \subseteq N
  for c in itertools.product([0, 1], repeat=game.num_players()):
    coalition = np.asarray(c)
    val_coalition = game.coalition_value(coalition)
    constraints.append(x @ coalition + e >= val_coalition)


def make_uniform_sampling_constraints_function(
    num: int,
) -> ConstraintsSamplingFuncType:
  """Simple uniform constraint sampler (with replacement)."""

  def func(game: coalitional_game.CoalitionalGame,
           x: cp.Variable, e: cp.Variable, constraints: List[Any]):
    for _ in range(num):
      coalition = np.random.randint(2, size=game.num_players())
      val_coalition = game.coalition_value(coalition)
      constraints.append(x @ coalition + e >= val_coalition)
  return func


def solve_least_core_lp(
    game: coalitional_game.CoalitionalGame,
    constraint_function: ConstraintsSamplingFuncType,
) -> Tuple[np.ndarray, float]:
  """Solve the LP described in Yan & Procaccia, equation (1).

  This LP enumerates all (exponentially many!) possible coalitions, with one
  constraint per coalition. Will not scale to games with too many players.

  Args:
    game: the game the LP solves.
    constraint_function: function that adds the constraints

  Returns:
    solution: an array with num_players entries,
    epsilon: the lowest epsilon.
  """
  # TODO(author5): handle errors gracefully. E.g. if solving the LP fails.

  num_players = game.num_players()
  val_gc = game.coalition_value(np.ones(game.num_players()))

  # min e
  # indices 0 - n-1 correspond to x_i, index n corresponds to e
  x = cp.Variable(num_players, nonneg=True)
  e = cp.Variable()  # note: epsilon can be negative when the core is non-empty!

  objective = cp.Minimize(e)
  constraints = []

  # \sum_{i in N} x_i  = v(N)
  constraints.append(x @ np.ones(num_players) == val_gc)

  # Add the constraints
  constraint_function(game, x, e, constraints)

  prob = cp.Problem(objective, constraints)
  _ = prob.solve()
  # The optimal value for x is stored in `x.value`.

  return x.value, e.value
