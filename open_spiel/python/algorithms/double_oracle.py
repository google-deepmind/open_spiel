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
"""Double Oracle algorithm.

Solves two-player zero-sum games, for more information see:
McMahan et al. (2003). Planning in the presence of cost functions controlled by
  an adversary. In Proceedings of the 20th International Conference on Machine
  Learning (ICML-03) (pp. 536-543).
"""

import numpy as np

from open_spiel.python.algorithms import lp_solver
from open_spiel.python.egt import utils
import pyspiel


def lens(lists):
  """Returns the sizes of lists in a list."""
  return list(map(len, lists))


def solve_subgame(subgame_payoffs):
  """Solves the subgame using OpenSpiel's LP solver."""
  p0_sol, p1_sol, _, _ = lp_solver.solve_zero_sum_matrix_game(
      pyspiel.create_matrix_game(*subgame_payoffs))
  p0_sol, p1_sol = np.asarray(p0_sol), np.asarray(p1_sol)
  return [p0_sol / p0_sol.sum(), p1_sol / p1_sol.sum()]


class DoubleOracleSolver(object):
  """Double Oracle solver."""

  def __init__(self, game, enforce_symmetry=False):
    """Initializes the Double Oracle solver.

    Args:
      game: pyspiel.MatrixGame (zero-sum).
      enforce_symmetry: If True, enforces symmetry in the strategies appended by
        each player, by using the first player's best response for the second
        player as well; also asserts the game is symmetric and that players are
        seeded with identical initial_strategies, default: False.
    """
    assert isinstance(game, pyspiel.MatrixGame)
    assert game.get_type().utility == pyspiel.GameType.Utility.ZERO_SUM
    # convert matrix game to numpy.ndarray of shape [2,rows,columns]
    self.payoffs = utils.game_payoffs_array(game)
    self.subgame_strategies = [[], []]
    self.enforce_symmetry = enforce_symmetry
    if self.enforce_symmetry:
      assert utils.is_symmetric_matrix_game(self.payoffs), (
          "enforce_symmetry is True, but payoffs are asymmetric!")

  def subgame_payoffs(self):
    # Select payoffs from the full game according to the subgame strategies.
    assert all(lens(self.subgame_strategies)), "Need > 0 strategies per player."
    subgame_payoffs = np.copy(self.payoffs)
    for player, indices in enumerate(self.subgame_strategies):
      subgame_payoffs = np.take(subgame_payoffs, indices, axis=player + 1)
    return subgame_payoffs

  def oracle(self, subgame_solution):
    """Computes the best responses.

    Args:
      subgame_solution: List of subgame solution policies.

    Returns:
      best_response: For both players from the original set of pure strategies.
      best_response_utility: Corresponding utility for both players.
    """
    assert lens(subgame_solution) == lens(self.subgame_strategies), (
        f"{lens(subgame_solution)} != {lens(self.subgame_strategies)}")
    best_response = [None, None]
    best_response_utility = [None, None]
    n_best_responders = 1 if self.enforce_symmetry else 2
    for player in range(n_best_responders):
      opponent = 1 - player
      # collect relevant payoff entries
      payoffs = np.take(
          self.payoffs[player],
          self.subgame_strategies[opponent],
          axis=opponent)
      # transpose to move player to leading dimension
      payoffs = np.transpose(payoffs, [player, opponent])
      avg_payoffs = (payoffs @ subgame_solution[opponent]).squeeze()
      best_response[player] = np.argmax(avg_payoffs)
      best_response_utility[player] = avg_payoffs[best_response[player]]

    if self.enforce_symmetry:
      best_response[1] = best_response[0]
      best_response_utility[1] = best_response_utility[0]

    return best_response, best_response_utility

  def step(self):
    """Performs one iteration."""
    subgame_payoffs = self.subgame_payoffs()
    subgame_solution = solve_subgame(subgame_payoffs)
    best_response, best_response_utility = self.oracle(subgame_solution)

    # Add best responses to the subgame strategies (if not included yet).
    self.subgame_strategies = [
        sorted(set(strategies + [br]))
        for strategies, br in zip(self.subgame_strategies, best_response)
    ]
    return best_response, best_response_utility

  def solve_yield(self,
                  initial_strategies,
                  max_steps,
                  tolerance,
                  verbose,
                  yield_subgame=False):
    """Solves game using Double Oracle, yielding intermediate results.

    Args:
      initial_strategies: List of pure strategies for both players, optional.
      max_steps: Maximum number of iterations, default: 20.
      tolerance: Stop if the estimated value of the game is below the tolerance.
      verbose: If False, no warning is shown, default: True.
      yield_subgame: If True, yields the subgame on each iteration. Otherwise,
        yields the final results only, default: False.

    Yields:
      solution: Policies for both players.
      iteration: The number of iterations performed.
      value: Estimated value of the game.
    """
    if self.enforce_symmetry and initial_strategies:
      assert np.array_equal(initial_strategies[0], initial_strategies[1]), (
          f"Players must use same initial_strategies as symmetry is enforced."
          f"\ninitial_strategies[0]: {initial_strategies[0]}, "
          f"\ninitial_strategies[1]: {initial_strategies[1]}")

    self.subgame_strategies = (initial_strategies if initial_strategies
                               else [[0], [0]])
    iteration = 0
    while iteration < max_steps:
      if yield_subgame:
        yield None, iteration, None, self.subgame_payoffs()
      iteration += 1
      last_subgame_size = lens(self.subgame_strategies)
      _, best_response_utility = self.step()
      value = sum(best_response_utility)
      if abs(value) < tolerance:
        if verbose:
          print("Last iteration={}; value below tolerance {} < {}."
                .format(iteration, value, tolerance))
        break
      if lens(self.subgame_strategies) == last_subgame_size:
        if verbose:
          print(
              "Last iteration={}; no strategies added, increase tolerance={} or check subgame solver."
              .format(iteration, tolerance))
        break

    # Compute subgame solution and return solution in original strategy space.
    subgame_solution = solve_subgame(self.subgame_payoffs())
    solution = [np.zeros(k) for k in self.payoffs.shape[1:]]
    for p in range(2):
      solution[p][self.subgame_strategies[p]] = subgame_solution[p].squeeze()

    yield solution, iteration, value, self.subgame_payoffs()

  def solve(self,
            initial_strategies=None,
            max_steps=20,
            tolerance=5e-5,
            verbose=True):
    """Solves the game using Double Oracle, returning the final solution."""
    solution, iteration, value = None, None, None
    generator = self.solve_yield(initial_strategies, max_steps, tolerance,
                                 verbose, yield_subgame=False)
    for solution, iteration, value, _ in generator:
      pass
    return solution, iteration, value
