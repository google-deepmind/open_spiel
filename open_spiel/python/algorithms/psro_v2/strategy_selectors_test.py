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
"""Tests for open_spiel.python.algorithms.psro_v2.strategy_selectors."""

from absl.testing import absltest
import numpy as np
from open_spiel.python.algorithms.psro_v2 import strategy_selectors


class FakeSolver(object):

  def __init__(self, strategies, policies):
    self.strategies = strategies
    self.policies = policies

  def get_policies(self):
    return self.policies

  def get_meta_strategies(self):
    return self.strategies


def equal_to_transposition_lists(a, b):
  return [set(x) for x in a] == [set(x) for x in b]


EPSILON_MIN_POSITIVE_PROBA = 1e-8


def rectified_alias(solver, number_policies_to_select):
  """Returns every strategy with nonzero selection probability.

  Args:
    solver: A GenPSROSolver instance.
    number_policies_to_select: Number policies to select

  Returns:
    used_policies: A list, each element a list of the policies used per player.
  """
  del number_policies_to_select

  used_policies = []
  used_policy_indexes = []

  policies = solver.get_policies()
  num_players = len(policies)
  meta_strategy_probabilities = solver.get_meta_strategies()

  for k in range(num_players):
    current_policies = policies[k]
    current_probabilities = meta_strategy_probabilities[k]

    current_indexes = [
        i for i in range(len(current_policies))
        if current_probabilities[i] > EPSILON_MIN_POSITIVE_PROBA
    ]
    current_policies = [
        current_policies[i]
        for i in current_indexes
    ]

    used_policy_indexes.append(current_indexes)
    used_policies.append(current_policies)
  return used_policies, used_policy_indexes


def probabilistic_alias(solver, number_policies_to_select):
  """Returns [kwargs] policies randomly, proportionally with selection probas.

  Args:
    solver: A GenPSROSolver instance.
    number_policies_to_select: Number policies to select
  """
  policies = solver.get_policies()
  num_players = len(policies)
  meta_strategy_probabilities = solver.get_meta_strategies()

  print(policies, meta_strategy_probabilities)
  used_policies = []
  used_policy_indexes = []
  for k in range(num_players):
    current_policies = policies[k]
    current_selection_probabilities = meta_strategy_probabilities[k]
    effective_number = min(number_policies_to_select, len(current_policies))

    selected_indexes = list(
        np.random.choice(
            list(range(len(current_policies))),
            effective_number,
            replace=False,
            p=current_selection_probabilities))
    selected_policies = [current_policies[i] for i in selected_indexes]
    used_policies.append(selected_policies)
    used_policy_indexes.append(selected_indexes)
  return used_policies, used_policy_indexes


def top_k_probabilities_alias(solver, number_policies_to_select):
  """Returns [kwargs] policies with highest selection probabilities.

  Args:
    solver: A GenPSROSolver instance.
    number_policies_to_select: Number policies to select
  """
  policies = solver.get_policies()
  num_players = len(policies)
  meta_strategy_probabilities = solver.get_meta_strategies()

  used_policies = []
  used_policy_indexes = []
  for k in range(num_players):
    current_policies = policies[k]
    current_selection_probabilities = meta_strategy_probabilities[k]
    effective_number = min(number_policies_to_select, len(current_policies))

    # pylint: disable=g-complex-comprehension
    selected_indexes = [
        index for _, index in sorted(
            zip(current_selection_probabilities,
                list(range(len(current_policies)))),
            key=lambda pair: pair[0])
    ][:effective_number]

    selected_policies = [current_policies[i] for i in selected_indexes]
    used_policies.append(selected_policies)
    used_policy_indexes.append(selected_indexes)
  return used_policies, used_policy_indexes


class StrategySelectorsTest(absltest.TestCase):

  def test_vital(self):
    n_tests = 1000
    number_strategies = 50
    number_players = 3
    for i in range(n_tests):
      probabilities = np.random.uniform(size=(number_players,
                                              number_strategies))
      probabilities /= np.sum(probabilities, axis=1).reshape(-1, 1)
      probabilities = list(probabilities)

      policies = [list(range(number_strategies)) for _ in range(number_players)]

      solver = FakeSolver(probabilities, policies)

      # To see how rectified reacts to 0 probabilities.
      probabilities[0][0] = 0
      probabilities[-1][-1] = 0
      a, b = strategy_selectors.rectified(solver, 1)
      c, d = rectified_alias(solver, 1)

      self.assertEqual(a, c, "Rectified failed.")
      self.assertEqual(b, d, "Rectified failed.")

      a, b = strategy_selectors.top_k_probabilities(solver, 3)
      c, d = top_k_probabilities_alias(solver, 3)

      self.assertEqual(a, c, "Top k failed.")
      self.assertEqual(b, d, "Top k failed.")

      n_nonzero_policies = 2
      probabilities = [np.zeros(number_strategies) for _ in range(
          number_players)]

      for player in range(number_players):
        for _ in range(n_nonzero_policies):
          i = np.random.randint(0, high=number_strategies)
          while probabilities[player][i] > 1e-12:
            i = np.random.randint(0, high=number_strategies)
          probabilities[player][i] = 1.0 / n_nonzero_policies
        probabilities[player] /= np.sum(probabilities[player])

      solver = FakeSolver(probabilities, policies)
      a, b = strategy_selectors.probabilistic(solver, n_nonzero_policies)
      c, d = probabilistic_alias(solver, n_nonzero_policies)

      self.assertTrue(equal_to_transposition_lists(a, c),
                      "Probabilistic failed.")
      self.assertTrue(equal_to_transposition_lists(b, d),
                      "Probabilistic failed.")

if __name__ == "__main__":
  absltest.main()
