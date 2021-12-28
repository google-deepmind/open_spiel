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

"""Tests for the heuristic_payoff_table library."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

from open_spiel.python.egt import heuristic_payoff_table
from open_spiel.python.egt import utils
import pyspiel


class ModuleLevelTest(absltest.TestCase):

  def test__multinomial_coefficients(self):
    distributions = np.asarray([
        [2, 0],
        [1, 1],
        [1, 0],
    ])
    coefficients = heuristic_payoff_table._multinomial_coefficients(
        distributions)

    np.testing.assert_array_equal([1., 2., 1.], coefficients)

    distributions = np.asarray([
        [3, 0],
        [2, 1],
        [1, 2],
        [0, 3],
    ])
    coefficients = heuristic_payoff_table._multinomial_coefficients(
        distributions)
    np.testing.assert_array_equal([1., 3., 3., 1.], coefficients)

    distributions = np.asarray([
        [2, 0, 0],
        [0, 2, 0],
        [0, 0, 2],
        [1, 1, 0],
        [1, 0, 1],
        [0, 1, 1],
    ])
    coefficients = heuristic_payoff_table._multinomial_coefficients(
        distributions)
    np.testing.assert_array_equal([1., 1., 1., 2., 2., 2.], coefficients)


class PayoffTableTest(parameterized.TestCase):

  @parameterized.parameters(
      (5, 2),
      (2, 2),
  )
  def test_construction(self, num_players, num_strategies):
    logging.info("Testing payoff table construction.")
    table = heuristic_payoff_table.PayoffTable(num_players, num_strategies)
    num_rows = utils.n_choose_k(num_players + num_strategies - 1, num_players)
    distributions = np.array(
        list(utils.distribute(num_players, num_strategies)))
    payoffs = np.full([int(num_rows), num_strategies], np.nan)
    np.testing.assert_array_equal(
        np.concatenate([distributions, payoffs], axis=1), table())

  def test_from_heuristic_payoff_table(self):
    team_compositions = np.asarray([
        [2, 0],
        [1, 1],
        [0, 2],
    ])
    payoffs = np.asarray([
        [1, 2],
        [3, 4],
        [5, 6],
    ])
    hpt = np.hstack([team_compositions, payoffs])

    table = heuristic_payoff_table.from_heuristic_payoff_table(hpt)
    np.testing.assert_array_equal(team_compositions, table._distributions)
    np.testing.assert_array_equal(payoffs, table._payoffs)
    self.assertEqual(3, table.num_rows)

    distributions = np.asarray([
        [2, 0, 0],
        [0, 2, 0],
        [0, 0, 2],
        [1, 1, 0],
        [1, 0, 1],
        [0, 1, 1],
    ])
    shape = distributions.shape
    payoffs = np.reshape(np.arange(np.prod(shape)), shape)

    hpt = np.hstack([distributions, payoffs])
    table = heuristic_payoff_table.from_heuristic_payoff_table(hpt)
    np.testing.assert_array_equal(distributions, table._distributions)
    np.testing.assert_array_equal(payoffs, table._payoffs)
    self.assertEqual(distributions.shape[0], table.num_rows)

  @parameterized.parameters(("matrix_rps",))
  def test_from_matrix_game(self, game):
    game = pyspiel.load_matrix_game(game)
    payoff_tables = utils.game_payoffs_array(game)
    logging.info("Testing payoff table construction for matrix game.")
    table = heuristic_payoff_table.from_matrix_game(payoff_tables[0])
    print(table())

  @parameterized.parameters((np.array([0.7, 0.2, 0.1]),))
  def test_expected_payoff(self, strategy):
    logging.info("Testing expected payoff for matrix game.")
    game = pyspiel.load_matrix_game("matrix_rps")
    payoff_tables = utils.game_payoffs_array(game)
    table = heuristic_payoff_table.from_matrix_game(payoff_tables[0])
    expected_payoff = table.expected_payoff(strategy)
    print(expected_payoff)
    assert len(expected_payoff) == table._num_strategies

  def test_from_elo_scores(self):
    elo_scores = [800, 400, 400]
    elo_1 = 10**(800 / 400)
    elo_2 = 10**(400 / 400)  # This is also the associated value for player 3.
    expected = np.asarray([
        [2, 0, 0, 1 / 2, 0, 0],
        [0, 2, 0, 0, 1 / 2, 0],
        [0, 0, 2, 0, 0, 1 / 2],
        [1, 1, 0, elo_1 / (elo_1 + elo_2), elo_2 / (elo_1 + elo_2), 0],
        [1, 0, 1, elo_1 / (elo_1 + elo_2), 0, elo_2 / (elo_1 + elo_2)],
        [0, 1, 1, 0, 1 / 2, 1 / 2],
    ])

    htp = heuristic_payoff_table.from_elo_scores(elo_scores)

    np.testing.assert_array_almost_equal(
        utils.sort_rows_lexicographically(expected),
        utils.sort_rows_lexicographically(htp()),
        verbose=True)


if __name__ == "__main__":
  absltest.main()
