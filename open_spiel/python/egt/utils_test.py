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

"""Tests for open_spiel.python.egt.utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
from absl.testing import absltest

from absl.testing import parameterized
import numpy as np

from open_spiel.python.egt import utils
import pyspiel


def _generate_prob_profiles(num_items, num_slots):
  """Another implementation of `distribution` for test purposes.

  This function is the original implementation from Karl. jblespiau@ find it
  useful to add it here as: 1) an additional test of our function 2) a check
  that the initial code is correct too.

  Args:
    num_items: The number of items to distribute.
    num_slots: The number of slots.

  Returns:
    A numpy array of shape [num_distributions, num_slots].
  """
  if num_slots == 1:
    return np.array([num_items])

  num_rows = utils.n_choose_k(num_items + num_slots - 1, num_items)
  distributions = np.empty([num_rows, num_slots])

  ind = 0
  for i in range(0, num_items + 1):
    n_tmp = num_items - i
    k_tmp = num_slots - 1
    distributions_tmp = _generate_prob_profiles(n_tmp, k_tmp)
    distributions[ind:ind +
                  np.shape(distributions_tmp)[0], :] = np.column_stack(
                      (np.array((np.ones(np.shape(distributions_tmp)[0]) * i)),
                       distributions_tmp))
    ind = ind + np.shape(distributions_tmp)[0]

  return distributions


class UtilsTest(parameterized.TestCase):

  @parameterized.parameters(
      (5, 3, False),
      (2, 2, True),
  )
  def test_distribution(self, num_items, num_slots, normalize):
    distribution = list(utils.distribute(num_items, num_slots, normalize))
    # Correct length.
    # See https://en.wikipedia.org/wiki/Stars_and_bars_%28combinatorics%29.
    self.assertLen(distribution,
                   utils.n_choose_k(num_items + num_slots - 1, num_items))
    # No duplicates.
    self.assertLen(distribution, len(set(distribution)))
    sum_distribution = num_items if not normalize else 1
    for d in distribution:
      self.assertTrue(sum_distribution, sum(d))
      self.assertTrue((np.asarray(d) >= 0).all())

  @parameterized.parameters(
      (5, 3),
      (2, 2),
      (3, 3),
      (10, 5),
  )
  def test_distribution_equivalent_implementation(self, num_items, num_slots):
    distribution = np.vstack(
        utils.distribute(num_items, num_slots, normalize=False))

    other_implementation = _generate_prob_profiles(num_items, num_slots)
    np.testing.assert_array_equal(
        utils.sort_rows_lexicographically(distribution),
        utils.sort_rows_lexicographically(other_implementation))

  def test_sort_rows_lexicographically(self):
    array = np.asarray([
        [1, 1, 0],
        [1, 2, 0],
        [3, 1, 0],
        [0, 0, 4],
    ])
    expected = np.asarray([
        [0, 0, 4],
        [1, 1, 0],
        [1, 2, 0],
        [3, 1, 0],
    ])

    np.testing.assert_equal(expected, utils.sort_rows_lexicographically(array))

  def test_id_profile_mapping(self):
    """Tests forward and backward mapping of pure strategy profiles to IDs."""

    num_strats_per_population = np.asarray([4, 4, 4, 9])
    num_pure_profiles = np.prod(num_strats_per_population)

    strat_ranges = [
        range(num_strats) for num_strats in num_strats_per_population
    ]

    id_list = []
    for strat_profile in itertools.product(strat_ranges[0], strat_ranges[1],
                                           strat_ranges[2], strat_ranges[3]):
      profile_id = utils.get_id_from_strat_profile(num_strats_per_population,
                                                   strat_profile)
      id_list.append(profile_id)

      # Tests backward mapping (ID-to-profile lookup)
      strat_profile_from_id = utils.get_strat_profile_from_id(
          num_strats_per_population, profile_id)
      np.testing.assert_array_equal(strat_profile, strat_profile_from_id)

    # Tests forward mapping (profile-to-ID lookup)
    np.testing.assert_array_equal(id_list, range(num_pure_profiles))

  def test_get_valid_next_profiles(self):
    """Tests next-profile generator."""

    num_strats_per_population = np.asarray([4, 5, 9, 7])
    cur_profile = np.asarray([1, 1, 2, 1])
    next_profiles = utils.get_valid_next_profiles(num_strats_per_population,
                                                  cur_profile)

    num_next_profiles = 0
    for _, _ in next_profiles:
      num_next_profiles += 1

    expected = (num_strats_per_population - 1).sum()
    np.testing.assert_equal(expected, num_next_profiles)

  def test_constant_sum_checker(self):
    """Tests if verification of constant-sum game is correct."""

    game = pyspiel.load_matrix_game("matrix_rps")
    payoff_tables = utils.game_payoffs_array(game)
    payoffs_are_hpt_format = utils.check_payoffs_are_hpt(payoff_tables)
    game_is_constant_sum, payoff_sum = utils.check_is_constant_sum(
        payoff_tables[0], payoffs_are_hpt_format)
    self.assertTrue(game_is_constant_sum)
    self.assertEqual(payoff_sum, 0.)

  def test_game_payoffs_array_rps(self):
    """Test `game_payoffs_array` for rock-paper-scissors."""
    game = pyspiel.load_matrix_game("matrix_rps")
    payoff_matrix = np.empty(shape=(2, 3, 3))
    payoff_row = np.array([[0., -1., 1.], [1., 0., -1.], [-1., 1., 0.]])
    payoff_matrix[0] = payoff_row
    payoff_matrix[1] = -1. * payoff_row
    np.testing.assert_allclose(utils.game_payoffs_array(game), payoff_matrix)

  def test_game_payoffs_array_pd(self):
    """Test `game_payoffs_array` for prisoners' dilemma."""
    game = pyspiel.load_matrix_game("matrix_pd")
    payoff_matrix = np.empty(shape=(2, 2, 2))
    payoff_row = np.array([[5., 0.], [10., 1.]])
    payoff_matrix[0] = payoff_row
    payoff_matrix[1] = payoff_row.T
    np.testing.assert_allclose(utils.game_payoffs_array(game), payoff_matrix)

  @parameterized.parameters(
      (100, 2, 0.),
      (100, 3, 0.),
      (100, 4, 0.),
      (100, 2, 0.05),
  )
  def test_sample_from_simplex(self, n, dim, vmin):
    """Test `sample_from_simplex`."""
    x = utils.sample_from_simplex(n, dim=dim, vmin=vmin)
    np.testing.assert_allclose(np.sum(x, axis=1), np.ones(n))
    self.assertTrue(np.alltrue(x <= 1. - vmin))
    self.assertTrue(np.alltrue(x >= vmin))


if __name__ == "__main__":
  absltest.main()
