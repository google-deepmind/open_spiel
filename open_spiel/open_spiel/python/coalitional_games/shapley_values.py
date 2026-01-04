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

"""Functions to compute Shapley values and their approximations."""

import itertools
import numpy as np
from open_spiel.python.coalitional_games import coalitional_game


def compute_shapley_values(
    game: coalitional_game.CoalitionalGame,
) -> np.ndarray:
  """Compute the Shapley values exactly.

  Uses Eq (2) of Mitchell et al. "Sampling Permutations for Shapley Value
  Estimation". https://people.math.sc.edu/cooper/shapley.pdf

  Args:
    game: the game to compute Shapley values for.

  Returns:
    shapley_values: a numpy array of Shapley values per player.
  """

  shapley_values_sum = np.zeros(game.num_players(), dtype=float)
  coalition = np.zeros(game.num_players(), dtype=int)
  empty_coalition_value = game.coalition_value(coalition)
  num_perms = 0
  for perm_tup in itertools.permutations(range(game.num_players())):
    perm = list(perm_tup)
    value_with = empty_coalition_value
    coalition.fill(0)
    for idx in range(game.num_players()):
      value_without = value_with  # re-use the one computed from the last iter
      i = perm[idx]
      coalition[i] = 1
      value_with = game.coalition_value(coalition)
      shapley_values_sum[i] += value_with - value_without
    num_perms += 1
  return shapley_values_sum / num_perms


def compute_approximate_shapley_values(
    game: coalitional_game.CoalitionalGame,
    num_samples: int,
) -> np.ndarray:
  """Compute the Shapley values using Monte Carlo estimation.

  Specifically, applies the implementation described in Section 2.3 of Mitchell
  et al. "Sampling Permutations for Shapley Value Estimation".
  https://people.math.sc.edu/cooper/shapley.pdf

  Args:
    game: the game to compute Shapley values for.
    num_samples: number of permutations to sample

  Returns:
    shapley_values: a numpy array of Shapley values per player.
  """

  shapley_values_sum = np.zeros(game.num_players(), dtype=float)
  coalition = np.zeros(game.num_players(), dtype=int)
  empty_coalition_value = game.coalition_value(coalition)
  for _ in range(num_samples):
    perm = np.random.permutation(game.num_players())
    value_with = empty_coalition_value
    coalition.fill(0)
    for idx in range(game.num_players()):
      value_without = value_with  # re-use the one computed from the last iter
      i = perm[idx]
      coalition[i] = 1
      value_with = game.coalition_value(coalition)
      shapley_values_sum[i] += value_with - value_without
  return shapley_values_sum / num_samples

