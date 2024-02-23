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

"""Some general utility functions for coalitional games."""

import itertools
import numpy as np
from open_spiel.python.coalitional_games import coalitional_game


def compute_payoff_epsilon(
    game: coalitional_game.CoalitionValueCalculator,
    p: np.ndarray
) -> float:
  """For a payoff vector p, get max_e s.t. p dot c + e >= V(c).

  Warning! Enumerates all coalitions.

  Args:
    game: the game to enumerate.
    p: the payoff vector.

  Returns:
    the value max_e s.t. p dot c + e >= V(C) for all subsets C subseteq N.
  """
  epsilon = 0
  for c in itertools.product([0, 1], repeat=game.get_num_players()):
    coalition = np.asarray(c)
    val_c = game.get_coalition_values(coalition)
    payoffs_to_coalition = np.inner(p, coalition)
    epsilon = max(epsilon, val_c - payoffs_to_coalition)
  return epsilon
