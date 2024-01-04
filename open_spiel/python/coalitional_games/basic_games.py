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

"""Some basic coalitional games.

Many of these are taken from examples in the "Computational Aspects of
Cooperative Game Theory" text book by Georgios Chalkiadakis, Edith Elkind, and
Michael Wooldridge.
"""

from typing import Dict, Tuple

import numpy as np

from open_spiel.python.coalitional_games import coalitional_game


class IceCreamGame(coalitional_game.CoalitionalGame):
  """Example 2.2 from CACGT book by Chalkiadakis, Elkind, and Wooldridge."""

  def __init__(self):
    super().__init__(num_players=3)

  def coalition_value(self, coalition: np.ndarray) -> float:
    """Encodes the payoffs."""
    # indices ordered as C M P
    if coalition.sum() < 2:
      return 0.0
    elif coalition[0] == 1 and coalition[1] == 1 and coalition[2] == 0:
      # {C, M}
      return 500.0
    elif coalition[0] == 1 and coalition[1] == 0 and coalition[2] == 1:
      # {C, P}
      return 500.0
    elif coalition[0] == 0 and coalition[1] == 1 and coalition[2] == 1:
      # {M, P}
      return 750.0
    elif coalition.sum() == 3:
      return 1000.0
    else:
      raise RuntimeError("Invalid coalition")


class TabularGame(coalitional_game.CoalitionalGame):
  """A game represented by a table of values."""

  def __init__(self, table: Dict[Tuple[int, ...], float]):
    super().__init__(num_players=-1)  # set num players to -1 for now
    for key in table:
      if self._num_players < 0:
        self._num_players = len(key)
      else:
        assert len(key) == self._num_players
    assert self._num_players >= 1
    self._table = table

  def coalition_value(self, coalition: np.ndarray) -> float:
    return self._table[tuple(coalition)]

