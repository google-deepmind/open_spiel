# Copyright 2026 DeepMind Technologies Limited
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

"""Card helpers for the gin tests: 'Td' <-> card id (suit order s c d h)."""

import numpy as np

from open_spiel.python.jax.games.gin_rummy import cards_and_melds as cm

SUITS = "scdh"
RANKS = "A23456789TJQK"


def card(name: str) -> int:
  """The card id of a two-character name such as "As".

  Args:
    name: Rank then suit.

  Returns:
    The card id.
  """
  assert len(name) == 2, name
  return SUITS.index(name[1]) * 13 + RANKS.index(name[0])


def cards(names: str) -> list[int]:
  """The card ids of space-separated names.

  Args:
    names: The card names.

  Returns:
    The card ids.
  """
  return [card(n) for n in names.split()]


def hand_mask(names: str) -> np.ndarray:
  """The int8[52] membership mask of space-separated card names.

  Args:
    names: The card names.

  Returns:
    The mask.
  """
  m = np.zeros(cm.NUM_CARDS, np.int8)
  m[cards(names)] = 1
  return m


def meld_id(names: str) -> int:
  """Index in [0, 185) of the meld with exactly these cards.

  Args:
    names: The meld's card names.

  Returns:
    The meld index.
  """
  target = hand_mask(names).astype(bool)
  masks = np.asarray(cm.get_meld_masks()).astype(bool)
  hits = np.nonzero((masks == target[None, :]).all(axis=1))[0]
  assert len(hits) == 1, (names, hits)
  return int(hits[0])
