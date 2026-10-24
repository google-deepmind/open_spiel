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

"""Gin rummy constants: card encoding, phases, action ids and melds.

Also the meld masks, overlaps and points, and the state dtypes.

Card id = suit * 13 + rank, suits in the order spades, clubs, diamonds,
hearts and ranks ace (0) to king (12), as in OpenSpiel. The 185 melds are
the sets of three or four cards of a rank and the runs of three to five
cards of a suit. Every encoding here (cards, melds, phases, action ids,
limits) is OpenSpiel's gin_rummy's.
"""

import functools

import chex
import jax.numpy as jnp
import numpy as np

__all__ = [
    "ACTION_DRAW_STOCK",
    "ACTION_DRAW_UPCARD",
    "ACTION_KNOCK",
    "ACTION_MELD_BASE",
    "ACTION_PASS",
    "ALL_MELDS",
    "CHANCE_PLAYER",
    "DP_STATE_SIZE",
    "GIN_BONUS",
    "HAND_SIZE",
    "KNOCK_THRESHOLD",
    "MAX_HAND_SIZE",
    "MAX_NUM_DRAW_UPCARD_ACTIONS",
    "MAX_VALID_MELDS",
    "NUM_ACTIONS",
    "NUM_CARDS",
    "NUM_MELDS",
    "NUM_RANKS",
    "NUM_SUITS",
    "PHASE_DEAL",
    "PHASE_DISCARD",
    "PHASE_DRAW",
    "PHASE_FIRST_UPCARD",
    "PHASE_GAME_OVER",
    "PHASE_KNOCK",
    "PHASE_LAYOFF",
    "PHASE_WALL",
    "STATE_DTYPES",
    "TERMINAL_PLAYER",
    "UNDERCUT_BONUS",
    "WALL_STOCK_SIZE",
    "assert_state_dtypes",
    "card_rank",
    "card_str",
    "card_suit",
    "generate_all_melds",
    "get_card_points",
    "get_card_values",
    "get_meld_cards_bool",
    "get_meld_masks",
    "get_meld_overlap",
    "get_meld_overlap_padded",
    "get_meld_points",
    "get_meld_points_padded",
    "get_meld_sizes",
    "make_card",
    "meld_to_mask",
]

# =============================================================================
# Constants: Python ints, cast to jnp types at point of use
# =============================================================================
NUM_CARDS = 52
NUM_RANKS = 13
NUM_SUITS = 4
HAND_SIZE = 10
KNOCK_THRESHOLD = 10
GIN_BONUS = 25
UNDERCUT_BONUS = 25


# Card point values (A=1, 2-9=face, 10/J/Q/K=10)
@functools.cache
def get_card_points():
  """Returns the deadwood points of each rank.

  Returns:
    int32[13]: ace 1 through 10, face cards 10.
  """
  return np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10], dtype=np.int32)


@functools.cache
def get_card_values():
  """Returns the deadwood point value of each card id.

  Returns:
    int8[52]: the value of rank i % 13.
  """
  return np.array(
      [get_card_points()[i % NUM_RANKS] for i in range(NUM_CARDS)],
      dtype=np.int8,
  )


# Game phases
PHASE_DEAL = 0  # Chance node: dealing cards
PHASE_FIRST_UPCARD = 1
PHASE_DRAW = 2
PHASE_DISCARD = 3
PHASE_KNOCK = 4
PHASE_LAYOFF = 5
PHASE_WALL = 6  # the wall phase (stock at the wall size)
PHASE_GAME_OVER = 7

# Limits
MAX_NUM_DRAW_UPCARD_ACTIONS = 50  # the game ends on the 50th upcard draw
WALL_STOCK_SIZE = 2  # When stock <= this, enter wall phase

# Actions
# 0-51: card actions (deal in chance nodes, discard in regular play)
ACTION_DRAW_UPCARD = 52
ACTION_DRAW_STOCK = 53
ACTION_PASS = 54
ACTION_KNOCK = 55
ACTION_MELD_BASE = 56
# Actions 56-240 are meld declarations (185 melds)

NUM_MELDS = 185
NUM_ACTIONS = 241  # 0-51 cards, 52-55 special, 56-240 melds

# Special player IDs
CHANCE_PLAYER = -1
TERMINAL_PLAYER = -4

# =============================================================================
# State dtype specification for chex validation
# =============================================================================
# Maps state field names to expected dtypes. Used by assert_state_dtypes()
# to catch int64 promotion bugs early.
STATE_DTYPES = {
    # Hands and cards
    "player0_hand": jnp.int8,
    "player1_hand": jnp.int8,
    "deck": jnp.int8,
    "discard_pile": jnp.int8,
    "upcard": jnp.int8,
    # Game state
    "current_player": jnp.int8,
    "phase": jnp.int8,
    "oklahoma": jnp.bool_,
    # Rule parameters (game.DEFAULT_RULES) and their state
    "allow_upcard_throwback": jnp.bool_,
    "wall_knock_pick": jnp.bool_,
    "spade_doubles": jnp.bool_,
    "score_multiplier": jnp.int8,
    "upcard_taken": jnp.int8,
    "knock_card": jnp.int8,
    "deal_cards_dealt": jnp.int8,
    "prev_player": jnp.int8,
    # Tracking
    "pass_count": jnp.int8,
    "pass_on_first_upcard": jnp.bool_,
    "drawn_from_discard": jnp.int8,
    "num_draw_upcard": jnp.int32,
    "prev_upcard": jnp.int8,
    "repeated_move": jnp.bool_,
    # Knock/layoff state
    "knocked": jnp.bool_,
    "knocker": jnp.int8,
    "layed_melds": jnp.bool_,
    "layoffs_mask": jnp.bool_,
    "finished_layoffs": jnp.bool_,
    "deadwood": jnp.int32,  # Critical: must stay int32
    "knocker_deadwood": jnp.int32,  # Critical: must stay int32
    # Game outcome
    "done": jnp.bool_,
    "winner": jnp.int8,
    "p0_score": jnp.int32,  # Critical: must stay int32
    "p1_score": jnp.int32,  # Critical: must stay int32
    # Cached meld computation
    "cached_min_deadwood": jnp.int32,
    "cached_legal_knocker_melds": jnp.bool_,
    "cached_legal_discard_mask": jnp.bool_,
    "cached_all_discard_deadwoods": jnp.int16,
    "can_knock": jnp.bool_,
    "player_deadwood": jnp.int16,
}


def assert_state_dtypes(state):
  """Raises if a field of the state has a dtype other than STATE_DTYPES.

  A check for tests.

  Args:
    state: The state dict.
  """
  for field, expected_dtype in STATE_DTYPES.items():
    if field in state:
      chex.assert_type(state[field], expected_dtype)


# =============================================================================
# Meld encoding
# =============================================================================


def card_rank(card):
  """Returns the rank (0-12) of a card: card = suit * 13 + rank.

  Args:
    card: The card id.

  Returns:
    The rank.
  """
  return card % NUM_RANKS


def card_suit(card):
  """Returns the suit (0-3) of a card: card = suit * 13 + rank.

  Args:
    card: The card id.

  Returns:
    The suit.
  """
  return card // NUM_RANKS


def make_card(suit, rank):
  """Returns the card of a suit and rank: card = suit * 13 + rank.

  Args:
    suit: The suit.
    rank: The rank.

  Returns:
    The card id.
  """
  return suit * NUM_RANKS + rank


def card_str(card):
  """Returns a card as a human-readable string (e.g., 'As', '2c', 'Kh').

  Args:
    card: The card id.

  Returns:
    The string.
  """
  suit = card // NUM_RANKS
  rank = card % NUM_RANKS
  rank_chars = "A23456789TJQK"
  suit_chars = "scdh"  # spades, clubs, diamonds, hearts
  return rank_chars[rank] + suit_chars[suit]


def generate_all_melds():
  """Generates all valid melds (sets and runs).

  185 melds total, in OpenSpiel's order. Card encoding: card = suit * 13 +
  rank.

  Returns:
    The melds, each a list of card ids.
  """
  melds = [None] * 185

  # Rank melds (sets) - indices 0-64
  # Same rank, different suits
  for rank in range(NUM_RANKS):
    # Size 3: missing one suit
    for missing_suit in range(NUM_SUITS):
      meld_id = rank * 5 + missing_suit
      cards = tuple(
          make_card(s, rank) for s in range(NUM_SUITS) if s != missing_suit
      )
      melds[meld_id] = cards
    # Size 4: all suits
    meld_id = rank * 5 + 4
    cards = tuple(make_card(s, rank) for s in range(NUM_SUITS))
    melds[meld_id] = cards

  # Suit melds (runs) - indices 65-184
  # Same suit, consecutive ranks
  offset = 65
  # Size 3 runs: 11 per suit (start ranks 0-10)
  for suit in range(NUM_SUITS):
    for start_rank in range(NUM_RANKS - 2):  # 0-10
      meld_id = offset + suit * 11 + start_rank
      cards = tuple(
          make_card(suit, r) for r in range(start_rank, start_rank + 3)
      )
      melds[meld_id] = cards

  offset = 109  # 65 + 44
  # Size 4 runs: 10 per suit (start ranks 0-9)
  for suit in range(NUM_SUITS):
    for start_rank in range(NUM_RANKS - 3):  # 0-9
      meld_id = offset + suit * 10 + start_rank
      cards = tuple(
          make_card(suit, r) for r in range(start_rank, start_rank + 4)
      )
      melds[meld_id] = cards

  offset = 149  # 109 + 40
  # Size 5 runs: 9 per suit (start ranks 0-8)
  for suit in range(NUM_SUITS):
    for start_rank in range(NUM_RANKS - 4):  # 0-8
      meld_id = offset + suit * 9 + start_rank
      cards = tuple(
          make_card(suit, r) for r in range(start_rank, start_rank + 5)
      )
      melds[meld_id] = cards

  return melds


ALL_MELDS = generate_all_melds()


def meld_to_mask(meld):
  """Returns a meld as a card mask.

  Args:
    meld: The meld's card ids.

  Returns:
    int8[52]: 1 for each card of the meld.
  """
  mask = np.zeros(NUM_CARDS, dtype=np.int8)
  for card in meld:
    mask[card] = 1
  return mask


@functools.cache
def get_meld_masks():
  """Returns the card mask of every meld.

  Returns:
    int8[185, 52].
  """
  return np.array([meld_to_mask(m) for m in ALL_MELDS], dtype=np.int8)


@functools.cache
def get_meld_cards_bool():
  """Returns which cards each meld contains.

  Returns:
    bool[185, 52]: [m, c] is True if meld m contains card c.
  """
  return np.array([meld_to_mask(m) for m in ALL_MELDS], dtype=np.bool_)


@functools.cache
def get_meld_sizes():
  """Returns the size of each meld.

  Returns:
    int8[185]: 3, 4 or 5 cards per meld.
  """
  return np.array([len(m) for m in ALL_MELDS], dtype=np.int8)


# Maximum number of valid melds to consider (padded size for DP)
MAX_VALID_MELDS = 32
# Maximum hand size (11 cards before discard, 10 after)
MAX_HAND_SIZE = 11
# DP state space size (2^11 = 2048)
DP_STATE_SIZE = 2048

# =============================================================================
# Deadwood calculation
# =============================================================================


# Precompute meld overlap matrix
def _compute_overlap_matrix():
  """Which pairs of melds share a card (a meld overlaps itself).

  Returns:
    bool[NUM_MELDS, NUM_MELDS], True where melds i and j cannot both be used.
  """
  overlap = np.zeros((NUM_MELDS, NUM_MELDS), dtype=np.bool_)
  for i in range(NUM_MELDS):
    for j in range(NUM_MELDS):
      # A meld overlaps with itself (can't use same meld twice)
      if i == j:
        overlap[i, j] = True
      else:
        set_i = set(ALL_MELDS[i])
        set_j = set(ALL_MELDS[j])
        if set_i & set_j:
          overlap[i, j] = True
  return np.array(overlap)


@functools.cache
def get_meld_overlap():
  """Returns which melds share a card.

  Returns:
    bool[185, 185]: the overlap matrix.
  """
  return _compute_overlap_matrix()


@functools.cache
def get_meld_overlap_padded():
  """Returns the meld overlap matrix padded to (NUM_MELDS+1, NUM_MELDS+1).

  Padding row/col (index NUM_MELDS) overlaps with everything, so
  invalid meld indices become unusable in combo enumeration.

  Returns:
    bool[186, 186].
  """
  overlap = _compute_overlap_matrix()
  # Pad with True (overlaps with everything)
  padded = np.ones((NUM_MELDS + 1, NUM_MELDS + 1), dtype=np.bool_)
  padded[:NUM_MELDS, :NUM_MELDS] = overlap
  return np.array(padded)


@functools.cache
def _compute_meld_points():
  """The point value of every meld.

  Returns:
    int32[NUM_MELDS], each meld's points.
  """
  # Use plain numpy for computation to avoid JAX tracing issues
  card_points_np = np.array(
      [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10], dtype=np.int32
  )
  points = np.zeros(NUM_MELDS, dtype=np.int32)
  for i, meld in enumerate(ALL_MELDS):
    for card in meld:
      rank = card % NUM_RANKS
      points[i] += card_points_np[rank]
  return np.array(points)


@functools.cache
def get_meld_points():
  """Returns the deadwood points each meld removes.

  Returns:
    int32[185].
  """
  return _compute_meld_points()


@functools.cache
def get_meld_points_padded():
  """Returns the meld points padded to (NUM_MELDS+1,).

  Padding entry (index NUM_MELDS) has 0 points, so invalid meld indices
  contribute nothing to combo deadwood calculations.

  Returns:
    int32[186].
  """
  points = _compute_meld_points()
  padded = np.zeros(NUM_MELDS + 1, dtype=np.int32)
  padded[:NUM_MELDS] = points
  return np.array(padded)
