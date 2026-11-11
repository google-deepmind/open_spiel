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

"""Heads-up no-limit Texas hold'em with 200-chip stacks and blinds 1/2.

The game is OpenSpiel's `universal_poker` in this configuration
(`UNIVERSAL_POKER_GAME_STRING`):

  universal_poker(betting=nolimit,bettingAbstraction=fullgame,blind=2 1,
      firstPlayer=2 1 1 1,numBoardCards=0 3 1 1,numHoleCards=2,
      numPlayers=2,numRanks=13,numRounds=4,numSuits=4,stack=200 200)

`no_limit_holdem` accepts those parameters and refuses any other value.

State: a `GameState` with the hole and board cards (`INVALID_CARD` until
dealt), each player's total commitment, the betting round and the action
history. Chance actions are card ids 0..51 (rank = id // 4), uniform over the
cards left. Player 1 posts the small blind and acts first preflop; player 0
acts first after the flop.

Actions, as in universal_poker: 0 folds, 1 calls or checks, and a ∈ 2..200
raises to a total commitment of a chips. A betting round holds at most
`MAX_ROUND_ACTIONS` (64) actions, so raising closes once a raise would leave
no room for both players to answer it.

Observation (f32[108]): observer one-hot, own hole cards and board as 52-card
one-hots, each player's commitment. Information state (f32[460]): the same
one-hots, then per action a (call, raise) bit pair and its size. Returns:
chips won or lost.

Options: `use_showdown_cache` (default True) scores showdowns with a precomputed
five-card table (about 48 KiB, built at import); False computes hand ranks
instead, with the same scores (docs/no_limit_holdem_hand_evaluator.md).
`track_information_state_history=False` keeps no action history in the
state (less memory per state): such a game has no information state, and
`info_state_shape`, `information_state_tensor` and
`information_state_from_os` raise ValueError.
"""

import itertools
import math
import typing
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from open_spiel.python.jax.games import game as game_lib
from open_spiel.python.jax.games import interop

if typing.TYPE_CHECKING:
  import pyspiel


FOLD = 0
CALL = 1
MIN_RAISE_ACTION = 2
STACK = 200
NUM_ACTIONS = STACK + 1

NUM_PLAYERS = 2
NUM_RANKS = 13
NUM_SUITS = 4
NUM_CARDS = NUM_RANKS * NUM_SUITS
NUM_HOLE_CARDS = 2
MAX_BOARD_CARDS = 5
MAX_DEALT_CARDS = NUM_PLAYERS * NUM_HOLE_CARDS + MAX_BOARD_CARDS
MAX_GAME_LENGTH = 118
# Actions per betting round (ACPC's MAX_NUM_ACTIONS).
MAX_ROUND_ACTIONS = 64
OBS_DIM = 2 * (NUM_PLAYERS + NUM_CARDS)
INFO_STATE_DIM = NUM_PLAYERS + 2 * NUM_CARDS + 3 * MAX_GAME_LENGTH

SMALL_BLIND_PLAYER = 1
BIG_BLIND_PLAYER = 0
SMALL_BLIND = 1
BIG_BLIND = 2
TERMINAL = game_lib.TERMINAL_PLAYER_ID


UNIVERSAL_POKER_GAME_STRING = (
    "universal_poker(betting=nolimit,bettingAbstraction=fullgame,"
    "blind=2 1,firstPlayer=2 1 1 1,numBoardCards=0 3 1 1,"
    "numHoleCards=2,numPlayers=2,numRanks=13,numRounds=4,numSuits=4,"
    "stack=200 200)"
)

_SUPPORTED_KWARGS = {
    "betting": "nolimit",
    "bettingAbstraction": "fullgame",
    "blind": "2 1",
    "firstPlayer": "2 1 1 1",
    "numBoardCards": "0 3 1 1",
    "numHoleCards": 2,
    "numPlayers": 2,
    "numRanks": 13,
    "numRounds": 4,
    "numSuits": 4,
    "stack": "200 200",
}


class GameState(NamedTuple):
  """State for heads-up no-limit hold'em.

  Attributes:
    action_chars: (MAX_GAME_LENGTH,), int32 ascii code.
    action_sizes: (MAX_GAME_LENGTH,), int32.
    all_in: Scalar bool.
    board_cards: (5,), int32, padded with INVALID_CARD.
    board_count: Scalar int32.
    current_player: -1 chance, 0/1 player, -4 terminal.
    deck_mask: (52,), bool.
    folded_player: Scalar int32, -1 when no fold.
    history_len: Scalar int32.
    is_terminal: Scalar bool.
    last_raise_size: Scalar int32.
    num_dealt: Scalar int32, total chance cards dealt.
    private_cards: (2, 2), int32, padded with INVALID_CARD.
    round: 0 preflop, 1 flop, 2 turn, 3 river.
    round_actions: Scalar int32.
    spent: (2,), current total commitment per player.
  """

  private_cards: jax.Array
  board_cards: jax.Array
  deck_mask: jax.Array
  num_dealt: jax.Array
  board_count: jax.Array

  current_player: jax.Array
  round: jax.Array
  spent: jax.Array
  last_raise_size: jax.Array
  round_actions: jax.Array
  all_in: jax.Array
  folded_player: jax.Array

  action_chars: jax.Array
  action_sizes: jax.Array
  history_len: jax.Array

  is_terminal: jax.Array


def _card_rank(card: jax.Array) -> jax.Array:
  """The rank of `card`.

  Args:
    card: The card.

  Returns:
    The rank.
  """
  return card // NUM_SUITS


def _card_suit(card: jax.Array) -> jax.Array:
  """The suit of `card`.

  Args:
    card: The card.

  Returns:
    The suit.
  """
  return card % NUM_SUITS


def _append_history(
    state: GameState, char_code: jax.Array, size: jax.Array
) -> GameState:
  """Returns `state` with one action appended to its history.

  Args:
    state: The state.
    char_code: The action's character code ("d", "f", "c" or "p").
    size: The raise target, or 0.

  Returns:
    The state with the action appended.

  Raises:
    ValueError: The history already holds MAX_GAME_LENGTH actions (checked
      eagerly when `history_len` is concrete, else by a debug callback).
  """
  idx = state.history_len
  try:
    idx_value = int(idx)
  except (
      TypeError,
      jax.errors.ConcretizationTypeError,
      jax.errors.TracerIntegerConversionError,
  ):
    idx_value = None
  if idx_value is not None and idx_value >= MAX_GAME_LENGTH:
    raise ValueError(
        "no_limit_holdem action history exceeded MAX_GAME_LENGTH="
        f"{MAX_GAME_LENGTH}"
    )
  jax.lax.cond(
      idx < MAX_GAME_LENGTH,
      lambda _: None,
      lambda i: jax.debug.callback(_check_history_capacity, i),
      idx,
  )
  return state._replace(
      action_chars=state.action_chars.at[idx].set(char_code.astype(jnp.int32)),
      action_sizes=state.action_sizes.at[idx].set(size.astype(jnp.int32)),
      history_len=idx + jnp.int32(1),
  )


def _check_history_capacity(idx) -> None:
  if int(idx) >= MAX_GAME_LENGTH:
    raise ValueError(
        "no_limit_holdem action history exceeded MAX_GAME_LENGTH="
        f"{MAX_GAME_LENGTH}"
    )


def _check_showdown_cards_complete(cards) -> None:
  if (cards < 0).any():
    raise ValueError(
        "no_limit_holdem showdown requires two private cards per player "
        "and five board cards"
    )


def _check_showdown_cards_complete_if_eager(state: GameState) -> None:
  """Checks that a terminal showdown has every card dealt, when not traced.

  Under tracing the check is skipped.

  Args:
    state: The state.

  Raises:
    ValueError: The state is a showdown with a card not yet dealt.
  """
  try:
    is_terminal = bool(state.is_terminal)
    any_folded = int(state.folded_player) >= 0
  except (
      TypeError,
      jax.errors.ConcretizationTypeError,
      jax.errors.TracerBoolConversionError,
      jax.errors.TracerIntegerConversionError,
  ):
    return
  if is_terminal and not any_folded:
    all_cards = jnp.concatenate(
        [state.private_cards.reshape(-1), state.board_cards]
    )
    _check_showdown_cards_complete(all_cards)


def _cards_one_hot(cards: jax.Array) -> jax.Array:
  """Returns the 52-card one-hot of the dealt cards.

  Args:
    cards: Card ids; negative entries (not yet dealt) are skipped.

  Returns:
    f32[NUM_CARDS], 1 at each dealt card.
  """
  one_hot = jnp.zeros(NUM_CARDS, dtype=jnp.float32)

  def body(i, acc):
    """Sets card `i`'s bit when it is dealt.

    Args:
      i: The card slot.
      acc: The one-hot so far.

    Returns:
      The updated one-hot.
    """
    card = cards[i]
    return jnp.where(card >= 0, acc.at[jnp.maximum(card, 0)].set(1.0), acc)

  return jax.lax.fori_loop(0, cards.shape[0], body, one_hot)


def _top_ranks(mask: jax.Array, n: int) -> jax.Array:
  """The `n` highest ranks set in `mask`, -1 past the last.

  Args:
    mask: bool[13], the ranks present.
    n: The number of ranks.

  Returns:
    i32[n], the ranks.
  """
  # Missing slots become -1, which sorts below real ranks 0..12 and is only
  # read in categories where that kicker slot is irrelevant.
  ranks = jnp.arange(NUM_RANKS, dtype=jnp.int32)
  keyed = jnp.where(mask, ranks, jnp.int32(-1))
  ordered = jnp.flip(jnp.sort(keyed))
  return ordered[:n]


def _pack_score(category: jax.Array, kickers: jax.Array) -> jax.Array:
  """Packs a hand category and its kickers into one score.

  Args:
    category: The hand category.
    kickers: The kicker ranks.

  Returns:
    i32[], the packed score.
  """
  padded = jnp.pad(kickers.astype(jnp.int32), (0, 5 - kickers.shape[0]))
  score = category.astype(jnp.int32)
  for i in range(5):
    score = score * jnp.int32(15) + padded[i]
  return score


def _straight_high(present: jax.Array) -> jax.Array:
  """The high rank of the best straight, or -1.

  Args:
    present: bool[13], the ranks present.

  Returns:
    i32[], the high rank (3 for the wheel).
  """
  highs = []
  for high in range(4, NUM_RANKS):
    window = present[high - 4 : high + 1]
    highs.append(jnp.where(jnp.all(window), high, -1))
  wheel = jnp.where(
      present[12] & present[0] & present[1] & present[2] & present[3], 3, -1
  )
  return jnp.max(jnp.array(highs + [wheel], dtype=jnp.int32))


def _score_5_categorical(cards: jax.Array) -> jax.Array:
  """Returns the packed score of a 5-card hand, computed by category.

  The LUT-free reference; a higher score is a better hand.

  Args:
    cards: i32[5], the card ids.

  Returns:
    The i32 packed score: the category, then up to five kickers, base 15.
  """
  ranks = _card_rank(cards)
  suits = _card_suit(cards)
  counts = jnp.sum(
      ranks[:, None] == jnp.arange(NUM_RANKS, dtype=jnp.int32)[None, :],
      axis=0,
  )
  present = counts > 0
  flush = jnp.all(suits == suits[0])
  straight = _straight_high(present)

  four = jnp.max(jnp.where(counts == 4, jnp.arange(NUM_RANKS), -1))
  trips = jnp.max(jnp.where(counts == 3, jnp.arange(NUM_RANKS), -1))
  pair_ranks = _top_ranks(counts == 2, 2)
  num_pairs = jnp.sum(counts == 2)

  straight_flush_score = _pack_score(jnp.int32(8), jnp.array([straight]))
  four_score = _pack_score(
      jnp.int32(7),
      jnp.concatenate(
          [jnp.array([four]), _top_ranks(present & (counts < 4), 1)]
      ),
  )
  full_house_score = _pack_score(
      jnp.int32(6), jnp.array([trips, pair_ranks[0]], dtype=jnp.int32)
  )
  flush_score = _pack_score(jnp.int32(5), _top_ranks(present, 5))
  straight_score = _pack_score(jnp.int32(4), jnp.array([straight]))
  trips_score = _pack_score(
      jnp.int32(3),
      jnp.concatenate([jnp.array([trips]), _top_ranks(counts == 1, 2)]),
  )
  two_pair_score = _pack_score(
      jnp.int32(2), jnp.concatenate([pair_ranks, _top_ranks(counts == 1, 1)])
  )
  pair_score = _pack_score(
      jnp.int32(1),
      jnp.concatenate([jnp.array([pair_ranks[0]]), _top_ranks(counts == 1, 3)]),
  )
  high_score = _pack_score(jnp.int32(0), _top_ranks(present, 5))

  return jnp.where(
      flush & (straight >= 0),
      straight_flush_score,
      jnp.where(
          four >= 0,
          four_score,
          jnp.where(
              (trips >= 0) & (num_pairs >= 1),
              full_house_score,
              jnp.where(
                  flush,
                  flush_score,
                  jnp.where(
                      straight >= 0,
                      straight_score,
                      jnp.where(
                          trips >= 0,
                          trips_score,
                          jnp.where(
                              num_pairs >= 2,
                              two_pair_score,
                              jnp.where(num_pairs == 1, pair_score, high_score),
                          ),
                      ),
                  ),
              ),
          ),
      ),
  )


def _np_top_ranks(mask: np.ndarray, n: int) -> np.ndarray:
  """Row-wise `_top_ranks` in numpy.

  Args:
    mask: bool[B, 13], the ranks present.
    n: The number of ranks.

  Returns:
    int32[B, n], the ranks.
  """
  ranks = np.arange(NUM_RANKS, dtype=np.int32)
  keyed = np.where(mask, ranks[None, :], np.int32(-1))
  return np.flip(np.sort(keyed, axis=1), axis=1)[:, :n]


def _np_pack_score(
    category: np.ndarray | int, kickers: np.ndarray
) -> np.ndarray:
  """Row-wise `_pack_score` in numpy.

  Args:
    category: The categories, or one for every row.
    kickers: int[B, k], the kickers.

  Returns:
    int32[B], the packed scores.
  """
  padded = np.zeros((kickers.shape[0], 5), dtype=np.int32)
  padded[:, : kickers.shape[1]] = kickers.astype(np.int32)
  score = np.asarray(category, dtype=np.int32)
  if score.ndim == 0:
    score = np.full(kickers.shape[0], score, dtype=np.int32)
  for i in range(5):
    score = score * np.int32(15) + padded[:, i]
  return score.astype(np.int32)


def _np_score_5(cards: np.ndarray) -> np.ndarray:
  """The packed scores of 5-card hands, in numpy.

  Args:
    cards: int[B, 5], the hands.

  Returns:
    int32[B], the scores.
  """
  sorted_ranks = np.sort(cards // NUM_SUITS, axis=1)
  suits = cards % NUM_SUITS
  flush = np.all(suits == suits[:, :1], axis=1)
  return _np_score_5_from_ranks_flush(sorted_ranks, flush)


def _np_score_5_from_ranks_flush(
    sorted_ranks: np.ndarray, flush: np.ndarray
) -> np.ndarray:
  """Returns the packed scores of 5-card hands from their ranks and flush bits.

  The batched NumPy counterpart of `_score_5_categorical`.

  Args:
    sorted_ranks: int[N, 5], each hand's ranks.
    flush: [N], whether each hand is a flush.

  Returns:
    i32[N], the packed scores.
  """
  counts = np.sum(
      sorted_ranks[:, :, None]
      == np.arange(NUM_RANKS, dtype=np.int16)[None, None, :],
      axis=1,
  ).astype(np.int8)
  present = counts > 0
  flush = flush.astype(np.bool_)

  straight = np.full(sorted_ranks.shape[0], -1, dtype=np.int32)
  for high in range(4, NUM_RANKS):
    straight = np.where(
        np.all(present[:, high - 4 : high + 1], axis=1), high, straight
    )
  wheel = np.all(present[:, [12, 0, 1, 2, 3]], axis=1)
  straight = np.where(wheel, np.int32(3), straight)

  rank_values = np.arange(NUM_RANKS, dtype=np.int32)[None, :]
  four = np.max(np.where(counts == 4, rank_values, -1), axis=1)
  trips = np.max(np.where(counts == 3, rank_values, -1), axis=1)
  pair_ranks = _np_top_ranks(counts == 2, 2)
  num_pairs = np.sum(counts == 2, axis=1)

  straight_flush_score = _np_pack_score(8, straight[:, None])
  four_score = _np_pack_score(
      7,
      np.concatenate(
          [four[:, None], _np_top_ranks(present & (counts < 4), 1)], axis=1
      ),
  )
  full_house_score = _np_pack_score(
      6, np.stack([trips, pair_ranks[:, 0]], axis=1)
  )
  flush_score = _np_pack_score(5, _np_top_ranks(present, 5))
  straight_score = _np_pack_score(4, straight[:, None])
  trips_score = _np_pack_score(
      3,
      np.concatenate([trips[:, None], _np_top_ranks(counts == 1, 2)], axis=1),
  )
  two_pair_score = _np_pack_score(
      2,
      np.concatenate([pair_ranks, _np_top_ranks(counts == 1, 1)], axis=1),
  )
  pair_score = _np_pack_score(
      1,
      np.concatenate(
          [pair_ranks[:, :1], _np_top_ranks(counts == 1, 3)], axis=1
      ),
  )
  high_score = _np_pack_score(0, _np_top_ranks(present, 5))

  score = high_score
  score = np.where(num_pairs == 1, pair_score, score)
  score = np.where(num_pairs >= 2, two_pair_score, score)
  score = np.where(trips >= 0, trips_score, score)
  score = np.where(straight >= 0, straight_score, score)
  score = np.where(flush, flush_score, score)
  score = np.where((trips >= 0) & (num_pairs >= 1), full_house_score, score)
  score = np.where(four >= 0, four_score, score)
  score = np.where(flush & (straight >= 0), straight_flush_score, score)
  return score.astype(np.int32)


def _rank_flush_index(
    sorted_ranks: np.ndarray, flush: np.ndarray
) -> np.ndarray:
  """The lookup-table index of sorted ranks and a flush flag.

  Args:
    sorted_ranks: int[B, 5], the sorted ranks.
    flush: bool[B], the flush flags.

  Returns:
    int32[B], the indices.
  """
  adjusted = sorted_ranks + np.arange(5, dtype=np.int32)[None, :]
  rank_index = np.sum(
      _FIVE_CARD_RANK_PATTERN_CHOOSE[adjusted, np.arange(1, 6, dtype=np.int32)],
      axis=1,
  )
  return rank_index * 2 + flush.astype(np.int32)


def _build_five_card_rank_pattern_choose() -> np.ndarray:
  """Binomial coefficients for the rank-pattern index.

  Returns:
    int32[NUM_RANKS + 5, 6], n choose k.
  """
  choose = np.zeros((NUM_RANKS + 5, 6), dtype=np.int32)
  for n in range(NUM_RANKS + 5):
    for k in range(1, 6):
      if n >= k:
        choose[n, k] = math.comb(n, k)
  return choose


def _build_five_card_score_lut() -> np.ndarray:
  """Returns the packed score of every 5-card rank pattern, flush or not.

  The table is indexed by `_rank_flush_index`; patterns with five cards of one
  rank are impossible and stay -1.

  Returns:
    i32[2 * C(NUM_RANKS + 4, 5)], the scores.
  """
  rank_patterns = np.array(
      list(itertools.combinations_with_replacement(range(NUM_RANKS), 5)),
      dtype=np.int16,
  )
  possible = (
      np.max(
          np.sum(
              rank_patterns[:, :, None]
              == np.arange(NUM_RANKS, dtype=np.int16)[None, None, :],
              axis=1,
          ),
          axis=1,
      )
      <= 4
  )
  rank_patterns = rank_patterns[possible]
  flush = np.concatenate([
      np.zeros(rank_patterns.shape[0], dtype=np.int32),
      np.ones(rank_patterns.shape[0], dtype=np.int32),
  ])
  ranks = np.concatenate([rank_patterns, rank_patterns], axis=0)
  scores = _np_score_5_from_ranks_flush(ranks, flush)
  lut = np.full(math.comb(NUM_RANKS + 4, 5) * 2, -1, dtype=np.int32)
  lut[_rank_flush_index(ranks, flush)] = scores
  return lut


_FIVE_CARD_RANK_PATTERN_CHOOSE = _build_five_card_rank_pattern_choose()
_FIVE_CARD_SCORE_LUT = _build_five_card_score_lut()


def _five_card_lut_index(cards: jax.Array) -> jax.Array:
  """The lookup-table index of a 5-card hand.

  Args:
    cards: i32[5], the cards.

  Returns:
    i32[], the index.
  """
  ranks = jnp.sort(_card_rank(cards).astype(jnp.int32))
  suits = _card_suit(cards)
  flush = jnp.all(suits == suits[0]).astype(jnp.int32)
  choose = jnp.asarray(_FIVE_CARD_RANK_PATTERN_CHOOSE, dtype=jnp.int32)
  adjusted = ranks + jnp.arange(5, dtype=jnp.int32)
  rank_index = jnp.sum(
      choose[adjusted, jnp.arange(1, 6, dtype=jnp.int32)], dtype=jnp.int32
  )
  return rank_index * jnp.int32(2) + flush


def _score_5(cards: jax.Array) -> jax.Array:
  """Returns the packed score for a 5-card poker hand via a precomputed LUT.

  Args:
    cards: i32[5], the hand.

  Returns:
    i32[], the packed score.
  """
  return jnp.asarray(_FIVE_CARD_SCORE_LUT, dtype=jnp.int32)[
      _five_card_lut_index(cards)
  ]


# NumPy (no JAX call at import); converted where it is used.
_FIVE_CARD_COMBOS = np.array(
    [
        [0, 1, 2, 3, 4],
        [0, 1, 2, 3, 5],
        [0, 1, 2, 3, 6],
        [0, 1, 2, 4, 5],
        [0, 1, 2, 4, 6],
        [0, 1, 2, 5, 6],
        [0, 1, 3, 4, 5],
        [0, 1, 3, 4, 6],
        [0, 1, 3, 5, 6],
        [0, 1, 4, 5, 6],
        [0, 2, 3, 4, 5],
        [0, 2, 3, 4, 6],
        [0, 2, 3, 5, 6],
        [0, 2, 4, 5, 6],
        [0, 3, 4, 5, 6],
        [1, 2, 3, 4, 5],
        [1, 2, 3, 4, 6],
        [1, 2, 3, 5, 6],
        [1, 2, 4, 5, 6],
        [1, 3, 4, 5, 6],
        [2, 3, 4, 5, 6],
    ],
    dtype=np.int32,
)


def best_seven_card_score(cards: jax.Array) -> jax.Array:
  """Returns an integer rank for a 7-card poker hand.

  Args:
    cards: i32[7], the cards.

  Returns:
    The rank of the best five cards; higher is better.
  """
  scores = jax.vmap(lambda combo: _score_5(cards[combo]))(
      jnp.asarray(_FIVE_CARD_COMBOS)
  )
  return jnp.max(scores)


def best_seven_card_score_uncached(cards: jax.Array) -> jax.Array:
  """Returns an integer rank for a 7-card poker hand without the LUT cache.

  Args:
    cards: i32[7], the cards.

  Returns:
    The rank of the best five cards; higher is better.
  """
  scores = jax.vmap(lambda combo: _score_5_categorical(cards[combo]))(
      jnp.asarray(_FIVE_CARD_COMBOS)
  )
  return jnp.max(scores)


class NoLimitHoldem(game_lib.Game, interop.PyspielInterop):
  """Heads-up no-limit hold'em; see the module docstring for parameters."""

  def __init__(
      self,
      *,
      use_showdown_cache: bool = True,
      track_information_state_history: bool = True,
      **universal_poker_params,
  ):
    """Heads-up no-limit hold'em.

    Args:
      use_showdown_cache: score showdowns with the five-card table (module
        doc); False computes hand ranks, with the same scores.
      track_information_state_history: False keeps no action history; the
        game then has no information state (module doc).
      **universal_poker_params: OpenSpiel universal_poker parameters (betting,
        numPlayers, stack, ...); only the heads-up no-limit values
        (`_SUPPORTED_KWARGS`) are accepted.

    Raises:
      TypeError: an option is not a bool.
      ValueError: a universal_poker parameter is unknown or not the
        supported value.
    """
    kwargs = universal_poker_params
    if not isinstance(use_showdown_cache, (bool, np.bool_)):
      raise TypeError(
          "use_showdown_cache must be a bool, got "
          f"{type(use_showdown_cache).__name__}"
      )
    if not isinstance(track_information_state_history, (bool, np.bool_)):
      raise TypeError(
          "track_information_state_history must be a bool, got "
          f"{type(track_information_state_history).__name__}"
      )
    self._use_showdown_cache = bool(use_showdown_cache)
    self._track_information_state_history = bool(
        track_information_state_history
    )
    unsupported = set(kwargs) - set(_SUPPORTED_KWARGS)
    if unsupported:
      raise ValueError(
          "no_limit_holdem supports only the heads-up no-limit "
          f"configuration; unsupported parameters: {sorted(unsupported)}"
      )
    mismatches = {
        key: (kwargs[key], expected)
        for key, expected in _SUPPORTED_KWARGS.items()
        if key in kwargs and kwargs[key] != expected
    }
    if mismatches:
      details = ", ".join(
          f"{key}={actual!r} (expected {expected!r})"
          for key, (actual, expected) in sorted(mismatches.items())
      )
      raise ValueError(
          "no_limit_holdem supports only the heads-up no-limit "
          f"configuration; mismatched parameters: {details}"
      )

  @property
  def num_actions(self) -> int:
    """The number of distinct player actions.

    OpenSpiel calls it `num_distinct_actions`.
    """
    return NUM_ACTIONS

  @property
  def num_players(self) -> int:
    """The number of players."""
    return NUM_PLAYERS

  @property
  def obs_shape(self) -> tuple[int, ...]:
    """The shape of `observation_tensor`."""
    return (OBS_DIM,)

  @property
  def info_state_shape(self) -> tuple[int, ...]:
    """The shape of `information_state_tensor`.

    It is `obs_shape` unless overridden.
    """
    self._require_history()
    return (INFO_STATE_DIM,)

  def _require_history(self) -> None:
    if not self._track_information_state_history:
      raise ValueError(
          "this NoLimitHoldem keeps no action history "
          "(track_information_state_history=False), so it has no "
          "information state"
      )

  @property
  def max_chance_outcomes(self) -> int:
    """The number of distinct chance outcomes; 0 for games without chance."""
    return NUM_CARDS

  def new_initial_state(self) -> GameState:
    """Returns the initial state.

    Returns:
      The state before the first move.
    """
    return GameState(
        private_cards=jnp.full(
            (NUM_PLAYERS, NUM_HOLE_CARDS),
            game_lib.INVALID_CARD,
            dtype=jnp.int32,
        ),
        board_cards=jnp.full(
            MAX_BOARD_CARDS, game_lib.INVALID_CARD, dtype=jnp.int32
        ),
        deck_mask=jnp.ones(NUM_CARDS, dtype=jnp.bool_),
        num_dealt=jnp.int32(0),
        board_count=jnp.int32(0),
        current_player=jnp.int32(game_lib.CHANCE_PLAYER_ID),
        round=jnp.int32(0),
        spent=jnp.array([BIG_BLIND, SMALL_BLIND], dtype=jnp.int32),
        last_raise_size=jnp.int32(BIG_BLIND),
        round_actions=jnp.int32(0),
        all_in=jnp.bool_(False),
        folded_player=jnp.int32(-1),
        action_chars=jnp.zeros(MAX_GAME_LENGTH, dtype=jnp.int32),
        action_sizes=jnp.zeros(MAX_GAME_LENGTH, dtype=jnp.int32),
        history_len=jnp.int32(0),
        is_terminal=jnp.bool_(False),
    )

  def _deal_card(self, state: GameState, action: jax.Array) -> GameState:
    """Deals one card: hole cards first, then the board.

    Args:
      state: The state.
      action: The card.

    Returns:
      The state after the deal.
    """
    card = jnp.asarray(action).astype(jnp.int32)
    deal_idx = state.num_dealt
    new_deck = state.deck_mask.at[card].set(False)

    p = jnp.minimum(deal_idx // NUM_HOLE_CARDS, NUM_PLAYERS - 1)
    slot = deal_idx % NUM_HOLE_CARDS
    is_private = deal_idx < NUM_PLAYERS * NUM_HOLE_CARDS
    new_private = state.private_cards.at[p, slot].set(card)
    board_slot = state.board_count
    new_board = state.board_cards.at[board_slot].set(card)

    new_state = state._replace(
        private_cards=jnp.where(is_private, new_private, state.private_cards),
        board_cards=jnp.where(is_private, state.board_cards, new_board),
        deck_mask=new_deck,
        num_dealt=state.num_dealt + 1,
        board_count=jnp.where(
            is_private, state.board_count, state.board_count + 1
        ),
    )
    if self._track_information_state_history:
      new_state = _append_history(new_state, jnp.int32(ord("d")), jnp.int32(0))
    private_done = new_state.num_dealt == NUM_PLAYERS * NUM_HOLE_CARDS
    board_complete = new_state.board_count == MAX_BOARD_CARDS
    need_more_flop = (
        (~is_private) & (state.round == 1) & (new_state.board_count < 3)
    )
    next_chance = need_more_flop | (new_state.all_in & ~board_complete)
    next_player = jnp.where(
        board_complete & new_state.all_in,
        jnp.int32(TERMINAL),
        jnp.where(
            next_chance,
            jnp.int32(game_lib.CHANCE_PLAYER_ID),
            jnp.where(
                private_done | (~is_private),
                jnp.int32(0),
                jnp.int32(game_lib.CHANCE_PLAYER_ID),
            ),
        ),
    )
    # The configured preflop first player is player 1.
    next_player = jnp.where(
        private_done & is_private, jnp.int32(1), next_player
    )
    terminal = board_complete & new_state.all_in
    return new_state._replace(current_player=next_player, is_terminal=terminal)

  def _start_next_round(self, state: GameState) -> GameState:
    """Ends the betting round: the next deal, or terminal after the river.

    Args:
      state: The state.

    Returns:
      The state at the next round.
    """
    next_round = state.round + jnp.int32(1)
    terminal = next_round > jnp.int32(3)
    all_in_runout = jnp.any(state.spent == STACK)
    current = jnp.where(
        terminal,
        jnp.int32(TERMINAL),
        jnp.int32(game_lib.CHANCE_PLAYER_ID),
    )
    return state._replace(
        round=next_round,
        current_player=current,
        round_actions=jnp.int32(0),
        last_raise_size=jnp.int32(BIG_BLIND),
        all_in=state.all_in | all_in_runout,
        is_terminal=terminal,
    )

  def _apply_player_action(
      self, state: GameState, action: jax.Array
  ) -> GameState:
    """Applies a fold, call or raise.

    Args:
      state: The state.
      action: The action.

    Returns:
      The state after the action.
    """
    cur = state.current_player
    opp = 1 - cur
    max_spent = jnp.maximum(state.spent[0], state.spent[1])
    facing_bet = state.spent[cur] < max_spent

    is_fold = action == FOLD
    is_call = action == CALL
    is_raise = action >= MIN_RAISE_ACTION
    target = jnp.asarray(action).astype(jnp.int32)
    call_target = max_spent
    new_commitment = jnp.where(
        is_raise, target, jnp.where(is_call, call_target, state.spent[cur])
    )
    new_spent = state.spent.at[cur].set(new_commitment)
    raise_size = target - max_spent

    if self._track_information_state_history:
      char_code = jnp.where(
          is_raise,
          jnp.int32(ord("p")),
          jnp.where(is_call, jnp.int32(ord("c")), jnp.int32(ord("f"))),
      )
      size = jnp.where(is_raise, target, jnp.int32(0))
      state = _append_history(state, char_code, size)
    updated = state._replace(
        spent=new_spent,
        last_raise_size=jnp.where(is_raise, raise_size, state.last_raise_size),
        round_actions=state.round_actions + 1,
        folded_player=jnp.where(is_fold, cur, state.folded_player),
        all_in=state.all_in | jnp.any(new_spent == STACK),
    )

    preflop_small_blind_call = (
        (state.round == 0)
        & (state.round_actions == 0)
        & (cur == SMALL_BLIND_PLAYER)
    )
    round_over_from_call = is_call & facing_bet & (~preflop_small_blind_call)
    round_over_from_checks = (
        is_call & (~facing_bet) & (state.round_actions >= 1)
    )
    round_over = round_over_from_call | round_over_from_checks
    terminal_from_fold = is_fold

    folded_terminal = updated._replace(
        current_player=jnp.int32(TERMINAL), is_terminal=jnp.bool_(True)
    )
    next_round_state = self._start_next_round(updated)
    continue_state = updated._replace(current_player=opp)
    return jax.tree.map(
        lambda f, r, c: jnp.where(
            terminal_from_fold, f, jnp.where(round_over, r, c)
        ),
        folded_terminal,
        next_round_state,
        continue_state,
    )

  def apply_action(self, state: GameState, action: jax.Array) -> GameState:
    """Returns the state after `action`.

    Args:
      state: The current state.
      action: i32[], a legal player action, or a chance action id with
        nonzero probability at a chance node.

    Returns:
      The next state. Applying an illegal action gives an unspecified state;
      it does not raise, because the method is traced.
    """
    is_chance = state.current_player == game_lib.CHANCE_PLAYER_ID
    chance_state = self._deal_card(state, action)
    player_state = self._apply_player_action(state, action)
    return jax.tree.map(
        lambda c, p: jnp.where(is_chance, c, p), chance_state, player_state
    )

  def chance_outcomes(self, state: GameState) -> jax.Array:
    """Returns the chance distribution at `state`.

    Index i is the probability of chance action id i. At a chance node the
    entries are non-negative and sum to 1; outcomes that cannot occur have
    probability 0. Elsewhere the values are unspecified. The shape is static,
    so the method is safe under `jit` and `vmap`. To enumerate outcomes
    exactly, iterate over the nonzero entries; to simulate, sample an index.

    Games without chance nodes inherit this default, which returns `f32[0]`.

    Args:
      state: The current state.

    Returns:
      f32[max_chance_outcomes]: the chance distribution.

    Raises:
      NotImplementedError: if the game declares chance outcomes but does not
        override this method.
    """
    probs = state.deck_mask.astype(jnp.float32)
    return probs / jnp.maximum(jnp.sum(probs), 1.0)

  def current_player(self, state: GameState) -> jax.Array:
    """Returns the player to move.

    Args:
      state: The current state.

    Returns:
      i32[]: the player to move, or `CHANCE_PLAYER_ID` at a chance node. The
      value at a terminal state is game-specific.
    """
    return state.current_player

  def is_terminal(self, state: GameState) -> jax.Array:
    """Returns whether the game is over.

    Args:
      state: The current state.

    Returns:
      bool[]: True if the game is over.
    """
    return state.is_terminal

  def is_chance_node(self, state: GameState) -> jax.Array:
    """Returns whether the next action is a chance outcome.

    Args:
      state: The current state.

    Returns:
      bool[]: True if the next action is a chance outcome.
    """
    return (
        state.current_player == game_lib.CHANCE_PLAYER_ID
    ) & ~state.is_terminal

  def legal_actions_mask(self, state: GameState) -> jax.Array:
    """Returns the legal player actions.

    Args:
      state: The current state.

    Returns:
      bool[num_actions]: the legal player actions. All False at a terminal state
      and, for most games, at a chance node.
    """
    cur = jnp.maximum(state.current_player, 0)
    max_spent = jnp.maximum(state.spent[0], state.spent[1])
    facing_bet = state.spent[cur] < max_spent
    min_raise_to = max_spent + state.last_raise_size
    # A round holds at most MAX_ROUND_ACTIONS actions, and a raise must leave
    # room for each player to answer it (ACPC's raiseIsValid).
    room_to_raise = state.round_actions + NUM_PLAYERS <= MAX_ROUND_ACTIONS
    can_raise = (min_raise_to <= STACK) & room_to_raise
    actions = jnp.arange(NUM_ACTIONS, dtype=jnp.int32)
    all_in_raise = (actions == STACK) & (STACK > max_spent)
    raise_mask = (
        (actions >= min_raise_to) & (actions <= STACK) & can_raise
    ) | (all_in_raise & room_to_raise)
    player_mask = raise_mask.at[FOLD].set(facing_bet).at[CALL].set(True)
    chance_mask = jnp.concatenate([
        state.deck_mask,
        jnp.zeros(NUM_ACTIONS - NUM_CARDS, dtype=jnp.bool_),
    ])
    mask = jnp.where(
        state.current_player == game_lib.CHANCE_PLAYER_ID,
        chance_mask,
        player_mask,
    )
    return jnp.where(
        state.is_terminal, jnp.zeros(NUM_ACTIONS, dtype=jnp.bool_), mask
    )

  def returns(self, state: GameState) -> jax.Array:
    """Returns each player's return.

    Args:
      state: The current state.

    Returns:
      f32[num_players]: each player's return; zero before the end.
    """
    _check_showdown_cards_complete_if_eager(state)
    p0_folded = state.folded_player == 0
    p1_folded = state.folded_player == 1
    any_folded = p0_folded | p1_folded

    def showdown_outcome():
      """Compares the two players' best hands.

      Returns:
        (player 0 wins, player 1 wins, tie).
      """
      p0_cards = jnp.concatenate([state.private_cards[0], state.board_cards])
      p1_cards = jnp.concatenate([state.private_cards[1], state.board_cards])
      score_fn = (
          best_seven_card_score
          if self._use_showdown_cache
          else best_seven_card_score_uncached
      )
      p0_score = score_fn(p0_cards)
      p1_score = score_fn(p1_cards)
      return p0_score > p1_score, p1_score > p0_score, p0_score == p1_score

    p0_wins_showdown, p1_wins_showdown, draw = jax.lax.cond(
        state.is_terminal & (~any_folded),
        showdown_outcome,
        lambda: (jnp.bool_(False), jnp.bool_(False), jnp.bool_(False)),
    )
    p0_wins = p1_folded | ((~p0_folded) & (~p1_folded) & p0_wins_showdown)
    p1_wins = p0_folded | ((~p0_folded) & (~p1_folded) & p1_wins_showdown)
    pot = jnp.sum(state.spent).astype(jnp.float32)
    p0_share = jnp.where(
        p0_wins,
        pot,
        jnp.where(draw & (~p0_folded) & (~p1_folded), pot / 2.0, 0.0),
    )
    p1_share = jnp.where(
        p1_wins,
        pot,
        jnp.where(draw & (~p0_folded) & (~p1_folded), pot / 2.0, 0.0),
    )
    ret = jnp.array(
        [
            p0_share - state.spent[0].astype(jnp.float32),
            p1_share - state.spent[1].astype(jnp.float32),
        ],
        dtype=jnp.float32,
    )
    return jnp.where(
        state.is_terminal, ret, jnp.zeros(NUM_PLAYERS, dtype=jnp.float32)
    )

  def observation_tensor(
      self, state: GameState, observing_player: jax.Array
  ) -> jax.Array:
    """Returns what `observing_player` observes now.

    Args:
      state: The current state.
      observing_player: i32[], the observing player.

    Returns:
      f32[*obs_shape]: the observation tensor.
    """
    player = jnp.asarray(observing_player).astype(jnp.int32)
    player_oh = jax.nn.one_hot(player, NUM_PLAYERS).astype(jnp.float32)
    private = _cards_one_hot(state.private_cards[player])
    public = _cards_one_hot(state.board_cards)
    spent = state.spent.astype(jnp.float32)
    return jnp.concatenate([player_oh, private, public, spent])

  def information_state_tensor(
      self, state: GameState, observing_player: jax.Array
  ) -> jax.Array:
    """Returns everything `observing_player` has observed so far.

    In a perfect-information game the current observation determines the
    information state, and the game returns its observation tensor.

    Args:
      state: The current state.
      observing_player: i32[], the observing player.

    Returns:
      f32[*info_state_shape]: the information-state tensor.
    """
    self._require_history()
    player = jnp.asarray(observing_player).astype(jnp.int32)
    player_oh = jax.nn.one_hot(player, NUM_PLAYERS).astype(jnp.float32)
    private = _cards_one_hot(state.private_cards[player])
    public = _cards_one_hot(state.board_cards)

    active = jnp.arange(MAX_GAME_LENGTH) < state.history_len
    chars = jnp.where(active, state.action_chars, jnp.int32(0))
    first_bit = ((chars == ord("c")) | (chars == ord("a"))).astype(jnp.float32)
    second_bit = ((chars == ord("p")) | (chars == ord("a"))).astype(jnp.float32)
    action_bits = jnp.stack([first_bit, second_bit], axis=1).reshape(-1)
    sizes = jnp.where(active, state.action_sizes, 0).astype(jnp.float32)
    return jnp.concatenate([player_oh, private, public, action_bits, sizes])

  def observation_from_os(
      self, os_state: "pyspiel.State", player: int
  ) -> jax.Array:
    """Returns the JAX game's observation tensor of `player` at `os_state`.

    Args:
      os_state: The pyspiel state.
      player: The observing player.

    Returns:
      The observation tensor the JAX game gives `player` in the same position.

    Raises:
      NotImplementedError: if the game has no pyspiel conversion.
    """
    return jnp.array(os_state.observation_tensor(player), dtype=jnp.float32)

  def information_state_from_os(
      self, os_state: "pyspiel.State", player: int
  ) -> jax.Array:
    """Returns the JAX game's information-state tensor of `player`.

    Games whose information state is their observation inherit this
    default, which returns `observation_from_os`.

    Args:
      os_state: The pyspiel state.
      player: The observing player.

    Returns:
      The information-state tensor the JAX game gives `player` in the same
      position.
    """
    self._require_history()
    return jnp.array(
        os_state.information_state_tensor(player), dtype=jnp.float32
    )
