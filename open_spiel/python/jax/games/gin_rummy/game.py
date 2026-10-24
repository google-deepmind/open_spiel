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

"""Gin rummy: the state machine and the `GinRummy` game.

The module-level functions (`init_state`, `init_dealt_state`, `step`,
`legal_actions_mask`, `chance_outcomes`, `get_returns`) operate on the state
dict directly; `GinRummy` wraps them in the `Game` interface. The rules, action
ids, deal and scoring are those of OpenSpiel's `gin_rummy`, and the default
observation is OpenSpiel's default observer tensor, with one scoring difference:
when the knocker's deadwood (after layoffs) equals the defender's, this game
scores an undercut (the defender gets the difference, 0, plus the 25-point
undercut bonus), while released OpenSpiel (2.0.2 and earlier) awards the
undercut only when the defender's count is strictly lower and scores the tie 0
for both.
"""

import typing

import jax
import jax.numpy as jnp
import numpy as np

from open_spiel.python.jax.games import game as game_lib
from open_spiel.python.jax.games import interop
from open_spiel.python.jax.games.gin_rummy.cards_and_melds import (
    ACTION_DRAW_STOCK,
)
from open_spiel.python.jax.games.gin_rummy.cards_and_melds import (
    ACTION_DRAW_UPCARD,
)
from open_spiel.python.jax.games.gin_rummy.cards_and_melds import ACTION_KNOCK
from open_spiel.python.jax.games.gin_rummy.cards_and_melds import (
    ACTION_MELD_BASE,
)
from open_spiel.python.jax.games.gin_rummy.cards_and_melds import ACTION_PASS
from open_spiel.python.jax.games.gin_rummy.cards_and_melds import (
    assert_state_dtypes,
)
from open_spiel.python.jax.games.gin_rummy.cards_and_melds import CHANCE_PLAYER
from open_spiel.python.jax.games.gin_rummy.cards_and_melds import (
    get_card_values,
)
from open_spiel.python.jax.games.gin_rummy.cards_and_melds import get_meld_masks
from open_spiel.python.jax.games.gin_rummy.cards_and_melds import GIN_BONUS
from open_spiel.python.jax.games.gin_rummy.cards_and_melds import (
    KNOCK_THRESHOLD,
)
from open_spiel.python.jax.games.gin_rummy.cards_and_melds import (
    MAX_NUM_DRAW_UPCARD_ACTIONS,
)
from open_spiel.python.jax.games.gin_rummy.cards_and_melds import NUM_ACTIONS
from open_spiel.python.jax.games.gin_rummy.cards_and_melds import NUM_CARDS
from open_spiel.python.jax.games.gin_rummy.cards_and_melds import NUM_MELDS
from open_spiel.python.jax.games.gin_rummy.cards_and_melds import PHASE_DEAL
from open_spiel.python.jax.games.gin_rummy.cards_and_melds import PHASE_DISCARD
from open_spiel.python.jax.games.gin_rummy.cards_and_melds import PHASE_DRAW
from open_spiel.python.jax.games.gin_rummy.cards_and_melds import (
    PHASE_FIRST_UPCARD,
)
from open_spiel.python.jax.games.gin_rummy.cards_and_melds import (
    PHASE_GAME_OVER,
)
from open_spiel.python.jax.games.gin_rummy.cards_and_melds import PHASE_KNOCK
from open_spiel.python.jax.games.gin_rummy.cards_and_melds import PHASE_LAYOFF
from open_spiel.python.jax.games.gin_rummy.cards_and_melds import PHASE_WALL
from open_spiel.python.jax.games.gin_rummy.cards_and_melds import (
    TERMINAL_PLAYER,
)
from open_spiel.python.jax.games.gin_rummy.cards_and_melds import UNDERCUT_BONUS
from open_spiel.python.jax.games.gin_rummy.cards_and_melds import (
    WALL_STOCK_SIZE,
)
from open_spiel.python.jax.games.gin_rummy.deadwood import compute_all_meld_info
from open_spiel.python.jax.games.gin_rummy.deadwood import (
    compute_legal_knocker_melds,
)
from open_spiel.python.jax.games.gin_rummy.deadwood import hand_total_points
from open_spiel.python.jax.games.gin_rummy.deadwood import valid_melds_mask
from open_spiel.python.jax.games.gin_rummy.layoffs import compute_layoff_cards
from open_spiel.python.jax.games.gin_rummy.layoffs import init_lazy_constants
from open_spiel.python.jax.games.gin_rummy.layoffs import (
    lazy_constants_initialized,
)

if typing.TYPE_CHECKING:
  import pyspiel


OBSERVATION_TENSOR_SIZE = 644

#: Optional rule variants. The defaults are the rules of OpenSpiel's
#: `gin_rummy` (except the equal-deadwood undercut, module doc); the
#: non-default settings are variants released OpenSpiel does not implement.
#:
#: * `allow_upcard_throwback`: when False, the upcard taken this turn (the
#:   first upcard, a draw-phase take, or the wall knock's pick-up) may not be
#:   discarded, nor be the discard after a knock. A knock is offered only if
#:   the minimum deadwood over the other discards is within the knock card.
#:   The repeated draw-upcard stop then never fires.
#: * `wall_knock_pick` (standard gin only): when False, a non-knock discard
#:   by the player who drew the 50th card ends the hand as a draw
#:   (winner 2, scores 0/0) with no wall phase.
#: * `spade_doubles` (Oklahoma only): a spade initial upcard doubles the hand
#:   score, gin and undercut bonuses included.
DEFAULT_RULES = {
    "allow_upcard_throwback": True,
    "wall_knock_pick": True,
    "spade_doubles": False,
}
RULE_NAMES = tuple(DEFAULT_RULES)

#: Observation planes a rule variant adds, appended after the 644 default
#: floats only when that parameter is not at its default, so a default game's
#: tensor is OpenSpiel's. In this order:
#:
#: * ``allow_upcard_throwback=False``: 52 floats, the one-hot of
#:   ``upcard_taken`` (the upcard taken this turn, which the taker may not
#:   discard); all zero when no card is held back.
#: * ``spade_doubles=True``: 1 float, 1.0 iff ``score_multiplier == 2`` (a
#:   spade initial upcard doubled the stake).
UPCARD_TAKEN_PLANE_SIZE = NUM_CARDS
SCORE_MULTIPLIER_PLANE_SIZE = 1


def observation_tensor_size(
    allow_upcard_throwback=DEFAULT_RULES["allow_upcard_throwback"],
    spade_doubles=DEFAULT_RULES["spade_doubles"],
):
  """Returns the observation size for these rules (644 under the defaults).

  Args:
    allow_upcard_throwback: The rule variant (`DEFAULT_RULES`).
    spade_doubles: The rule variant (`DEFAULT_RULES`).

  Returns:
    The observation tensor size.
  """
  return (
      OBSERVATION_TENSOR_SIZE
      + (0 if allow_upcard_throwback else UPCARD_TAKEN_PLANE_SIZE)
      + (SCORE_MULTIPLIER_PLANE_SIZE if spade_doubles else 0)
  )


# Suit order is "scdh" (card = suit * 13 + rank), so suit 0 is spades.
_SPADE_SUIT = 0


def _is_concrete(x):
  """Whether `x` is a concrete value, not a tracer.

  Args:
    x: A value.

  Returns:
    True outside tracing.
  """
  return not isinstance(x, jax.core.Tracer)


def _check_rules(
    oklahoma, allow_upcard_throwback, wall_knock_pick, spade_doubles, *, where
):
  """Raises on an inconsistent combination of concrete rule values.

  `spade_doubles` is an Oklahoma rule (refused without `oklahoma`) and
  `wall_knock_pick=False` a standard-gin one (refused with `oklahoma`). A
  traced value (vmap over a mixed batch) cannot be checked here; `step` then
  applies the doubling only where `oklahoma` holds and the drawn wall only
  where it does not, so a refused combination plays as the game `oklahoma`
  names with that rule at its default.

  Args:
    oklahoma: The oklahoma rule.
    allow_upcard_throwback: The allow_upcard_throwback rule.
    wall_knock_pick: The wall_knock_pick rule.
    spade_doubles: The spade_doubles rule.
    where: The caller's name, for the error message.

  Raises:
    TypeError: if a concrete rule value is not a bool.
    ValueError: if a concrete combination is inconsistent.
  """
  for name, v in (
      ("allow_upcard_throwback", allow_upcard_throwback),
      ("wall_knock_pick", wall_knock_pick),
      ("spade_doubles", spade_doubles),
  ):
    if _is_concrete(v) and np.asarray(v).dtype != np.bool_:
      raise TypeError(f"{where}: {name}={v!r} is not a bool.")
  if (
      _is_concrete(oklahoma)
      and _is_concrete(spade_doubles)
      and np.any(np.asarray(spade_doubles) & ~np.asarray(oklahoma, dtype=bool))
  ):
    raise ValueError(
        f"{where}: spade_doubles=True requires oklahoma=True (spade "
        "doubling is an Oklahoma rule)."
    )
  if (
      _is_concrete(oklahoma)
      and _is_concrete(wall_knock_pick)
      and np.any(
          ~np.asarray(wall_knock_pick) & np.asarray(oklahoma, dtype=bool)
      )
  ):
    raise ValueError(
        f"{where}: wall_knock_pick=False requires oklahoma=False (Oklahoma "
        "gin always plays the wall)."
    )


__all__ = [
    "DEFAULT_RULES",
    "GinRummy",
    "RULE_NAMES",
    "OBSERVATION_TENSOR_SIZE",
    "SCORE_MULTIPLIER_PLANE_SIZE",
    "UPCARD_TAKEN_PLANE_SIZE",
    "observation_tensor_size",
    "chance_outcomes",
    "chance_outcomes_mask",
    "get_hand",
    "get_returns",
    "init_dealt_state",
    "init_lazy_constants",
    "lazy_constants_initialized",
    "oklahoma_knock_card",
    "deal_deck_order",
    "STOCK_START",
    "assert_state_dtypes",
    "init_state",
    "is_chance_node",
    "legal_actions_mask",
    "legal_actions_mask_common",
    "step",
]


def oklahoma_knock_card(upcard):
  """Returns the knock card the first upcard sets under Oklahoma rules.

  As in OpenSpiel's `gin_rummy(oklahoma=True)`, the upcard's point value
  (face cards 10) becomes the knock card, and an ace means the hand must be
  played for gin (knock card 0).

  Args:
    upcard: The first upcard's card id.

  Returns:
    i8[]: the knock card.
  """
  upcard_value = jnp.asarray(get_card_values(), dtype=jnp.int32)[
      jnp.asarray(upcard, dtype=jnp.int32)
  ]
  return jnp.where(upcard_value == 1, jnp.int32(0), upcard_value).astype(
      jnp.int8
  )


def init_state(
    oklahoma=False,
    allow_upcard_throwback=DEFAULT_RULES["allow_upcard_throwback"],
    wall_knock_pick=DEFAULT_RULES["wall_knock_pick"],
    spade_doubles=DEFAULT_RULES["spade_doubles"],
):
  """Returns the state before the deal; the deal is 21 chance nodes.

  Args:
    oklahoma: If True, the first upcard sets the hand's knock card (an ace
      means gin only). Otherwise the knock card is `KNOCK_THRESHOLD`.
    allow_upcard_throwback: A rule variant (`DEFAULT_RULES`). The rule
      variants are carried in the state, so they may be traced and a batch may
      mix them.
    wall_knock_pick: A rule variant (`DEFAULT_RULES`), carried in the state.
    spade_doubles: A rule variant (`DEFAULT_RULES`), carried in the state.

  Returns:
    The pre-deal state.

  Raises:
    TypeError: on concrete inconsistent rules (`_check_rules`).
    ValueError: on concrete inconsistent rules (`_check_rules`).
  """
  _check_rules(
      oklahoma,
      allow_upcard_throwback,
      wall_knock_pick,
      spade_doubles,
      where="gin_rummy.init_state",
  )
  return _init_state_jit(
      oklahoma, allow_upcard_throwback, wall_knock_pick, spade_doubles
  )


@jax.jit
def _init_state_jit(
    oklahoma, allow_upcard_throwback, wall_knock_pick, spade_doubles
):
  """The state before the deal, for the given rule variants.

  Args:
    oklahoma: Whether Oklahoma rules hold.
    allow_upcard_throwback: The upcard-throwback rule.
    wall_knock_pick: The wall-knock-pick rule.
    spade_doubles: The spade-doubles rule.

  Returns:
    The initial state dict.
  """
  return {
      # Card locations
      "player0_hand": jnp.zeros(NUM_CARDS, dtype=jnp.int8),
      "player1_hand": jnp.zeros(NUM_CARDS, dtype=jnp.int8),
      # Public knowledge, as OpenSpiel's known_cards_: known_cards[p][c] == 1
      # iff card c is known to be in p's hand (p took it from the discard
      # pile and has not shed it since).
      "known_cards": jnp.zeros((2, NUM_CARDS), dtype=jnp.int8),
      "deck": jnp.ones(
          NUM_CARDS, dtype=jnp.int8
      ),  # All cards in deck initially
      "discard_pile": jnp.zeros(NUM_CARDS, dtype=jnp.int8),
      "upcard": jnp.int8(-1),  # -1 = no upcard
      # Game state
      "current_player": jnp.int8(CHANCE_PLAYER),  # Start with chance node
      "phase": jnp.int8(PHASE_DEAL),
      "oklahoma": jnp.asarray(oklahoma, dtype=jnp.bool_),
      # The rule variants (DEFAULT_RULES); fixed for the hand.
      "allow_upcard_throwback": jnp.asarray(
          allow_upcard_throwback, dtype=jnp.bool_
      ),
      "wall_knock_pick": jnp.asarray(wall_knock_pick, dtype=jnp.bool_),
      "spade_doubles": jnp.asarray(spade_doubles, dtype=jnp.bool_),
      # 2 when spade_doubles and oklahoma hold and the initial upcard is a
      # spade (set when it is dealt), else 1. Multiplies the hand score.
      "score_multiplier": jnp.int8(1),
      # The upcard the current player took this turn and so may not discard
      # (-1 = none). Set only when allow_upcard_throwback is False; cleared
      # by the next discard.
      "upcard_taken": jnp.int8(-1),
      # Under Oklahoma rules this is overwritten when the first upcard is
      # dealt; until then (and always in standard play) it is 10.
      "knock_card": jnp.int8(KNOCK_THRESHOLD),
      "deal_cards_dealt": jnp.int8(0),  # 0-20 during deal, then 21 = done
      "prev_player": jnp.int8(
          -1
      ),  # Player who requested stock draw (-1 = none/initial deal)
      # Tracking
      "pass_count": jnp.int8(0),
      "pass_on_first_upcard": jnp.zeros(
          2, dtype=jnp.bool_
      ),  # [player] = True if passed
      "drawn_from_discard": jnp.int8(-1),
      "num_draw_upcard": jnp.int32(0),
      "prev_upcard": jnp.int8(-1),  # Upcard before current player's discard
      "repeated_move": jnp.bool_(False),  # For repeated move termination
      # Knock/layoff state
      "knocked": jnp.zeros(2, dtype=jnp.bool_),
      "knocker": jnp.int8(-1),
      "layed_melds": jnp.zeros((2, NUM_MELDS), dtype=jnp.bool_),
      "layoffs_mask": jnp.zeros(NUM_CARDS, dtype=jnp.bool_),
      "finished_layoffs": jnp.bool_(False),
      "deadwood": jnp.zeros(2, dtype=jnp.int32),
      "knocker_deadwood": jnp.int32(
          -1
      ),  # Knocker's deadwood when entering layoff (-1 = not set)
      # Game outcome
      "done": jnp.bool_(False),
      "winner": jnp.int8(-1),
      "p0_score": jnp.int32(0),
      "p1_score": jnp.int32(0),
      # Meld caches, recomputed by step when the phase needs them.
      "cached_min_deadwood": jnp.int32(999),
      "cached_legal_knocker_melds": jnp.zeros(NUM_MELDS, dtype=jnp.bool_),
      "cached_legal_discard_mask": jnp.zeros(NUM_CARDS, dtype=jnp.bool_),
      "cached_all_discard_deadwoods": jnp.full(NUM_CARDS, 999, dtype=jnp.int16),
      "can_knock": jnp.bool_(False),
      # A derived public feature, not a rule: each player's minimum deadwood
      # at its last discard (999 before one). Policies may read it (the
      # simple bot's draw decision does); the rules never do.
      "player_deadwood": jnp.full(2, 999, dtype=jnp.int16),
  }


@jax.jit
def is_chance_node(state):
  """Returns whether `state` is a chance node.

  Args:
    state: The state.

  Returns:
    bool[]: True at a deal or stock-draw chance node.
  """
  return state["phase"] == PHASE_DEAL


@jax.jit
def get_hand(state, player):
  """Returns the hand of `player` as a card mask.

  Args:
    state: The state.
    player: The player.

  Returns:
    i8[52]: the card mask.
  """
  return jnp.where(player == 0, state["player0_hand"], state["player1_hand"])


#: Index of the first stock card in :func:`deal_deck_order`: 10 cards to each
#: player, then the upcard, then the stock.
STOCK_START = 21


def deal_deck_order(key):
  """Returns the shuffled deck `init_dealt_state(key, ...)` deals.

  ``[:10]`` is player 0's hand, ``[10:20]`` player 1's, ``[20]`` the upcard and
  ``[STOCK_START:]`` the stock, top first. ``init_dealt_state`` computes its
  deal through this function, so the two cannot drift apart.

  Args:
    key: The deal's PRNG key.

  Returns:
    i32[52]: the deck order.
  """
  _, shuffle_key, _ = jax.random.split(key, 3)
  return jax.random.permutation(shuffle_key, 52)


#: fold_in salt of the knock-limit draw's stream (folded into the first-player
#: key).
_KNOCK_LIMIT_KEY_SALT = 0xC0FFEE


def _init_dealt_state_body(
    key,
    oklahoma=False,
    randomized_knock_limit=False,
    allow_upcard_throwback=DEFAULT_RULES["allow_upcard_throwback"],
    wall_knock_pick=DEFAULT_RULES["wall_knock_pick"],
    spade_doubles=DEFAULT_RULES["spade_doubles"],
    meld_info=compute_all_meld_info,
):
  """Returns a dealt state, skipping the 21 deal chance nodes.

  The state is at PHASE_FIRST_UPCARD with its meld caches filled.

  Args:
      key: JAX PRNG key
      oklahoma: if True (may be a traced bool, so mixed batches work under
        vmap), the hand gets a non-standard knock card instead of the fixed
        ``KNOCK_THRESHOLD``. By default that knock card is derived from the
        dealt upcard (standard Oklahoma: ace => gin-only); with
        ``randomized_knock_limit`` it is instead sampled uniformly.
      randomized_knock_limit: a sampling option, not a rule of OpenSpiel's
        gin_rummy. If True (may be a traced bool), an Oklahoma hand's knock
        card is sampled uniformly from ``{0, ..., KNOCK_THRESHOLD}``
        independently of the upcard, so every knock limit is equally likely
        (the upcard's point value makes 10 four times as likely as the others:
        T, J, Q and K all give 10). False is standard Oklahoma. Has no effect
        on non-Oklahoma deals (knock card stays at ``KNOCK_THRESHOLD``).
      allow_upcard_throwback: a rule variant (`DEFAULT_RULES`; may be
        traced). The rule variants do not consume randomness, so the deal is
        the same for every rules setting.
      wall_knock_pick: a rule variant (`DEFAULT_RULES`; may be traced).
      spade_doubles: a rule variant (`DEFAULT_RULES`; may be traced).
      meld_info: the function computing the meld caches; the backend's.

  Returns:
      (state, new_key).
  """
  deck_order = deal_deck_order(key)
  key, _, player_key = jax.random.split(key, 3)
  # The deck order, first player and returned key do not depend on
  # randomized_knock_limit: the knock limit draws from its own stream.
  knock_key = jax.random.fold_in(player_key, _KNOCK_LIMIT_KEY_SALT)

  # Allocate cards from the shuffled deck (deal_deck_order, above)
  p0_cards = deck_order[:10]
  p1_cards = deck_order[10:20]
  upcard = deck_order[20]
  stock_cards = deck_order[STOCK_START:]

  # Build hand masks
  p0_hand = jnp.zeros(52, dtype=jnp.int8).at[p0_cards].set(1)
  p1_hand = jnp.zeros(52, dtype=jnp.int8).at[p1_cards].set(1)

  # Build deck mask (remaining stock)
  deck_mask = jnp.zeros(52, dtype=jnp.int8).at[stock_cards].set(1)

  # Random starting player (0 or 1)
  first_player = jax.random.randint(player_key, (), 0, 2, dtype=jnp.int8)

  oklahoma = jnp.asarray(oklahoma, dtype=jnp.bool_)
  randomized_knock_limit = jnp.asarray(randomized_knock_limit, dtype=jnp.bool_)
  # Randomized knock limit: sample uniformly over {0..KNOCK_THRESHOLD}
  # independently of the upcard. randint's high bound is exclusive, so
  # KNOCK_THRESHOLD + 1 includes the limit-10 (standard) value.
  random_knock_card = jax.random.randint(
      knock_key, (), 0, KNOCK_THRESHOLD + 1, dtype=jnp.int32
  ).astype(jnp.int8)
  # Oklahoma deals take a non-standard knock card: the uniform sample under
  # randomized_knock_limit, else the upcard's value.
  oklahoma_limit = jnp.where(
      randomized_knock_limit,
      random_knock_card,
      oklahoma_knock_card(upcard),
  )
  knock_card = jnp.where(oklahoma, oklahoma_limit, jnp.int8(KNOCK_THRESHOLD))
  allow_upcard_throwback = jnp.asarray(allow_upcard_throwback, dtype=jnp.bool_)
  wall_knock_pick = jnp.asarray(wall_knock_pick, dtype=jnp.bool_)
  spade_doubles = jnp.asarray(spade_doubles, dtype=jnp.bool_)
  score_multiplier = jnp.where(
      oklahoma & spade_doubles & (upcard // 13 == _SPADE_SUIT),
      jnp.int8(2),
      jnp.int8(1),
  )

  # The same keys and dtypes as init_state().
  state = {
      "player0_hand": p0_hand,
      "player1_hand": p1_hand,
      # Public knowledge; see init_state().
      "known_cards": jnp.zeros((2, NUM_CARDS), dtype=jnp.int8),
      "deck": deck_mask,
      "upcard": upcard.astype(jnp.int8),
      "prev_upcard": jnp.int8(-1),
      "discard_pile": jnp.zeros(52, dtype=jnp.int8),
      "phase": jnp.int8(PHASE_FIRST_UPCARD),
      "current_player": first_player,
      "oklahoma": oklahoma,
      "allow_upcard_throwback": allow_upcard_throwback,
      "wall_knock_pick": wall_knock_pick,
      "spade_doubles": spade_doubles,
      "score_multiplier": score_multiplier,
      "upcard_taken": jnp.int8(-1),
      "knock_card": knock_card,
      "done": jnp.bool_(False),
      "deal_cards_dealt": jnp.int8(21),
      "pass_count": jnp.int8(0),
      "pass_on_first_upcard": jnp.zeros(2, dtype=jnp.bool_),
      "drawn_from_discard": jnp.int8(-1),
      "prev_player": jnp.int8(-1),
      "num_draw_upcard": jnp.int32(0),
      "knocker": jnp.int8(-1),
      "knocker_deadwood": jnp.int32(-1),
      "finished_layoffs": jnp.bool_(False),
      "layed_melds": jnp.zeros((2, NUM_MELDS), dtype=jnp.bool_),
      "p0_score": jnp.int32(0),
      "p1_score": jnp.int32(0),
      "winner": jnp.int8(-1),
      "knocked": jnp.zeros(2, dtype=jnp.bool_),
      "deadwood": jnp.zeros(2, dtype=jnp.int32),
      "player_deadwood": jnp.full(2, 999, dtype=jnp.int16),
      "layoffs_mask": jnp.zeros(52, dtype=jnp.bool_),
      "repeated_move": jnp.bool_(False),
      # Meld cache fields - initialized below
      "cached_min_deadwood": jnp.int32(999),
      "cached_legal_knocker_melds": jnp.zeros(NUM_MELDS, dtype=jnp.bool_),
      "cached_legal_discard_mask": jnp.zeros(NUM_CARDS, dtype=jnp.bool_),
      "cached_all_discard_deadwoods": jnp.full(NUM_CARDS, 999, dtype=jnp.int16),
      "can_knock": jnp.bool_(False),
  }

  # Compute initial meld cache for first player's hand + upcard
  first_hand = jnp.where(first_player == 0, p0_hand, p1_hand)
  effective_hand = first_hand.at[upcard].set(
      1
  )  # Include upcard for FIRST_UPCARD decision
  _, all_deadwoods = meld_info(effective_hand)
  # Under allow_upcard_throwback=False a taken upcard cannot be the discard,
  # so the "take it" evaluation minimises over the other cards.
  hand_mask = (effective_hand > 0) & (
      allow_upcard_throwback | (jnp.arange(NUM_CARDS) != upcard)
  )
  min_deadwood = jnp.min(jnp.where(hand_mask, all_deadwoods, jnp.int32(999)))

  state["cached_min_deadwood"] = jnp.int32(min_deadwood)
  state["cached_all_discard_deadwoods"] = all_deadwoods.astype(jnp.int16)
  state["can_knock"] = min_deadwood <= knock_card.astype(jnp.int32)

  return state, key


_init_dealt_state_jit = jax.jit(_init_dealt_state_body)


def init_dealt_state(
    key,
    oklahoma=False,
    randomized_knock_limit=False,
    allow_upcard_throwback=DEFAULT_RULES["allow_upcard_throwback"],
    wall_knock_pick=DEFAULT_RULES["wall_knock_pick"],
    spade_doubles=DEFAULT_RULES["spade_doubles"],
):
  """Returns a dealt state; see `_init_dealt_state_body`.

  Args:
    key: The PRNG key.
    oklahoma: Whether the hand gets an Oklahoma knock card.
    randomized_knock_limit: Whether an Oklahoma knock card is sampled
      uniformly instead of set by the upcard.
    allow_upcard_throwback: A rule variant (`DEFAULT_RULES`).
    wall_knock_pick: A rule variant (`DEFAULT_RULES`).
    spade_doubles: A rule variant (`DEFAULT_RULES`).

  Returns:
    (state, new_key).

  Raises:
    TypeError: on concrete inconsistent rules.
    ValueError: on concrete inconsistent rules.
  """
  _check_rules(
      oklahoma,
      allow_upcard_throwback,
      wall_knock_pick,
      spade_doubles,
      where="gin_rummy.init_dealt_state",
  )
  return _init_dealt_state_jit(
      key,
      oklahoma,
      randomized_knock_limit,
      allow_upcard_throwback,
      wall_knock_pick,
      spade_doubles,
  )


# =============================================================================
# Legal actions
# =============================================================================


@jax.jit
def chance_outcomes_mask(state):
  """Returns the cards still in the deck.

  Args:
    state: The state.

  Returns:
    bool[52]: True for each card in the deck.
  """
  return state["deck"] > 0


def chance_outcomes(state):
  """Returns the chance distribution: uniform over the cards in the deck.

  Args:
    state: The state.

  Returns:
    f32[52], indexed by card id (the chance action); all zero when the deck is
    empty.
  """
  probs = chance_outcomes_mask(state).astype(jnp.float32)
  return probs / jnp.maximum(jnp.sum(probs), 1.0)


def _common_phases_mask(state):
  """The legal-action mask of chance nodes and the common phases.

  The common phases are FIRST_UPCARD, DRAW, WALL and DISCARD.

  Args:
    state: The state.

  Returns:
    The mask and what the KNOCK and LAYOFF phases read: (mask, phase,
    is_chance, hand, card_in_hand, legal_discard_mask).
  """
  phase = state["phase"]
  player = state["current_player"]

  # Initialize mask
  mask = jnp.zeros(NUM_ACTIONS, dtype=jnp.bool_)

  # === Chance nodes: legal actions are cards in deck ===
  is_chance = is_chance_node(state)
  deck_cards = state["deck"] > 0
  mask = jnp.where(is_chance, mask.at[:NUM_CARDS].set(deck_cards), mask)

  # For non-chance nodes:
  hand = get_hand(state, player)
  card_in_hand = hand > 0

  # Meld information is cached by step().
  legal_discard_mask = state["cached_legal_discard_mask"]
  can_knock = state["can_knock"]

  # Stock info
  stock_count = jnp.sum(state["deck"])
  is_wall = stock_count <= WALL_STOCK_SIZE
  has_upcard = state["upcard"] >= 0

  # === Phase: FirstUpcard ===
  first_upcard = (phase == PHASE_FIRST_UPCARD) & ~is_chance
  both_passed = state["pass_count"] >= 2
  # Before both pass: draw_upcard (if upcard exists) or pass
  mask = jnp.where(
      first_upcard & has_upcard & ~both_passed,
      mask.at[ACTION_DRAW_UPCARD].set(True),
      mask,
  )
  mask = jnp.where(
      first_upcard & ~both_passed, mask.at[ACTION_PASS].set(True), mask
  )
  # After both players pass on the first upcard, only a stock draw is legal.
  mask = jnp.where(
      first_upcard & both_passed, mask.at[ACTION_DRAW_STOCK].set(True), mask
  )

  # === Phase: Draw ===
  draw_phase = (phase == PHASE_DRAW) & ~is_chance
  can_draw_upcard = draw_phase & ~is_wall & has_upcard
  can_draw_stock = draw_phase & ~is_wall & (stock_count > 0)
  mask = jnp.where(can_draw_upcard, mask.at[ACTION_DRAW_UPCARD].set(True), mask)
  mask = jnp.where(can_draw_stock, mask.at[ACTION_DRAW_STOCK].set(True), mask)

  # === Phase: Wall ===
  wall_phase = (phase == PHASE_WALL) & ~is_chance
  mask = jnp.where(wall_phase, mask.at[ACTION_PASS].set(True), mask)

  # === Phase: Discard ===
  discard_phase = (phase == PHASE_DISCARD) & ~is_chance

  # === Knock eligibility (from cached can_knock) ===
  # WALL: can_knock was computed for hand+upcard in step()
  # DISCARD: can_knock was computed for 11-card hand in step()
  can_knock_wall = wall_phase & can_knock
  can_knock_discard = discard_phase & can_knock
  mask = jnp.where(can_knock_wall, mask.at[ACTION_KNOCK].set(True), mask)

  # Any card in hand may be discarded except the upcard taken this turn,
  # which is set only under allow_upcard_throwback=False.
  not_taken = jnp.arange(NUM_CARDS) != state["upcard_taken"]
  mask = jnp.where(
      discard_phase, mask.at[:NUM_CARDS].set(card_in_hand & not_taken), mask
  )
  mask = jnp.where(can_knock_discard, mask.at[ACTION_KNOCK].set(True), mask)

  return mask, phase, is_chance, hand, card_in_hand, legal_discard_mask


@jax.jit
def legal_actions_mask(state):
  """Returns the legal actions (cards at a chance node).

  Args:
    state: The state.

  Returns:
    bool[NUM_ACTIONS]: the legal actions.
  """
  mask, phase, is_chance, hand, card_in_hand, legal_discard_mask = (
      _common_phases_mask(state)
  )

  # === Phase: Knock ===
  knock_phase = (phase == PHASE_KNOCK) & ~is_chance
  hand_count = jnp.sum(hand)
  has_11_cards = hand_count == 11

  # With 11 cards: can only discard cards that keep deadwood <= knock_card
  # Use legal_discard_mask from unified meld computation (already includes hand
  # > 0 check)
  mask = jnp.where(
      knock_phase & has_11_cards,
      mask.at[:NUM_CARDS].set(legal_discard_mask),
      mask,
  )

  # With 10 or fewer cards: can lay melds that preserve knock
  # (deadwood <= knock_card). A meld is legal iff laying it still allows
  # achieving deadwood <= knock_card (cards are removed from hand when
  # melds are laid, remaining cards are all deadwood)
  knock_10 = knock_phase & (hand_count <= 10)
  # Read legal_knocker_melds from cache (computed in step())
  legal_knocker_melds = state["cached_legal_knocker_melds"]
  meld_mask = knock_10 & legal_knocker_melds
  # Calculate raw point total of remaining cards (no meld optimization - these
  # ARE the deadwood)
  remaining_points = hand_total_points(hand)
  can_pass_knock = remaining_points <= state["knock_card"].astype(jnp.int32)
  # Allow pass if remaining points are low enough
  mask = jnp.where(
      knock_10 & can_pass_knock, mask.at[ACTION_PASS].set(True), mask
  )
  mask = jnp.where(
      knock_10,
      mask.at[ACTION_MELD_BASE : ACTION_MELD_BASE + NUM_MELDS].set(meld_mask),
      mask,
  )

  # === Phase: Layoff ===
  layoff_phase = (phase == PHASE_LAYOFF) & ~is_chance
  finished = state["finished_layoffs"]
  knocker = state["knocker"]
  safe_knocker = jnp.where(knocker >= 0, knocker, 0)
  knocker_melds = state["layed_melds"][safe_knocker]

  # If knocker has gin (0 deadwood), opponent can't lay off - skip to meld phase
  # Use the scalar knocker_deadwood field set when entering layoff phase
  knocker_deadwood = state["knocker_deadwood"]
  is_gin = knocker_deadwood == 0
  finished = finished | is_gin  # Treat gin as if layoffs are finished

  layoff_possible = compute_layoff_cards(knocker_melds, state["layoffs_mask"])

  # Before finishing layoffs: can lay off cards or pass
  pre_finish = layoff_phase & ~finished
  layoff_card_mask = card_in_hand & layoff_possible
  mask = jnp.where(pre_finish, mask.at[:NUM_CARDS].set(layoff_card_mask), mask)
  mask = jnp.where(pre_finish, mask.at[ACTION_PASS].set(True), mask)

  # After finishing layoffs: can lay any valid meld from hand or pass
  post_finish = layoff_phase & finished
  mask = jnp.where(post_finish, mask.at[ACTION_PASS].set(True), mask)
  all_valid_melds_layoff = valid_melds_mask(hand)
  meld_mask_layoff = post_finish & all_valid_melds_layoff
  mask = jnp.where(
      post_finish,
      mask.at[ACTION_MELD_BASE : ACTION_MELD_BASE + NUM_MELDS].set(
          meld_mask_layoff
      ),
      mask,
  )

  return mask


@jax.jit
def legal_actions_mask_common(state):
  """Returns the legal actions in the FIRST_UPCARD, DRAW, DISCARD and WALL.

  Those are the phases without a meld computation.

  It skips the meld computation of the KNOCK and LAYOFF phases (the batched
  dense-cache backend's fast path). In the KNOCK and LAYOFF phases the mask
  lacks the meld actions; use `legal_actions_mask` there.

  Args:
    state: The state.

  Returns:
    bool[NUM_ACTIONS]: the legal actions.
  """
  return _common_phases_mask(state)[0]


# =============================================================================
# State transitions
# =============================================================================


def _step_body(
    state,
    action,
    meld_info=compute_all_meld_info,
    knocker_melds=compute_legal_knocker_melds,
):
  """Returns the state after `action`.

  Args:
    state: The current state.
    action: A legal action, or a card id at a chance node.
    meld_info: Computes (optimal melds, deadwood per discard) of a hand.
    knocker_melds: Computes the melds a knocker may lay under a knock limit.
      The backend chooses both; every backend gives the same results.

  Returns:
    The next state.
  """
  meld_masks = jnp.asarray(get_meld_masks())

  phase = state["phase"]
  player = state["current_player"]
  is_chance = is_chance_node(state)

  # jnp.where evaluates both branches, so card-indexed reads and writes use a
  # clipped action: non-card actions (PASS=54, KNOCK=55, melds >= 56) would
  # index out of range even where the predicate is False. Predicates use the
  # raw action.
  safe_card_action = jnp.clip(action, 0, NUM_CARDS - 1)

  # Copy all state fields
  new_state = {k: v for k, v in state.items()}

  # === Handle chance nodes ===

  # Deal phase: action is which card to deal
  in_deal = (phase == PHASE_DEAL) & is_chance
  deal_count = state["deal_cards_dealt"]

  # First 10 cards go to player 0
  to_p0 = in_deal & (deal_count < 10)
  new_p0_hand = jnp.where(
      to_p0,
      state["player0_hand"].at[safe_card_action].set(1),
      state["player0_hand"],
  )

  # Next 10 cards go to player 1
  to_p1 = in_deal & (deal_count >= 10) & (deal_count < 20)
  new_p1_hand = jnp.where(
      to_p1,
      state["player1_hand"].at[safe_card_action].set(1),
      state["player1_hand"],
  )

  # Card 21 is upcard
  to_upcard = in_deal & (deal_count == 20)
  new_upcard = jnp.where(to_upcard, jnp.int8(action), state["upcard"])

  # Under Oklahoma rules the initial upcard sets the knock card (an ace
  # means gin only).
  new_knock_card = jnp.where(
      to_upcard & state["oklahoma"],
      oklahoma_knock_card(safe_card_action),
      state["knock_card"],
  )
  # Under spade_doubles a spade initial upcard doubles the hand (Oklahoma).
  new_state["score_multiplier"] = jnp.where(
      to_upcard
      & state["oklahoma"]
      & state["spade_doubles"]
      & (safe_card_action // 13 == _SPADE_SUIT),
      jnp.int8(2),
      state["score_multiplier"],
  )

  # Remove card from deck
  new_deck = jnp.where(
      in_deal, state["deck"].at[safe_card_action].set(0), state["deck"]
  )

  # Update deal count
  new_deal_count = jnp.where(in_deal, deal_count + 1, deal_count)

  # Check if dealing is complete (21 cards dealt)
  deal_complete = in_deal & (new_deal_count >= 21)
  new_phase = jnp.where(deal_complete, jnp.int8(PHASE_FIRST_UPCARD), phase)
  new_player = jnp.where(deal_complete, jnp.int8(0), player)
  # prev_upcard stays as -1 (None) at deal complete - it tracks previous
  # discard, not initial upcard
  new_prev_upcard = state["prev_upcard"]

  # Keep as chance player during deal
  new_player = jnp.where(
      in_deal & ~deal_complete, jnp.int8(CHANCE_PLAYER), new_player
  )

  # Stock draw chance node: action is which card player gets
  # Detected by prev_player >= 0 (set when entering stock draw)
  stock_draw = (state["prev_player"] >= 0) & is_chance
  stock_player = state["prev_player"]  # Player who requested the draw
  hand = get_hand(state, stock_player)
  drawn_hand = hand.at[safe_card_action].set(1)
  new_p0_hand = jnp.where(
      stock_draw & (stock_player == 0), drawn_hand, new_p0_hand
  )
  new_p1_hand = jnp.where(
      stock_draw & (stock_player == 1), drawn_hand, new_p1_hand
  )
  new_deck = jnp.where(
      stock_draw, new_deck.at[safe_card_action].set(0), new_deck
  )
  new_phase = jnp.where(stock_draw, jnp.int8(PHASE_DISCARD), new_phase)
  # Restore current_player to the player who drew (they need to discard)
  new_player = jnp.where(stock_draw, stock_player, new_player)
  # Reset prev_player after stock draw resolves (so we can detect next stock
  # draw)
  new_prev_player = jnp.where(stock_draw, jnp.int8(-1), state["prev_player"])
  # After stock draw, drawn_from_discard = -1 (can discard anything)
  new_drawn_from_discard = jnp.where(
      stock_draw, jnp.int8(-1), state["drawn_from_discard"]
  )

  # Update state with chance node changes
  new_state["player0_hand"] = new_p0_hand
  new_state["player1_hand"] = new_p1_hand
  new_state["deck"] = new_deck
  new_state["upcard"] = new_upcard
  new_state["knock_card"] = new_knock_card
  # discard_pile unchanged during chance node (upcard handling done in Draw
  # phase)
  new_state["deal_cards_dealt"] = new_deal_count
  new_state["phase"] = new_phase
  new_state["current_player"] = new_player
  new_state["drawn_from_discard"] = new_drawn_from_discard
  new_state["prev_upcard"] = new_prev_upcard
  new_state["prev_player"] = new_prev_player

  # === Handle regular (non-chance) actions ===
  not_chance = ~is_chance

  # --- FirstUpcard phase ---
  first_upcard = (phase == PHASE_FIRST_UPCARD) & not_chance

  # Pass in first upcard
  pass_first = first_upcard & (action == ACTION_PASS)
  new_pass_count = jnp.where(
      pass_first, state["pass_count"] + 1, state["pass_count"]
  )
  # Record which players passed on the first upcard (kept for the hand).
  new_pass_on_first = jnp.where(
      pass_first,
      state["pass_on_first_upcard"].at[player].set(True),
      state["pass_on_first_upcard"],
  )
  both_passed = new_pass_count >= 2
  # After both pass the phase stays FirstUpcard, as in OpenSpiel. The first
  # stock draw goes to the first player to act, the first passer, which is
  # this second passer's opponent (1 - player); init_dealt_state may deal
  # either player first. The upcard moves to the discard pile on the draw.
  new_phase = new_state["phase"]  # Stay in FirstUpcard
  new_player = jnp.where(
      pass_first & both_passed,
      jnp.int8(1 - player),
      new_state["current_player"],
  )
  new_upcard_after_pass = new_state["upcard"]  # Keep upcard
  new_discard = state["discard_pile"]  # Don't modify discard pile
  # Switch player if only first pass
  new_player = jnp.where(
      pass_first & ~both_passed, jnp.int8(1 - player), new_player
  )

  # Draw stock in FirstUpcard (only legal after both passed)
  already_both_passed = state["pass_count"] >= 2
  draw_stock_first = (
      first_upcard & (action == ACTION_DRAW_STOCK) & already_both_passed
  )
  # Move upcard to discard, set prev_upcard, enter chance node
  upcard_val = state["upcard"]
  new_discard = jnp.where(
      draw_stock_first & (upcard_val >= 0),
      new_discard.at[upcard_val].set(1),
      new_discard,
  )
  new_prev_upcard = jnp.where(
      draw_stock_first & (upcard_val >= 0), upcard_val, new_prev_upcard
  )
  new_upcard_after_pass = jnp.where(
      draw_stock_first, jnp.int8(-1), new_upcard_after_pass
  )
  new_phase = jnp.where(draw_stock_first, jnp.int8(PHASE_DEAL), new_phase)
  new_player = jnp.where(draw_stock_first, jnp.int8(CHANCE_PLAYER), new_player)
  # Set prev_player to identify stock draw chance node
  new_prev_player = jnp.where(draw_stock_first, player, state["prev_player"])

  # Draw upcard in first upcard
  draw_first = first_upcard & (action == ACTION_DRAW_UPCARD)
  upcard_card = state["upcard"]
  hand = get_hand(state, player)
  hand_with_upcard = hand.at[upcard_card].set(1)
  new_p0_hand_draw = jnp.where(
      draw_first & (player == 0), hand_with_upcard, new_state["player0_hand"]
  )
  new_p1_hand_draw = jnp.where(
      draw_first & (player == 1), hand_with_upcard, new_state["player1_hand"]
  )
  new_upcard_draw = jnp.where(draw_first, jnp.int8(-1), new_upcard_after_pass)
  new_phase_draw = jnp.where(draw_first, jnp.int8(PHASE_DISCARD), new_phase)
  new_drawn_from_discard = jnp.where(
      draw_first, jnp.int8(upcard_card), new_state["drawn_from_discard"]
  )
  no_throwback = ~state["allow_upcard_throwback"]
  new_state["upcard_taken"] = jnp.where(
      draw_first & no_throwback, upcard_card, state["upcard_taken"]
  )
  # Set prev_upcard to the card drawn (for repeated move detection)
  new_prev_upcard = jnp.where(draw_first, upcard_card, new_prev_upcard)
  # Only Draw-phase draws count toward the draw-upcard limit, as in
  # OpenSpiel.

  new_state["pass_count"] = new_pass_count
  new_state["pass_on_first_upcard"] = new_pass_on_first
  new_state["phase"] = jnp.where(draw_first, new_phase_draw, new_phase)
  new_state["current_player"] = new_player
  new_state["upcard"] = jnp.where(
      draw_first, new_upcard_draw, new_upcard_after_pass
  )
  new_state["discard_pile"] = new_discard
  new_state["player0_hand"] = new_p0_hand_draw
  new_state["player1_hand"] = new_p1_hand_draw
  new_state["drawn_from_discard"] = new_drawn_from_discard
  # num_draw_upcard counts only Draw-phase draws.
  new_state["prev_upcard"] = new_prev_upcard
  new_state["prev_player"] = new_prev_player

  # --- Draw phase ---
  draw_phase = (phase == PHASE_DRAW) & not_chance

  # Draw from stock -> becomes chance node
  draw_stock = draw_phase & (action == ACTION_DRAW_STOCK)
  # A stock draw first moves the upcard to the discard pile and records it
  # as prev_upcard.
  upcard_stock = state["upcard"]
  has_upcard_stock = upcard_stock >= 0
  new_state["discard_pile"] = jnp.where(
      draw_stock & has_upcard_stock,
      new_state["discard_pile"].at[upcard_stock].set(1),
      new_state["discard_pile"],
  )
  new_state["upcard"] = jnp.where(draw_stock, jnp.int8(-1), new_state["upcard"])
  new_state["prev_upcard"] = jnp.where(
      draw_stock & has_upcard_stock, upcard_stock, new_state["prev_upcard"]
  )
  # Set up chance node: prev_player identifies who gets the card
  new_state["prev_player"] = jnp.where(
      draw_stock, player, new_state["prev_player"]
  )
  new_state["current_player"] = jnp.where(
      draw_stock, jnp.int8(CHANCE_PLAYER), new_state["current_player"]
  )
  # The drawn card is a chance outcome: enter the Deal phase.
  new_state["phase"] = jnp.where(
      draw_stock, jnp.int8(PHASE_DEAL), new_state["phase"]
  )

  # Draw upcard in draw phase
  draw_up = draw_phase & (action == ACTION_DRAW_UPCARD)
  new_num_upcard = jnp.where(
      draw_up, new_state["num_draw_upcard"] + 1, new_state["num_draw_upcard"]
  )
  too_many = new_num_upcard >= MAX_NUM_DRAW_UPCARD_ACTIONS

  # If too many upcard draws, game ends
  new_state["done"] = jnp.where(draw_up & too_many, True, new_state["done"])
  new_state["winner"] = jnp.where(
      draw_up & too_many, jnp.int8(2), new_state["winner"]
  )
  new_state["phase"] = jnp.where(
      draw_up & too_many, jnp.int8(PHASE_GAME_OVER), new_state["phase"]
  )

  # Otherwise draw the upcard
  draw_up_ok = draw_up & ~too_many
  upcard_d = state["upcard"]
  hand_d = get_hand(state, player)
  hand_with_up = hand_d.at[upcard_d].set(1)
  new_state["player0_hand"] = jnp.where(
      draw_up_ok & (player == 0), hand_with_up, new_state["player0_hand"]
  )
  new_state["player1_hand"] = jnp.where(
      draw_up_ok & (player == 1), hand_with_up, new_state["player1_hand"]
  )
  new_state["upcard"] = jnp.where(draw_up_ok, jnp.int8(-1), new_state["upcard"])
  new_state["phase"] = jnp.where(
      draw_up_ok, jnp.int8(PHASE_DISCARD), new_state["phase"]
  )
  new_state["drawn_from_discard"] = jnp.where(
      draw_up_ok, jnp.int8(upcard_d), new_state["drawn_from_discard"]
  )
  new_state["upcard_taken"] = jnp.where(
      draw_up_ok & no_throwback, upcard_d, new_state["upcard_taken"]
  )
  new_state["num_draw_upcard"] = new_num_upcard
  # Set prev_upcard to the card drawn (for repeated move detection)
  new_state["prev_upcard"] = jnp.where(
      draw_up_ok, upcard_d, new_state["prev_upcard"]
  )

  # --- Discard phase ---
  discard_phase = (phase == PHASE_DISCARD) & not_chance
  is_discard = discard_phase & (action < NUM_CARDS)
  is_knock = discard_phase & (action == ACTION_KNOCK)

  # Discard a card
  hand_disc = get_hand(new_state, player)
  hand_after_disc = hand_disc.at[safe_card_action].set(0)
  new_state["player0_hand"] = jnp.where(
      is_discard & (player == 0), hand_after_disc, new_state["player0_hand"]
  )
  new_state["player1_hand"] = jnp.where(
      is_discard & (player == 1), hand_after_disc, new_state["player1_hand"]
  )
  # Card becomes new upcard
  new_state["upcard"] = jnp.where(
      is_discard, jnp.int8(action), new_state["upcard"]
  )
  # Old upcard goes to discard pile
  old_upcard = state["upcard"]
  new_state["discard_pile"] = jnp.where(
      is_discard & (old_upcard >= 0),
      new_state["discard_pile"].at[old_upcard].set(1),
      new_state["discard_pile"],
  )

  # Check for repeated move termination (discarding same card as prev_upcard)
  prev_up = state["prev_upcard"]
  is_repeated = is_discard & (action == prev_up)
  was_repeated = state["repeated_move"]
  # If this is the second repeated move, game ends
  repeated_game_end = is_repeated & was_repeated
  new_state["done"] = jnp.where(repeated_game_end, True, new_state["done"])
  new_state["winner"] = jnp.where(
      repeated_game_end, jnp.int8(2), new_state["winner"]
  )  # Draw
  new_state["phase"] = jnp.where(
      repeated_game_end, jnp.int8(PHASE_GAME_OVER), new_state["phase"]
  )
  # If this is the first repeated move, set flag
  new_state["repeated_move"] = jnp.where(
      is_repeated & ~was_repeated, True, new_state["repeated_move"]
  )
  # Reset repeated_move if not a repeated move
  new_state["repeated_move"] = jnp.where(
      is_discard & ~is_repeated, False, new_state["repeated_move"]
  )

  # Check stock for wall phase (only if game didn't end from repeated move)
  stock_count = jnp.sum(new_state["deck"])
  enter_wall = (
      is_discard & (stock_count <= WALL_STOCK_SIZE) & ~repeated_game_end
  )
  # Without the wall pick the hand ends drawn at once. Standard gin only:
  # Oklahoma always plays the wall, including a traced oklahoma with
  # wall_knock_pick=False.
  wall_drawn = enter_wall & ~state["wall_knock_pick"] & ~state["oklahoma"]
  new_state["phase"] = jnp.where(
      is_discard & enter_wall & ~repeated_game_end,
      jnp.where(wall_drawn, jnp.int8(PHASE_GAME_OVER), jnp.int8(PHASE_WALL)),
      new_state["phase"],
  )
  new_state["done"] = jnp.where(wall_drawn, True, new_state["done"])
  new_state["winner"] = jnp.where(wall_drawn, jnp.int8(2), new_state["winner"])
  new_state["phase"] = jnp.where(
      is_discard & ~enter_wall & ~repeated_game_end,
      jnp.int8(PHASE_DRAW),
      new_state["phase"],
  )
  new_state["current_player"] = jnp.where(
      is_discard & ~repeated_game_end,
      jnp.int8(1 - player),
      new_state["current_player"],
  )
  new_state["drawn_from_discard"] = jnp.where(
      is_discard, jnp.int8(-1), new_state["drawn_from_discard"]
  )
  new_state["upcard_taken"] = jnp.where(
      is_discard, jnp.int8(-1), new_state["upcard_taken"]
  )
  # Note: prev_upcard is NOT updated on discard - it tracks the card drawn
  # from the upcard pile. It's only updated in draw_upcard actions
  # (FirstUpcard and Draw phases).

  # The derived player_deadwood feature (new_initial_state): the mover's
  # minimum deadwood at this discard.
  new_state["player_deadwood"] = jnp.where(
      is_discard,
      new_state["player_deadwood"]
      .at[player]
      .set(state["cached_min_deadwood"].astype(jnp.int16)),
      new_state["player_deadwood"],
  )

  # Knock action -> go to knock phase
  new_state["phase"] = jnp.where(
      is_knock, jnp.int8(PHASE_KNOCK), new_state["phase"]
  )
  new_state["knocker"] = jnp.where(is_knock, player, new_state["knocker"])
  new_state["knocked"] = jnp.where(
      is_knock, new_state["knocked"].at[player].set(True), new_state["knocked"]
  )

  # --- Wall phase ---
  wall_phase = (phase == PHASE_WALL) & not_chance
  wall_pass = wall_phase & (action == ACTION_PASS)
  wall_knock = wall_phase & (action == ACTION_KNOCK)

  # Pass in wall -> game ends immediately (single pass ends the game in wall
  # phase)
  new_state["done"] = jnp.where(wall_pass, True, new_state["done"])
  new_state["winner"] = jnp.where(
      wall_pass, jnp.int8(2), new_state["winner"]
  )  # Draw
  new_state["phase"] = jnp.where(
      wall_pass, jnp.int8(PHASE_GAME_OVER), new_state["phase"]
  )

  # Knock in wall - automatically add upcard to hand first
  wall_upcard = state["upcard"]
  hand_wall = get_hand(new_state, player)
  hand_with_wall_up = hand_wall.at[wall_upcard].set(1)
  new_state["player0_hand"] = jnp.where(
      wall_knock & (player == 0), hand_with_wall_up, new_state["player0_hand"]
  )
  new_state["player1_hand"] = jnp.where(
      wall_knock & (player == 1), hand_with_wall_up, new_state["player1_hand"]
  )
  new_state["upcard"] = jnp.where(wall_knock, jnp.int8(-1), new_state["upcard"])
  new_state["upcard_taken"] = jnp.where(
      wall_knock & no_throwback, wall_upcard, new_state["upcard_taken"]
  )
  new_state["phase"] = jnp.where(
      wall_knock, jnp.int8(PHASE_KNOCK), new_state["phase"]
  )
  new_state["knocker"] = jnp.where(wall_knock, player, new_state["knocker"])
  new_state["knocked"] = jnp.where(
      wall_knock,
      new_state["knocked"].at[player].set(True),
      new_state["knocked"],
  )

  # NOTE: Knocker's deadwood is NOT computed at knock time.
  # It's computed when entering layoff phase (see knock_pass handling below).
  # At that point, the knocker has already discarded and laid their melds,
  # so we can compute their actual remaining deadwood.

  # --- Knock phase ---
  knock_phase = (phase == PHASE_KNOCK) & not_chance
  knock_discard = knock_phase & (action < NUM_CARDS)
  knock_meld = (
      knock_phase
      & (action >= ACTION_MELD_BASE)
      & (action < ACTION_MELD_BASE + NUM_MELDS)
  )
  knock_pass = knock_phase & (action == ACTION_PASS)

  # Discard in knock (when have 11 cards)
  hand_k = get_hand(new_state, player)
  hand_after_k = hand_k.at[safe_card_action].set(0)
  new_state["player0_hand"] = jnp.where(
      knock_discard & (player == 0), hand_after_k, new_state["player0_hand"]
  )
  new_state["player1_hand"] = jnp.where(
      knock_discard & (player == 1), hand_after_k, new_state["player1_hand"]
  )
  # In KNOCK phase, discarded card goes directly to discard_pile (not upcard)
  # First add old upcard to discard_pile if it exists
  new_state["discard_pile"] = jnp.where(
      knock_discard & (state["upcard"] >= 0),
      new_state["discard_pile"].at[state["upcard"]].set(1),
      new_state["discard_pile"],
  )
  # Then add the discarded card to discard_pile
  new_state["discard_pile"] = jnp.where(
      knock_discard,
      new_state["discard_pile"].at[safe_card_action].set(1),
      new_state["discard_pile"],
  )
  # Clear the upcard since it's now in the discard pile
  new_state["upcard"] = jnp.where(
      knock_discard, jnp.int8(-1), new_state["upcard"]
  )
  new_state["upcard_taken"] = jnp.where(
      knock_discard, jnp.int8(-1), new_state["upcard_taken"]
  )

  # Lay meld in knock - mark meld as laid AND remove cards from hand
  meld_id = action - ACTION_MELD_BASE
  safe_meld_id = jnp.clip(meld_id, 0, NUM_MELDS - 1)
  new_state["layed_melds"] = jnp.where(
      knock_meld,
      new_state["layed_melds"].at[player, safe_meld_id].set(True),
      new_state["layed_melds"],
  )
  # Remove meld cards from hand
  meld_cards = meld_masks[safe_meld_id]
  hand_after_meld_k = get_hand(new_state, player) - meld_cards
  new_state["player0_hand"] = jnp.where(
      knock_meld & (player == 0), hand_after_meld_k, new_state["player0_hand"]
  )
  new_state["player1_hand"] = jnp.where(
      knock_meld & (player == 1), hand_after_meld_k, new_state["player1_hand"]
  )

  # Pass in knock -> go to layoff
  # Compute knocker's final deadwood (after discarding and laying melds)
  # The remaining cards ARE deadwood - just sum their values, don't look for
  # melds (any melds the knocker didn't lay are their choice, remaining cards
  # are deadwood).
  knocker_final_hand = get_hand(new_state, player)
  knocker_final_dw = hand_total_points(knocker_final_hand)
  new_state["knocker_deadwood"] = jnp.where(
      knock_pass, jnp.int32(knocker_final_dw), new_state["knocker_deadwood"]
  )

  new_state["phase"] = jnp.where(
      knock_pass, jnp.int8(PHASE_LAYOFF), new_state["phase"]
  )
  new_state["current_player"] = jnp.where(
      knock_pass, jnp.int8(1 - player), new_state["current_player"]
  )
  # If knocker has GIN, set finished_layoffs immediately (no layoffs allowed)
  knocker_has_gin = knocker_final_dw == 0
  new_state["finished_layoffs"] = jnp.where(
      knock_pass & knocker_has_gin, True, new_state["finished_layoffs"]
  )

  # --- Layoff phase ---
  layoff_phase = (phase == PHASE_LAYOFF) & not_chance
  finished = state["finished_layoffs"]
  # If knocker has gin (0 deadwood), treat as finished (no layoffs allowed, skip
  # to meld phase)
  knocker_deadwood_l = state["knocker_deadwood"]
  is_gin_l = knocker_deadwood_l == 0
  finished = finished | is_gin_l
  layoff_card = layoff_phase & ~finished & (action < NUM_CARDS)
  layoff_pass_pre = layoff_phase & ~finished & (action == ACTION_PASS)
  layoff_meld = (
      layoff_phase
      & finished
      & (action >= ACTION_MELD_BASE)
      & (action < ACTION_MELD_BASE + NUM_MELDS)
  )
  layoff_pass_post = layoff_phase & finished & (action == ACTION_PASS)

  # Lay off card onto knocker's melds
  hand_l = get_hand(new_state, player)
  hand_after_l = hand_l.at[safe_card_action].set(0)
  new_state["player0_hand"] = jnp.where(
      layoff_card & (player == 0), hand_after_l, new_state["player0_hand"]
  )
  new_state["player1_hand"] = jnp.where(
      layoff_card & (player == 1), hand_after_l, new_state["player1_hand"]
  )
  new_state["layoffs_mask"] = jnp.where(
      layoff_card,
      new_state["layoffs_mask"].at[safe_card_action].set(True),
      new_state["layoffs_mask"],
  )

  # Pass before finishing -> done with layoffs
  # Also set finished when knocker has GIN (no layoffs allowed)
  new_state["finished_layoffs"] = jnp.where(
      layoff_pass_pre | (layoff_phase & is_gin_l),
      True,
      new_state["finished_layoffs"],
  )

  # Lay own meld - mark meld as laid AND remove cards from hand
  meld_id_l = action - ACTION_MELD_BASE
  safe_meld_id_l = jnp.clip(meld_id_l, 0, NUM_MELDS - 1)
  new_state["layed_melds"] = jnp.where(
      layoff_meld,
      new_state["layed_melds"].at[player, safe_meld_id_l].set(True),
      new_state["layed_melds"],
  )
  # Remove meld cards from hand
  meld_cards_l = meld_masks[safe_meld_id_l]
  hand_after_meld_l = get_hand(new_state, player) - meld_cards_l
  new_state["player0_hand"] = jnp.where(
      layoff_meld & (player == 0), hand_after_meld_l, new_state["player0_hand"]
  )
  new_state["player1_hand"] = jnp.where(
      layoff_meld & (player == 1), hand_after_meld_l, new_state["player1_hand"]
  )

  # Pass after finishing -> calculate scores and end game
  knocker_p = new_state["knocker"]
  opponent = 1 - knocker_p
  # Clip player indices for eager array indexing: layoff_pass_post guards
  # scoring, but jnp.where evaluates both branches. knocker_p may be an
  # unset/terminal sentinel outside [0,1] (e.g., -1 or 2), making opponent
  # invalid too; clamp both to [0,1] for 2-element arrays.
  safe_knocker_p = jnp.clip(knocker_p, 0, 1)
  safe_opponent = jnp.clip(opponent, 0, 1)
  knocker_hand = jnp.where(
      safe_knocker_p == 0, new_state["player0_hand"], new_state["player1_hand"]
  )
  opponent_hand = jnp.where(
      safe_opponent == 0, new_state["player0_hand"], new_state["player1_hand"]
  )

  # Use raw point totals - melds have already been removed from hands
  # Any remaining cards are deadwood (player chose not to lay those melds)
  knocker_dw = hand_total_points(knocker_hand)
  opponent_dw = hand_total_points(opponent_hand)

  is_gin = knocker_dw == 0
  is_undercut = opponent_dw <= knocker_dw

  # Scoring - cast bonuses to int32 to prevent int64 promotion
  gin_score = opponent_dw + jnp.int32(GIN_BONUS)
  undercut_score = knocker_dw - opponent_dw + jnp.int32(UNDERCUT_BONUS)
  normal_score = knocker_dw - opponent_dw

  final_score = jnp.where(
      is_gin, gin_score, jnp.where(is_undercut, -undercut_score, -normal_score)
  )

  # Under spade_doubles the whole hand score, bonuses included, is doubled.
  final_score = final_score * state["score_multiplier"].astype(jnp.int32)
  p0_score = jnp.where(knocker_p == 0, final_score, -final_score)

  new_state["deadwood"] = jnp.where(
      layoff_pass_post,
      new_state["deadwood"]
      .at[safe_knocker_p]
      .set(jnp.int32(knocker_dw))
      .at[safe_opponent]
      .set(jnp.int32(opponent_dw)),
      new_state["deadwood"],
  )
  new_state["p0_score"] = jnp.where(
      layoff_pass_post, jnp.int32(p0_score), new_state["p0_score"]
  )
  new_state["p1_score"] = jnp.where(
      layoff_pass_post, jnp.int32(-p0_score), new_state["p1_score"]
  )
  new_state["done"] = jnp.where(layoff_pass_post, True, new_state["done"])
  winner = jnp.where(p0_score > 0, 0, jnp.where(p0_score < 0, 1, 2))
  new_state["winner"] = jnp.where(
      layoff_pass_post, jnp.int8(winner), new_state["winner"]
  )
  new_state["phase"] = jnp.where(
      layoff_pass_post, jnp.int8(PHASE_GAME_OVER), new_state["phase"]
  )

  # A finished game has TERMINAL_PLAYER to move.
  new_state["current_player"] = jnp.where(
      new_state["done"], jnp.int8(TERMINAL_PLAYER), new_state["current_player"]
  )

  # === MELD INFO CACHING ===
  # Recompute when entering phases that need meld info
  # For KNOCK and LAYOFF, recompute every step (hand changes within phase)
  old_phase = state["phase"]
  new_phase = new_state["phase"]

  entering_discard = (new_phase == PHASE_DISCARD) & (old_phase != PHASE_DISCARD)
  entering_wall = (new_phase == PHASE_WALL) & (old_phase != PHASE_WALL)
  entering_draw = (new_phase == PHASE_DRAW) & (old_phase != PHASE_DRAW)
  # For KNOCK and LAYOFF, always recompute (hand changes with each meld/discard)
  in_knock = new_phase == PHASE_KNOCK
  in_layoff = new_phase == PHASE_LAYOFF
  # For FIRST_UPCARD, always recompute (current player can change when one
  # passes)
  in_first_upcard = new_phase == PHASE_FIRST_UPCARD

  need_recompute = (
      entering_discard
      | entering_wall
      | entering_draw
      | in_knock
      | in_layoff
      | in_first_upcard
  )

  # Get the current player's hand in new state
  new_player = new_state["current_player"]
  safe_player = jnp.where(new_player >= 0, new_player, jnp.int8(0))
  current_hand = jnp.where(
      safe_player == 0, new_state["player0_hand"], new_state["player1_hand"]
  )

  # Build effective hand based on phase:
  # - FIRST_UPCARD/DRAW/WALL: evaluate hand + upcard (player has 10 cards,
  #   considering taking upcard)
  # - DISCARD/KNOCK/LAYOFF: use current hand as-is
  upcard_val = new_state["upcard"]
  effective_hand = jnp.where(
      (in_first_upcard | entering_draw | entering_wall) & (upcard_val >= 0),
      current_hand.at[upcard_val].set(1),
      current_hand,
  )

  # The knock card is per hand: set by the first upcard under Oklahoma
  # rules, KNOCK_THRESHOLD in standard play.
  knock_limit = new_state["knock_card"].astype(jnp.int32)
  _, new_all_discard_deadwoods = meld_info(effective_hand)
  new_legal_knocker_melds = knocker_melds(effective_hand, knock_limit)

  # The card that may not be the discard under allow_upcard_throwback=False:
  # at FIRST_UPCARD / DRAW / WALL the upcard the evaluation takes; at DISCARD
  # / KNOCK the upcard taken this turn. -1 (none) under the default rules.
  evaluates_upcard = in_first_upcard | entering_draw | entering_wall
  kept = jnp.where(
      new_state["allow_upcard_throwback"],
      jnp.int8(-1),
      jnp.where(evaluates_upcard, upcard_val, new_state["upcard_taken"]),
  )
  discardable = (effective_hand > 0) & (jnp.arange(NUM_CARDS) != kept)

  # Derive legal_discard_mask and min_deadwood from all_discard_deadwoods
  new_legal_discard_mask = discardable & (
      new_all_discard_deadwoods <= knock_limit
  )
  new_min_deadwood = jnp.min(
      jnp.where(discardable, new_all_discard_deadwoods, jnp.int32(999))
  )

  # can_knock: is there ANY card we can discard that keeps deadwood <=
  # threshold?
  new_can_knock = jnp.any(new_legal_discard_mask)

  # Conditionally update cached values (only when entering relevant phases)
  new_state["cached_min_deadwood"] = jnp.int32(
      jnp.where(need_recompute, new_min_deadwood, state["cached_min_deadwood"])
  )
  new_state["cached_legal_knocker_melds"] = jnp.where(
      need_recompute,
      new_legal_knocker_melds,
      state["cached_legal_knocker_melds"],
  )
  new_state["cached_legal_discard_mask"] = jnp.where(
      need_recompute, new_legal_discard_mask, state["cached_legal_discard_mask"]
  )
  new_state["cached_all_discard_deadwoods"] = jnp.where(
      need_recompute,
      new_all_discard_deadwoods.astype(jnp.int16),
      state["cached_all_discard_deadwoods"],
  )
  new_state["can_knock"] = jnp.where(
      need_recompute, new_can_knock, state["can_knock"]
  )

  # known_cards, as OpenSpiel's known_cards_: a bit is set when the player
  # takes the upcard (first upcard, draw or wall), and multiplying by the
  # hand clears every card that has left it (discard, knock discard, meld,
  # layoff), so known is always a subset of hand. Chance steps leave it
  # unchanged.
  took_upcard = (draw_first | draw_up_ok | wall_knock) & (state["upcard"] >= 0)
  up_onehot = jax.nn.one_hot(
      jnp.maximum(state["upcard"], 0), NUM_CARDS, dtype=jnp.int8
  ) * took_upcard.astype(jnp.int8)
  zeros52 = jnp.zeros(NUM_CARDS, dtype=jnp.int8)
  known0 = (
      jnp.maximum(
          state["known_cards"][0], jnp.where(player == 0, up_onehot, zeros52)
      )
      * new_state["player0_hand"]
  )
  known1 = (
      jnp.maximum(
          state["known_cards"][1], jnp.where(player == 1, up_onehot, zeros52)
      )
      * new_state["player1_hand"]
  )
  new_state["known_cards"] = jnp.stack([known0, known1]).astype(jnp.int8)

  return new_state


step = jax.jit(_step_body)


@jax.jit
def get_returns(state):
  """Returns each player's score.

  Args:
    state: The state.

  Returns:
    f32[2]: each player's score; zero before the end.
  """
  return jnp.array([state["p0_score"], state["p1_score"]], dtype=jnp.float32)


# =============================================================================
# The Game interface
# =============================================================================

# The cached constant tables are built once at import, as numpy arrays, so
# the first trace does not pay for them.
init_lazy_constants()


class GinRummy(game_lib.Game, interop.PyspielInterop):
  """Gin rummy, OpenSpiel's `gin_rummy` (two players).

  State: the dict of `init_state`. The deal is 21 chance nodes (ten cards to
  each player, then the upcard); stock draws are chance nodes too. Chance
  action ids are card ids 0..51 (card = suit * 13 + rank, suits in the order
  spades, clubs, diamonds, hearts), uniform over the cards left.

  Actions, as in OpenSpiel: 0..51 discard (or lay off) that card, 52 draw
  the upcard, 53 draw from the stock, 54 pass, 55 knock, and 56 onwards lay
  meld i - 56 of `ALL_MELDS`.

  Observation: OpenSpiel's default observer tensor, f32[644] under the
  default rules, plus the planes of `observation_tensor_size` for a
  non-default `allow_upcard_throwback` or `spade_doubles`. The planes follow
  the game's rules, so a batch whose states mix rules is observed in the
  game's layout. There is no information-state tensor, as in OpenSpiel.

  Payoffs: the hand score for the winner and its negation for the loser,
  with OpenSpiel's gin (25) and undercut (25) bonuses.
  """

  def __init__(
      self,
      oklahoma: bool = False,
      allow_upcard_throwback: bool = DEFAULT_RULES["allow_upcard_throwback"],
      wall_knock_pick: bool = DEFAULT_RULES["wall_knock_pick"],
      spade_doubles: bool = DEFAULT_RULES["spade_doubles"],
      backend: str = "reference",
  ):
    """Builds the game under the given rules.

    Args:
      oklahoma: If True, the first upcard sets each hand's knock card (an ace
        means gin only), as OpenSpiel's `gin_rummy(oklahoma=True)`.
      allow_upcard_throwback: A rule variant of `DEFAULT_RULES`; the default
        is OpenSpiel's rule.
      wall_knock_pick: A rule variant of `DEFAULT_RULES`; the default is
        OpenSpiel's rule. `wall_knock_pick=False` requires standard gin.
      spade_doubles: A rule variant of `DEFAULT_RULES`; the default is
        OpenSpiel's rule. `spade_doubles=True` requires `oklahoma`.
      backend: How the meld caches are computed: "reference" (lookup tables)
        or "dense_cache" (dense array code, faster on TPUs). Both play the
        identical game; see `dense_cache_game`.

    Raises:
      TypeError: on rules of the wrong type (`_check_rules`).
      ValueError: on inconsistent rules or an unknown backend.
    """
    _check_rules(
        oklahoma,
        allow_upcard_throwback,
        wall_knock_pick,
        spade_doubles,
        where="GinRummy",
    )
    if backend == "reference":
      self._step = step
    elif backend == "dense_cache":
      # Lazy: breaks a real cycle (dense_cache_game imports this module).
      from open_spiel.python.jax.games.gin_rummy import dense_cache_game  # pylint: disable=g-import-not-at-top,import-outside-toplevel  # noqa: PLC0415

      self._step = dense_cache_game.step
    else:
      raise ValueError(
          "GinRummy backend must be 'reference' or 'dense_cache', got "
          f"{backend!r}"
      )
    self._backend = backend
    self._num_players = 2
    self._oklahoma = bool(oklahoma)
    self._rules = {
        "allow_upcard_throwback": bool(allow_upcard_throwback),
        "wall_knock_pick": bool(wall_knock_pick),
        "spade_doubles": bool(spade_doubles),
    }

  @property
  def oklahoma(self) -> bool:
    """Whether the game plays Oklahoma rules."""
    return self._oklahoma

  @property
  def backend(self) -> str:
    """The name of the meld-computation backend."""
    return self._backend

  @property
  def rules(self) -> dict[str, bool]:
    """The rule parameters this game plays (a copy)."""
    return dict(self._rules)

  @property
  def num_actions(self) -> int:
    """The number of distinct player actions.

    OpenSpiel calls it `num_distinct_actions`.
    """
    return NUM_ACTIONS

  @property
  def num_players(self) -> int:
    """The number of players."""
    return self._num_players

  @property
  def obs_shape(self) -> tuple[int, ...]:
    """The shape of `observation_tensor`."""
    return (
        observation_tensor_size(
            self._rules["allow_upcard_throwback"],
            self._rules["spade_doubles"],
        ),
    )

  @property
  def max_chance_outcomes(self) -> int:
    """The number of distinct chance outcomes; 0 for games without chance."""
    return NUM_CARDS

  def new_initial_state(self):
    """Returns the initial state, a chance node before the deal.

    Returns:
      The pre-deal state.
    """
    return init_state(oklahoma=self._oklahoma, **self._rules)

  def apply_action(self, state, action):
    """Returns the state after `action`.

    Args:
      state: The current state.
      action: i32[], a legal player action, or a chance action id with
        nonzero probability at a chance node.

    Returns:
      The next state. Applying an illegal action gives an unspecified state;
      it does not raise, because the method is traced.
    """
    return self._step(state, action)

  def current_player(self, state) -> jax.Array:
    """Returns the player to move.

    Args:
      state: The current state.

    Returns:
      i32[]: the player to move, or `CHANCE_PLAYER_ID` at a chance node. The
      value at a terminal state is game-specific.
    """
    is_done = state["done"]
    is_chance = state["phase"] == PHASE_DEAL
    cp = state["current_player"].astype(jnp.int32)
    cp = jnp.where(is_chance, jnp.int32(game_lib.CHANCE_PLAYER_ID), cp)
    cp = jnp.where(is_done, jnp.int32(game_lib.TERMINAL_PLAYER_ID), cp)
    return cp

  def is_terminal(self, state) -> jax.Array:
    """Returns whether the game is over.

    Args:
      state: The current state.

    Returns:
      bool[]: True if the game is over.
    """
    return state["done"]

  def is_chance_node(self, state) -> jax.Array:
    """Returns whether the next action is a chance outcome.

    Args:
      state: The current state.

    Returns:
      bool[]: True if the next action is a chance outcome.
    """
    return state["phase"] == PHASE_DEAL

  def chance_outcomes(self, state) -> jax.Array:
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
    return chance_outcomes(state)

  def legal_actions_mask(self, state) -> jax.Array:
    """Returns the legal player actions.

    Args:
      state: The current state.

    Returns:
      bool[num_actions]: the legal player actions. All False at a terminal state
      and, for most games, at a chance node.
    """
    return legal_actions_mask(state)

  def observation_tensor(self, state, observing_player: jax.Array) -> jax.Array:
    """Returns OpenSpiel's default observer tensor, then rule-variant planes.

    The extra planes are those of the game's non-default rule variants.

    Args:
      state: The state.
      observing_player: i32[], the observing player.

    Returns:
      f32[obs_size]: the observation tensor.
    """
    player_plane = (
        jnp.arange(self.num_players, dtype=jnp.int32) == observing_player
    )

    p0_private = jnp.where(
        observing_player == 0,
        state["player0_hand"],
        jnp.zeros(NUM_CARDS, dtype=state["player0_hand"].dtype),
    )
    p1_private = jnp.where(
        observing_player == 1,
        state["player1_hand"],
        jnp.zeros(NUM_CARDS, dtype=state["player1_hand"].dtype),
    )
    private_hands = jnp.concatenate([p0_private, p1_private])

    current_player = state["current_player"]
    current_player_plane = (
        jnp.arange(self.num_players, dtype=jnp.int32) == current_player
    ) & (current_player >= 0)

    knock_card = jnp.arange(KNOCK_THRESHOLD, dtype=jnp.int32) < state[
        "knock_card"
    ].astype(jnp.int32)
    upcard = (jnp.arange(NUM_CARDS, dtype=jnp.int32) == state["upcard"]) & (
        state["upcard"] >= 0
    )
    stock_size = jnp.arange(NUM_CARDS, dtype=jnp.int32) < jnp.sum(state["deck"])

    planes = [
        player_plane,
        private_hands,
        current_player_plane,
        knock_card,
        upcard,
        state["discard_pile"] > 0,
        stock_size,
        state["layed_melds"].reshape((self.num_players * NUM_MELDS,)),
    ]
    if not self._rules["allow_upcard_throwback"]:
      taken = state["upcard_taken"].astype(jnp.int32)
      planes.append(
          (jnp.arange(NUM_CARDS, dtype=jnp.int32) == taken) & (taken >= 0)
      )
    if self._rules["spade_doubles"]:
      planes.append(
          (state["score_multiplier"].astype(jnp.int32) == 2).reshape((1,))
      )
    return jnp.concatenate(planes).astype(jnp.float32)

  def returns(self, state) -> jax.Array:
    """Returns each player's return.

    Args:
      state: The current state.

    Returns:
      f32[num_players]: each player's return; zero before the end.
    """
    return get_returns(state).astype(jnp.float32)

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

  provides_information_state_tensor = False

  @property
  def info_state_shape(self) -> tuple[int, ...]:
    """Raises NotImplementedError: there is no information-state tensor."""
    raise NotImplementedError(
        "gin_rummy has no information_state_tensor (as in OpenSpiel), so no "
        "info_state_shape; check provides_information_state_tensor."
    )

  def information_state_tensor(
      self, state, observing_player: jax.Array
  ) -> jax.Array:
    """Not available: gin rummy has no information-state tensor (as OpenSpiel).

    Args:
      state: Unused.
      observing_player: Unused.

    Raises:
      NotImplementedError: always.
    """
    del state, observing_player
    raise NotImplementedError(
        "pyspiel gin_rummy does not expose information_state_tensor."
    )

  def information_state_from_os(
      self, os_state: "pyspiel.State", player: int
  ) -> jax.Array:
    """Not available: pyspiel gin_rummy has no information-state tensor.

    Args:
      os_state: Unused.
      player: Unused.

    Raises:
      NotImplementedError: always.
    """
    del os_state, player
    raise NotImplementedError(
        "pyspiel gin_rummy does not expose information_state_tensor."
    )
