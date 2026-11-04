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

"""OpenSpiel's simple gin rummy bot as a jittable JAX policy.

`simple_bot_action_opt(state)` returns the action OpenSpiel's
SimpleGinRummyBot chooses at a player node, as a pure function of the state,
so a batch of bots runs under `vmap` inside a compiled rollout.

  action = simple_bot_action_opt(state)
  state = gin_rummy.step(state, action)

The bot plays the rule variants the state carries (`game.DEFAULT_RULES`):

* `allow_upcard_throwback=False`: the upcard taken this turn
  (`state["upcard_taken"]`) is never the bot's discard, at DISCARD or as the
  knocker's 11-card discard, and every deadwood the bot compares with the
  knock card (the knock at DISCARD, taking the upcard, the WALL knock) is
  the minimum over the other discards, as the engine's `can_knock` is. The
  bot therefore knocks exactly when the engine offers a knock. Where the
  optimal arrangement's only deadwood card is the taken upcard, the discard
  is the other card leaving the least deadwood (ties: higher rank, then
  higher card index). OpenSpiel's C++ bot does not support this variant.
* `spade_doubles=True` changes only the score, which the engine computes;
  the bot's choices do not depend on the stake.
"""

import jax
import jax.numpy as jnp

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
from open_spiel.python.jax.games.gin_rummy.cards_and_melds import get_meld_masks
from open_spiel.python.jax.games.gin_rummy.cards_and_melds import NUM_MELDS
from open_spiel.python.jax.games.gin_rummy.cards_and_melds import PHASE_DEAL
from open_spiel.python.jax.games.gin_rummy.cards_and_melds import PHASE_DISCARD
from open_spiel.python.jax.games.gin_rummy.cards_and_melds import PHASE_DRAW
from open_spiel.python.jax.games.gin_rummy.cards_and_melds import (
    PHASE_FIRST_UPCARD,
)
from open_spiel.python.jax.games.gin_rummy.cards_and_melds import PHASE_KNOCK
from open_spiel.python.jax.games.gin_rummy.cards_and_melds import PHASE_LAYOFF
from open_spiel.python.jax.games.gin_rummy.cards_and_melds import PHASE_WALL
from open_spiel.python.jax.games.gin_rummy.cards_and_melds import (
    WALL_STOCK_SIZE,
)
from open_spiel.python.jax.games.gin_rummy.deadwood import (
    _compute_optimal_meld_info,
)
from open_spiel.python.jax.games.gin_rummy.deadwood import (
    calculate_deadwood_compressed,
)
from open_spiel.python.jax.games.gin_rummy.deadwood import hand_to_4x13
from open_spiel.python.jax.games.gin_rummy.deadwood import optimal_melds_mask
from open_spiel.python.jax.games.gin_rummy.game import get_hand

__all__ = [
    "simple_bot_action_opt",
]


def _held_back_upcards(state):
  """Returns the upcards that may not be the discard, ``-1`` when none.

  ``taken``: at DISCARD / KNOCK, the upcard taken this turn. ``evaluated``: at
  FIRST_UPCARD / DRAW / WALL, the upcard whose take the bot is evaluating. Both
  are ``-1`` whenever ``allow_upcard_throwback`` holds (the default).

  Args:
    state: The state.

  Returns:
    (taken, evaluated), i32 scalars.
  """
  allow = state["allow_upcard_throwback"]
  taken = jnp.where(
      allow, jnp.int32(-1), state["upcard_taken"].astype(jnp.int32)
  )
  evaluated = jnp.where(allow, jnp.int32(-1), state["upcard"].astype(jnp.int32))
  return taken, evaluated


def _min_over_discards(indices, dws, held_back):
  """Min of ``dws`` over the discards other than ``held_back`` (-1 = none).

  Args:
    indices: The held cards.
    dws: The deadwood after each discard.
    held_back: The card held back, or -1.

  Returns:
    i16[], the minimum deadwood.
  """
  return jnp.min(jnp.where(indices != held_back, dws, jnp.int16(999)))


def _fallback_discard(hand, hand_indices, hand_dws, taken):
  """The non-taken held card leaving the least deadwood (ties: rank, index).

  Args:
    hand: int8[52], the hand.
    hand_indices: The held cards' indices.
    hand_dws: The deadwood after discarding each.
    taken: The card just taken, or -1.

  Returns:
    The card to discard.
  """
  ok = (jnp.take(hand, hand_indices) > 0) & (hand_indices != taken)
  score = (
      -hand_dws.astype(jnp.int32) * 10000
      + (hand_indices % 13) * 100
      + hand_indices
  )
  return hand_indices[jnp.argmax(jnp.where(ok, score, jnp.int32(-(1 << 30))))]


@jax.jit
def simple_bot_action_opt(state):
  """Returns the simple bot's action at a player node of `state` (module doc).

  Its deadwood comes from the compressed deadwood solver
  (`deadwood.calculate_deadwood_compressed`).

  Args:
    state: The state.

  Returns:
    i32[], the action.
  """
  meld_masks = jnp.asarray(get_meld_masks())

  phase = state["phase"]
  player = state["current_player"]
  hand = get_hand(state, player)
  upcard = state["upcard"]
  deck = state["deck"]

  # 1. Compute optimal melds for 11-card hand, then pick discard from deadwood
  # C++ approach: BestMeldGroup(11-card hand) → deadwood → sort by
  # RankComparator → back()
  hand_4x13 = hand_to_4x13(hand)
  _, optimal_meld_mask_11card = _compute_optimal_meld_info(hand_4x13)

  # Compute which cards are in optimal melds
  # melded_cards[i] = 1 if card i is in any optimal meld
  melded_cards = jnp.any(
      meld_masks * optimal_meld_mask_11card[:, None], axis=0
  )  # (52,)

  # Deadwood = cards in hand but not in optimal melds. Under
  # allow_upcard_throwback=False the upcard taken this turn is never the discard
  # (-1 by default: no card excluded).
  taken, evaluated = _held_back_upcards(state)
  card_indices = jnp.arange(52, dtype=jnp.int32)
  deadwood_mask = (hand > 0) & (~melded_cards) & (card_indices != taken)

  # Compute min deadwood for knock check
  hand_indices, hand_dws = calculate_deadwood_compressed(hand)
  min_dw_hand = _min_over_discards(hand_indices, hand_dws, taken)

  # Handle 11-card gin: if no deadwood, find card that preserves gin when
  # discarded
  has_deadwood = jnp.any(deadwood_mask)
  # For gin: pick card where discarding it still gives 0 deadwood
  gin_preserving = (
      (hand_dws == 0) & (hand > 0).take(hand_indices) & (hand_indices != taken)
  )
  gin_discard_idx = jnp.argmax(gin_preserving)  # First card that preserves gin
  gin_discard_card = hand_indices[gin_discard_idx]
  # The optimal arrangement's only deadwood is the taken upcard, so no
  # discard keeps gin; play the other discard leaving the least deadwood.
  # Never selected under the default rules (``taken`` is -1).
  gin_discard_card = jnp.where(
      jnp.any(gin_preserving) | (taken < 0),
      gin_discard_card,
      _fallback_discard(hand, hand_indices, hand_dws, taken),
  )

  # Find best discard from deadwood (highest rank, then highest card index)
  card_ranks = card_indices % 13
  # Score = rank * 100 + card_index: OpenSpiel's bot orders by rank, then card.
  discard_score = jnp.where(
      deadwood_mask,
      card_ranks.astype(jnp.int32) * 100 + card_indices,
      jnp.int32(-1),
  )
  deadwood_discard_card = jnp.argmax(discard_score)

  # Use gin discard if no deadwood, else use deadwood discard
  best_discard_card = jnp.where(
      has_deadwood, deadwood_discard_card, gin_discard_card
  )

  # 2. Compressed deadwood with upcard (for draw decisions)
  hand_with_upcard = jnp.where(upcard >= 0, hand.at[upcard].set(1), hand)
  # When a taken upcard cannot be thrown back, the take (and the
  # WALL knock) is evaluated over the other discards, as the engine's
  # ``can_knock`` is.
  up_indices, up_dws = calculate_deadwood_compressed(hand_with_upcard)
  min_dw_with_upcard = _min_over_discards(up_indices, up_dws, evaluated)

  # === DECISION LOGIC ===

  # Draw phase evaluation: take upcard if it enables knock OR if upcard is in
  # optimal melds.
  # C++ logic: GetBestDeadwood(hand, upcard) returns deadwood for 11-card hand
  # If upcard NOT in deadwood → upcard is in optimal melds → take it
  knock_limit = state["knock_card"].astype(jnp.int32)
  upcard_enables_knock = min_dw_with_upcard <= knock_limit

  # Check if upcard is in optimal melds for the 11-card hand
  hand_with_upcard_4x13 = hand_to_4x13(hand_with_upcard)
  _, optimal_meld_mask_11 = _compute_optimal_meld_info(hand_with_upcard_4x13)
  # Check if any optimal meld contains the upcard
  upcard_in_optimal = jnp.where(
      upcard >= 0,
      jnp.any(optimal_meld_mask_11 & (meld_masks[:, upcard] > 0)),
      False,
  )
  take_upcard = upcard_enables_knock | upcard_in_optimal

  # Chance action (PHASE_DEAL covers both initial deal and stock draw)
  is_chance = phase == PHASE_DEAL
  chance_action = jnp.argmax(deck)

  # First upcard action
  # After both players pass, must draw from stock
  both_passed = state["pass_count"] >= 2
  first_upcard_action = jnp.where(
      both_passed,
      ACTION_DRAW_STOCK,
      jnp.where(take_upcard, ACTION_DRAW_UPCARD, ACTION_PASS),
  )

  # Draw action
  is_wall = jnp.sum(deck) <= WALL_STOCK_SIZE
  draw_action = jnp.where(
      take_upcard & (upcard >= 0), ACTION_DRAW_UPCARD, ACTION_DRAW_STOCK
  )
  draw_action = jnp.where(is_wall & ~take_upcard, ACTION_PASS, draw_action)

  # Discard action
  can_knock = min_dw_hand <= knock_limit
  discard_action = jnp.where(can_knock, ACTION_KNOCK, best_discard_card)

  # Knock action: discard if 11 cards, else lay optimal melds, else pass
  hand_size = jnp.sum(hand)
  optimal_melds = optimal_melds_mask(hand)
  has_meld_to_lay = jnp.any(optimal_melds)
  # Pick highest-index optimal meld (order doesn't affect final score)
  meld_indices = jnp.arange(NUM_MELDS, dtype=jnp.int32)
  meld_score = jnp.where(optimal_melds, meld_indices, jnp.int32(-1))
  best_meld_idx = jnp.argmax(meld_score)
  meld_action = ACTION_MELD_BASE + best_meld_idx
  knock_action = jnp.where(
      hand_size >= 11,
      best_discard_card,
      jnp.where(has_meld_to_lay, meld_action, ACTION_PASS),
  )

  # Wall action
  wall_can_knock = min_dw_with_upcard <= knock_limit
  wall_action = jnp.where(
      wall_can_knock & (upcard >= 0), ACTION_KNOCK, ACTION_PASS
  )

  # Layoff action: pass on laying off cards, but lay own optimal melds
  # Check if we're in pre-finish (layoffs) or post-finish (own melds) stage
  finished_layoffs = state["finished_layoffs"]
  knocker_deadwood = state["knocker_deadwood"]
  is_gin = knocker_deadwood == 0
  # When knocker has GIN, defender just passes (no layoffs or own melds allowed)
  # Otherwise, in post-finish: lay optimal melds if any, else pass
  layoff_action = jnp.where(
      is_gin,
      ACTION_PASS,
      jnp.where(finished_layoffs & has_meld_to_lay, meld_action, ACTION_PASS),
  )

  # Multiplexer
  action = jnp.where(is_chance, chance_action, jnp.int32(0))
  action = jnp.where(phase == PHASE_FIRST_UPCARD, first_upcard_action, action)
  action = jnp.where(phase == PHASE_DRAW, draw_action, action)
  action = jnp.where(phase == PHASE_DISCARD, discard_action, action)
  action = jnp.where(phase == PHASE_KNOCK, knock_action, action)
  action = jnp.where(phase == PHASE_WALL, wall_action, action)
  action = jnp.where(phase == PHASE_LAYOFF, layoff_action, action)

  return action.astype(jnp.int32)
