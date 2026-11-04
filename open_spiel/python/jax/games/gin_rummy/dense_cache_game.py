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

"""The dense-cache backend of gin rummy: `GinRummy(backend="dense_cache")`.

The reference step computes each hand's meld caches with large lookup-table
gathers, which accelerators such as TPUs run slowly. This backend computes
the caches with the dense array code of `dense_cache`. Every state field
equals the reference backend's, so legal actions, observations and returns
are identical.

  game = GinRummy(backend="dense_cache")
  state = step(state, action)                 # this module's step
  states = batch_step(states, actions)        # a batch of states

`batch_step` and `batch_legal_actions_mask` also skip, inside the vmapped
step, the KNOCK-phase meld cache that only KNOCK states read, and compute it
for the whole batch only when some state is in KNOCK.
"""

import functools

import jax
import jax.numpy as jnp

from open_spiel.python.jax.games.gin_rummy import dense_cache
from open_spiel.python.jax.games.gin_rummy import game as reference
from open_spiel.python.jax.games.gin_rummy.cards_and_melds import NUM_MELDS
from open_spiel.python.jax.games.gin_rummy.cards_and_melds import PHASE_KNOCK
from open_spiel.python.jax.games.gin_rummy.cards_and_melds import PHASE_LAYOFF


def _zero_legal_knocker_melds(hand, threshold=reference.KNOCK_THRESHOLD):
  """Placeholder knocker melds for a batch step that fills them afterwards.

  Args:
    hand: The hand (unused).
    threshold: The knock threshold (unused).

  Returns:
    bool[NUM_MELDS], all False.
  """
  del hand, threshold
  return jnp.zeros(NUM_MELDS, dtype=jnp.bool_)


_step_body = functools.partial(
    # Uses private _step_body of a module in this package.
    reference._step_body,  # pylint: disable=protected-access
    meld_info=dense_cache.compute_all_meld_info_dense_discard_only,
    knocker_melds=dense_cache.compute_legal_knocker_melds_selection_matrix,
)

_step_body_without_legal_knocker_melds = functools.partial(  # pylint: disable=invalid-name
    # Uses private _step_body of a module in this package.
    reference._step_body,  # pylint: disable=protected-access
    meld_info=dense_cache.compute_all_meld_info_dense_discard_only,
    knocker_melds=_zero_legal_knocker_melds,
)

_init_dealt_state_body = functools.partial(
    # Uses private _init_dealt_state_body of a module in this package.
    reference._init_dealt_state_body,  # pylint: disable=protected-access
    meld_info=dense_cache.compute_all_meld_info_dense_discard_only,
)


def _batched_current_hands(states):
  """The hand of each state's player to move.

  Args:
    states: A batch of states.

  Returns:
    int8[B, 52], the hands.
  """
  safe_player = jnp.where(
      states["current_player"] >= 0, states["current_player"], 0
  )
  return jnp.where(
      safe_player[:, None] == 0, states["player0_hand"], states["player1_hand"]
  )


def _repair_batched_legal_knocker_melds(states):
  """Fills the KNOCK-phase meld cache of a batch after a batch step.

  The cache is computed for the whole batch only if some state is in KNOCK.
  Other states keep a stale or zero cache, which is never read:
  legal_actions_mask reads cached_legal_knocker_melds only in PHASE_KNOCK.

  Args:
    states: The batch of states.

  Returns:
    The states, with the cache filled where the phase is KNOCK.
  """
  in_knock = states["phase"] == PHASE_KNOCK

  def repair():
    """Fills the knocker-meld cache of the states in KNOCK.

    Returns:
      The states with the cache filled.
    """
    # Each state's own knock card: under Oklahoma rules it varies by hand.
    legal_melds = jax.vmap(
        dense_cache.compute_legal_knocker_melds_selection_matrix
    )(_batched_current_hands(states), states["knock_card"].astype(jnp.int32))
    return {
        **states,
        "cached_legal_knocker_melds": jnp.where(
            in_knock[:, None], legal_melds, states["cached_legal_knocker_melds"]
        ),
    }

  return jax.lax.cond(jnp.any(in_knock), repair, lambda: states)


@jax.jit
def batch_step(states, actions):
  """Returns the batch of states after `actions`; equal to vmapped `step`.

  Args:
    states: The batched states.
    actions: The actions.

  Returns:
    The next states.
  """
  states = jax.vmap(_step_body_without_legal_knocker_melds)(states, actions)
  return _repair_batched_legal_knocker_melds(states)


@jax.jit
def batch_legal_actions_mask(states):
  """Returns the legal actions; equal to vmapped `legal_actions_mask`.

  Args:
    states: The batched states.

  Returns:
    bool[B, NUM_ACTIONS].
  """
  common_masks = jax.vmap(reference.legal_actions_mask_common)(states)
  phase = states["phase"]
  needs_full_mask = (phase == PHASE_KNOCK) | (phase == PHASE_LAYOFF)

  def repair():
    """Uses the full mask for the states in KNOCK or LAYOFF.

    Returns:
      bool[B, NUM_ACTIONS].
    """
    full_masks = jax.vmap(reference.legal_actions_mask)(states)
    return jnp.where(needs_full_mask[:, None], full_masks, common_masks)

  return jax.lax.cond(jnp.any(needs_full_mask), repair, lambda: common_masks)


init_state = reference.init_state
_init_dealt_state_jit = jax.jit(_init_dealt_state_body)


def init_dealt_state(
    key,
    oklahoma=False,
    randomized_knock_limit=False,
    allow_upcard_throwback=reference.DEFAULT_RULES["allow_upcard_throwback"],
    wall_knock_pick=reference.DEFAULT_RULES["wall_knock_pick"],
    spade_doubles=reference.DEFAULT_RULES["spade_doubles"],
):
  """Returns `reference.init_dealt_state` with this backend's meld caches.

  Args:
    key: The PRNG key.
    oklahoma: Whether the hand gets an Oklahoma knock card.
    randomized_knock_limit: Whether an Oklahoma knock card is sampled.
    allow_upcard_throwback: A rule variant.
    wall_knock_pick: A rule variant.
    spade_doubles: A rule variant.

  Returns:
    (state, new_key).
  """
  # Uses private _check_rules of a module in this package.
  reference._check_rules(  # pylint: disable=protected-access
      oklahoma,
      allow_upcard_throwback,
      wall_knock_pick,
      spade_doubles,
      where="gin_rummy.dense_cache_game.init_dealt_state",
  )
  return _init_dealt_state_jit(
      key,
      oklahoma,
      randomized_knock_limit,
      allow_upcard_throwback,
      wall_knock_pick,
      spade_doubles,
  )


is_chance_node = reference.is_chance_node
get_hand = reference.get_hand
chance_outcomes_mask = reference.chance_outcomes_mask
legal_actions_mask = reference.legal_actions_mask
legal_actions_mask_common = reference.legal_actions_mask_common
step = jax.jit(_step_body)
get_returns = reference.get_returns
