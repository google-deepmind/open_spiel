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

"""Shared builders for the scripted gin policy tests."""

import jax
import jax.numpy as jnp
import numpy as np

from open_spiel.python.jax.games.gin_rummy import cards_and_melds as cm
from open_spiel.python.scale.core import plies
from open_spiel.python.scale.envs import gin
from open_spiel.python.scale.envs.tests import gin_cards

tree_map = jax.tree_util.tree_map


def make_info(seat=0, hand="", phase=cm.PHASE_DISCARD, **kw) -> gin.GinInfo:
  """A GinInfo with neutral defaults (no events, nothing laid, stock 20).

  Args:
    seat: The seat.
    hand: The hand's card names.
    phase: The phase.
    **kw: Other GinInfo fields.

  Returns:
    The GinInfo.
  """
  e = gin.EVENTS_MAX
  base = dict(
      seat=jnp.int32(seat),
      hand=jnp.asarray(
          gin_cards.hand_mask(hand) if hand else np.zeros(52, np.int8)
      ),
      discard_pile=jnp.zeros(52, jnp.int8),
      upcard=jnp.int32(-1),
      known_cards=jnp.zeros((2, 52), jnp.int8),
      phase=jnp.int32(phase),
      current_player=jnp.int32(seat),
      knock_card=jnp.int32(10),
      oklahoma=jnp.bool_(True),
      stock_count=jnp.int32(20),
      pass_count=jnp.int32(2),
      num_draw_upcard=jnp.int32(0),
      repeated_move=jnp.bool_(False),
      upcard_taken=jnp.int32(-1),
      allow_upcard_throwback=jnp.bool_(True),
      layed_melds=jnp.zeros((2, cm.NUM_MELDS), jnp.bool_),
      layoffs_mask=jnp.zeros(52, jnp.bool_),
      knocker=jnp.int32(-1),
      finished_layoffs=jnp.bool_(False),
      knocker_gin=jnp.bool_(False),
      done=jnp.bool_(False),
      scores=jnp.zeros(2, jnp.int32),
      ev_actor=jnp.zeros(e, jnp.int8),
      ev_type=jnp.zeros(e, jnp.int8),
      ev_public=jnp.zeros(e, jnp.int8),
      ev_private=jnp.zeros(e, jnp.int8),
      ev_count=jnp.int32(0),
      ev_error=jnp.int32(0),
  )
  for k, v in kw.items():
    assert k in base, k
    base[k] = jnp.asarray(v, base[k].dtype)
  return gin.GinInfo(**base)


def melds_mask(*names) -> np.ndarray:
  """The bool[NUM_MELDS] mask of the named melds.

  Args:
    *names: The melds' card names.

  Returns:
    The mask.
  """
  m = np.zeros(cm.NUM_MELDS, bool)
  for n in names:
    m[gin_cards.meld_id(n)] = True
  return m


def batch(*infos):
  """Stacks InfoStates into a batch.

  Args:
    *infos: The InfoStates.

  Returns:
    The batched InfoState.
  """
  return tree_map(lambda *x: jnp.stack(x), *infos)


def seat_steps_at(env, states, new_hand):
  """SeatStep [B] of each row's actor over batched states.

  Args:
    env: The env.
    states: The batched states.
    new_hand: The new-hand flag.

  Returns:
    The SeatSteps.
  """
  seat = jnp.clip(jax.vmap(env.actor)(states), 0, 1).astype(jnp.int32)
  return plies.seat_step(env, states, seat, new_hand, new_hand, -1, -1)
