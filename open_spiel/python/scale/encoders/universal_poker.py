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

"""Heads-up no-limit hold'em encoders over envs.universal_poker.HoldemInfo."""

import jax
import jax.numpy as jnp
import numpy as np

from open_spiel.python.scale.encoders import registry
from open_spiel.python.scale.envs import universal_poker as up

F32 = jnp.float32
# Chip amounts are scaled by multiplying with 1 / STACK (never dividing: XLA
# rewrites a division by a constant under jit, and jit must equal eager).
_INV_STACK = np.float32(1.0 / up.STACK)
# Per action slot: [fold, call/check, raise] one-hot ++ raise-to / STACK
# ++ round one-hot (4); padding is all zeros.
SLOT = 3 + 1 + up.NUM_ROUNDS
DIM = 2 + 2 * up.NUM_CARDS + up.NUM_ROUNDS + 2 + up.MAX_LEN * SLOT  # 968


def _cards_one_hot(cards: jax.Array) -> jax.Array:
  """f32[52]: 1 at each dealt card (-1 padding encodes nothing).

  Args:
    cards: The cards, -1 for padding.

  Returns:
    f32[52], the dealt cards.
  """
  return jnp.sum(jax.nn.one_hot(cards, up.NUM_CARDS, dtype=F32), axis=0)


def info_state(info: up.HoldemInfo) -> jax.Array:
  """The perfect-recall heads-up hold'em encoding.

  Args:
    info: one (unbatched) InfoState.

  Returns:
    f32[968]:

      one_hot(seat, 2) ++ own hole cards (52, multi-hot) ++ board (52,
      multi-hot) ++ one_hot(round, 4) ++ spent * (1/200) (2)
      ++ per player action slot (107 x 8): one_hot(kind: fold, call, raise)
      ++ raise-to * (1/200) (0 unless a raise) ++ one_hot(round of the
      action, 4).

    Slots past num_actions are zeros. Every value lies in [0, 1].
  """
  acts = jnp.asarray(info.actions, jnp.int32)
  kind = jnp.where(acts >= up.MIN_RAISE, 2, acts)  # -1 pad stays -1 -> zeros
  size = jnp.where(
      acts >= up.MIN_RAISE, acts.astype(F32) * _INV_STACK, F32(0.0)
  )
  slots = jnp.concatenate(
      [
          jax.nn.one_hot(kind, 3, dtype=F32),
          size[:, None],
          jax.nn.one_hot(info.action_rounds, up.NUM_ROUNDS, dtype=F32),
      ],
      axis=-1,
  )
  return jnp.concatenate([
      jax.nn.one_hot(info.seat, 2, dtype=F32),
      _cards_one_hot(info.hole),
      _cards_one_hot(info.board),
      jax.nn.one_hot(info.round, up.NUM_ROUNDS, dtype=F32),
      jnp.asarray(info.spent, F32) * _INV_STACK,
      slots.reshape(-1),
  ])


INFO_STATE = registry.register(
    registry.Encoder(
        name="universal_poker_info_state",
        game="universal_poker",
        shape=(DIM,),
        dtype=F32,
        fn=info_state,
    )
)
