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

"""gin_rummy_observation_tensor: OpenSpiel gin_rummy's default observation.

Computed from GinInfo. 644 floats, equal to pyspiel's (and the JAX engine's
`GinRummy.observation_tensor`)
at the default rule options:
  [0:2]     player plane (the observing seat)
  [2:106]   private hands: the seat's own hand in its 52-block, the other zero
  [106:108] current player plane (zero at chance / terminal)
  [108:118] knock card thermometer (i < knock_card)
  [118:170] upcard one-hot
  [170:222] discard pile
  [222:274] stock size thermometer (i < stock count)
  [274:644] laid melds, seat 0's 185 then seat 1's
A buffer error in the InfoState's transcript poisons the vector (all NaN),
so it cannot pass silently.
"""

import jax.numpy as jnp

from open_spiel.python.jax.games.gin_rummy import cards_and_melds as cm
from open_spiel.python.scale.encoders import registry

NAME = "gin_rummy_observation_tensor"
DIM = 644


def encode(info) -> jnp.ndarray:
  """The gin observation tensor; NaN where the engine reported an error.

  Args:
    info: one (unbatched) InfoState.

  Returns:
    f32[644], NaN everywhere on an engine error.
  """
  i32 = jnp.int32
  seat = info.seat.astype(i32)
  players = jnp.arange(2, dtype=i32)
  hand = info.hand.astype(jnp.float32)
  zeros = jnp.zeros(cm.NUM_CARDS, jnp.float32)
  cp = info.current_player.astype(i32)
  cards = jnp.arange(cm.NUM_CARDS, dtype=i32)
  planes = [
      (players == seat).astype(jnp.float32),
      jnp.where(seat == 0, hand, zeros),
      jnp.where(seat == 1, hand, zeros),
      ((players == cp) & (cp >= 0)).astype(jnp.float32),
      (
          jnp.arange(cm.KNOCK_THRESHOLD, dtype=i32)
          < info.knock_card.astype(i32)
      ).astype(jnp.float32),
      ((cards == info.upcard.astype(i32)) & (info.upcard >= 0)).astype(
          jnp.float32
      ),
      (info.discard_pile > 0).astype(jnp.float32),
      (cards < info.stock_count.astype(i32)).astype(jnp.float32),
      info.layed_melds.reshape(2 * cm.NUM_MELDS).astype(jnp.float32),
  ]
  x = jnp.concatenate(planes)
  return jnp.where(info.ev_error != 0, jnp.float32(jnp.nan), x)


registry.register(
    registry.Encoder(
        name=NAME,
        game="gin_rummy",
        shape=(DIM,),
        dtype=jnp.float32,
        fn=encode,
        reference="OpenSpiel gin_rummy observation_tensor (default observer)",
    )
)
