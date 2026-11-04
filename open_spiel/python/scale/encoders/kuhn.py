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

"""Kuhn poker encoders over KuhnInfo, registered for both Kuhn games.

  kuhn_poker_observation_tensor  f32[7]: OpenSpiel's kuhn_poker observation
                                 tensor
  kuhn_poker_info_index          f32[12]: one input per decision information
                                 state

The two games are one set of rules: "kuhn_poker" (envs.kuhn) adapts the JAX
games package's engine; "kuhn_poker_jax" (envs.kuhn_jax) is a self-contained
implementation that needs no game package. They share KuhnInfo, so each
encoding is registered once per game, as kuhn_poker_<x> and
kuhn_poker_jax_<x>.
"""

import jax
import jax.numpy as jnp

from open_spiel.python.scale.encoders import registry
from open_spiel.python.scale.envs import kuhn_jax

F32 = jnp.float32


def observation_tensor(info: kuhn_jax.KuhnInfo) -> jax.Array:
  """OpenSpiel's kuhn_poker observation tensor.

  The chips in tell one seat's betting histories apart, so in Kuhn poker this
  observation is also a perfect-recall information state.

  Args:
    info: one (unbatched) InfoState.

  Returns:
    f32[7]: one_hot(seat, 2) ++ one_hot(own card, 3) ++ each seat's chips in
    (ante 1 + its bets).
  """
  idx = jnp.arange(kuhn_jax.MAX_BETS, dtype=jnp.int32)
  active = idx < info.num_bets
  is_bet = jnp.asarray(info.bets, jnp.int32) == 1
  by_p0 = (idx % 2) == 0
  pot0 = jnp.int32(1) + jnp.sum((is_bet & by_p0 & active).astype(jnp.int32))
  pot1 = jnp.int32(1) + jnp.sum((is_bet & ~by_p0 & active).astype(jnp.int32))
  card = jnp.asarray(info.card, jnp.int32)
  card_oh = jnp.where(
      card >= 0,
      jax.nn.one_hot(jnp.maximum(card, 0), kuhn_jax.NUM_CARDS, dtype=F32),
      jnp.zeros((kuhn_jax.NUM_CARDS,), F32),
  )
  return jnp.concatenate([
      jax.nn.one_hot(info.seat, 2, dtype=F32),
      card_oh,
      jnp.stack([pot0, pot1]).astype(F32),
  ])


def info_index_onehot(info: kuhn_jax.KuhnInfo) -> jax.Array:
  """One input per decision information state.

  Args:
    info: one (unbatched) InfoState.

  Returns:
    f32[12]: one_hot(kuhn_jax.info_index).
  """
  return jax.nn.one_hot(
      kuhn_jax.info_index(info), kuhn_jax.NUM_INFO_STATES, dtype=F32
  )


_REFERENCE = "OpenSpiel kuhn_poker observation_tensor"
OBSERVATION_TENSOR = registry.register(
    registry.Encoder(
        name="kuhn_poker_observation_tensor",
        game="kuhn_poker",
        shape=(7,),
        dtype=F32,
        fn=observation_tensor,
        reference=_REFERENCE,
    )
)
INFO_INDEX = registry.register(
    registry.Encoder(
        name="kuhn_poker_info_index",
        game="kuhn_poker",
        shape=(kuhn_jax.NUM_INFO_STATES,),
        dtype=F32,
        fn=info_index_onehot,
    )
)
OBSERVATION_JAX = registry.register(
    registry.Encoder(
        name="kuhn_poker_jax_observation_tensor",
        game="kuhn_poker_jax",
        shape=(7,),
        dtype=F32,
        fn=observation_tensor,
        reference=_REFERENCE,
    )
)
INFO_INDEX_JAX = registry.register(
    registry.Encoder(
        name="kuhn_poker_jax_info_index",
        game="kuhn_poker_jax",
        shape=(kuhn_jax.NUM_INFO_STATES,),
        dtype=F32,
        fn=info_index_onehot,
    )
)
