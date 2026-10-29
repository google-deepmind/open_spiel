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

"""Liar's dice encoders over envs.liars_dice.LiarsDiceInfo."""

import jax
import jax.numpy as jnp

from open_spiel.python.scale.encoders import registry
from open_spiel.python.scale.envs import liars_dice

F32 = jnp.float32
# 33
WIDTH = 2 + liars_dice.NUM_DICE * liars_dice.DICE_SIDES + liars_dice.NUM_ACTIONS


def info_state(info: liars_dice.LiarsDiceInfo) -> jax.Array:
  """OpenSpiel's (and the JAX engine's) liars_dice information_state_tensor.

  Args:
    info: one (unbatched) InfoState.

  Returns:
    f32[33], perfect recall: one-hot(seat) ++ per own die one-hot(face - 1) ++
    the set of actions taken so far (bids strictly increase, so the set is the
    sequence).
  """
  dice = jax.nn.one_hot(
      info.dice - 1, liars_dice.DICE_SIDES, dtype=F32
  ).reshape(-1)
  taken = jnp.max(
      jax.nn.one_hot(info.bids, liars_dice.NUM_ACTIONS, dtype=F32), axis=0
  )
  return jnp.concatenate([jax.nn.one_hot(info.seat, 2, dtype=F32), dice, taken])


INFO_STATE = registry.register(
    registry.Encoder(
        name="liars_dice_info_state",
        game="liars_dice",
        shape=(WIDTH,),
        dtype=F32,
        fn=info_state,
    )
)
