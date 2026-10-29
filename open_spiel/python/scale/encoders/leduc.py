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

"""Leduc poker encoders over envs.leduc.LeducInfo.

leduc_poker_info_state           f32[30]: OpenSpiel's information_state_tensor
leduc_poker_observation_tensor   f32[16]: OpenSpiel's observation_tensor (not
perfect recall: the betting sequence is absent)
"""

import jax
import jax.numpy as jnp

from open_spiel.python.scale.encoders import registry
from open_spiel.python.scale.envs import leduc

F32 = jnp.float32
_CALL, _RAISE = 1, 2


def _header(info: leduc.LeducInfo) -> jax.Array:
  """Seat one-hot (2) ++ own card one-hot (6) ++ public card one-hot (6).

  A card of -1 (not dealt) encodes as zeros.

  Args:
    info: the seat's InfoState.

  Returns:
    f32[14].
  """
  return jnp.concatenate([
      jax.nn.one_hot(info.seat, 2, dtype=F32),
      jax.nn.one_hot(info.card, leduc.NUM_CARDS, dtype=F32),
      jax.nn.one_hot(info.public_card, leduc.NUM_CARDS, dtype=F32),
  ])


def info_state(info: leduc.LeducInfo) -> jax.Array:
  """OpenSpiel's leduc information_state_tensor (perfect recall).

  Args:
    info: one (unbatched) InfoState.

  Returns:
    f32[30]: the header, then per betting slot of round 1 and round 2 the bits
    [call, raise] (fold and padding are [0, 0]).
  """

  def bits(seq):
    """The [call, raise] bits of one round's betting slots.

    Args:
      seq: i32[slots], the round's action sequence.

    Returns:
      f32[2 * slots].
    """
    return (
        jnp.stack([seq == _CALL, seq == _RAISE], axis=-1)
        .astype(F32)
        .reshape(-1)
    )

  return jnp.concatenate([_header(info), bits(info.round1), bits(info.round2)])


def observation_tensor(info: leduc.LeducInfo) -> jax.Array:
  """The header and both seats' chips in. Not perfect recall.

  Args:
    info: one (unbatched) InfoState.

  Returns:
    f32[16]: the header ++ both seats' chips in (ante).
  """
  return jnp.concatenate([_header(info), info.ante.astype(F32)])


INFO_STATE = registry.register(
    registry.Encoder(
        name="leduc_poker_info_state",
        game="leduc_poker",
        shape=(30,),
        dtype=F32,
        fn=info_state,
        reference="OpenSpiel leduc_poker information_state_tensor",
    )
)
OBSERVATION_TENSOR = registry.register(
    registry.Encoder(
        name="leduc_poker_observation_tensor",
        game="leduc_poker",
        shape=(16,),
        dtype=F32,
        fn=observation_tensor,
        reference="OpenSpiel leduc_poker observation_tensor",
    )
)
