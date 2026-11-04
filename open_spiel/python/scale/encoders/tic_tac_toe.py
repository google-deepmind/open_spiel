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

"""Tic-tac-toe encoders over envs.tic_tac_toe.TicTacToeInfo.

tic_tac_toe_planes               f32[3, 3, 2]: [the seat's marks, the
                                 opponent's]
tic_tac_toe_observation_tensor   f32[27]: the JAX engine's tic_tac_toe
                                 observation_tensor
(HWC: mine, theirs, seat plane), ravelled; not
OpenSpiel's cell-state layout
"""

import jax
import jax.numpy as jnp

from open_spiel.python.scale.encoders import registry
from open_spiel.python.scale.envs import tic_tac_toe

F32 = jnp.float32


def planes(info: tic_tac_toe.TicTacToeInfo) -> jax.Array:
  """The seat's marks and the opponent's, HWC.

  Args:
    info: one (unbatched) InfoState.

  Returns:
    f32[3, 3, 2]: [the seat's marks, the opponent's marks].
  """
  b = info.board.reshape(3, 3)
  return jnp.stack([b == info.seat, b == 1 - info.seat], axis=-1).astype(F32)


def observation_tensor(info: tic_tac_toe.TicTacToeInfo) -> jax.Array:
  """The JAX engine's tic_tac_toe observation tensor, ravelled.

  Args:
    info: one (unbatched) InfoState.

  Returns:
    f32[27]: 3x3x3 HWC: mine, theirs, a plane filled with the seat id.
  """
  b = info.board.reshape(3, 3)
  seat_plane = jnp.full((3, 3), info.seat.astype(F32), F32)
  return jnp.stack(
      [
          (b == info.seat).astype(F32),
          (b == 1 - info.seat).astype(F32),
          seat_plane,
      ],
      axis=-1,
  ).reshape(-1)


PLANES = registry.register(
    registry.Encoder(
        name="tic_tac_toe_planes",
        game="tic_tac_toe",
        shape=(3, 3, 2),
        dtype=F32,
        fn=planes,
    )
)
OBSERVATION_TENSOR = registry.register(
    registry.Encoder(
        name="tic_tac_toe_observation_tensor",
        game="tic_tac_toe",
        shape=(27,),
        dtype=F32,
        fn=observation_tensor,
        reference="tic_tac_toe observation_tensor of the JAX engine, ravelled",
    )
)
