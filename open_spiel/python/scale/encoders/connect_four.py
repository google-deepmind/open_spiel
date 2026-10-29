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

"""Connect Four encoders over envs.connect_four.ConnectFourInfo.

  connect_four_planes         f32[6, 7, 2]: [the seat's discs, the opponent's]
  connect_four_player_planes  f32[6, 7, 3]: [the seat's discs, the opponent's,
                              empty]
  connect_four_flat_planes    f32[126]: empty cells (42) ++ the seat's (42) ++
                              the opponent's (42), for MLPs

Rows are board rows with row 0 at the top (the env's layout).
"""

import jax
import jax.numpy as jnp

from open_spiel.python.scale.encoders import registry
from open_spiel.python.scale.envs import connect_four

F32 = jnp.float32
R, C = connect_four.ROWS, connect_four.COLS


def planes(info: connect_four.ConnectFourInfo) -> jax.Array:
  """The seat's and the opponent's discs, HWC.

  Args:
    info: one (unbatched) InfoState.

  Returns:
    f32[6, 7, 2]: [the seat's discs, the opponent's].
  """
  b = info.board.reshape(R, C)
  return jnp.stack([b == info.seat, b == 1 - info.seat], axis=-1).astype(F32)


def player_planes(info: connect_four.ConnectFourInfo) -> jax.Array:
  """The seat's discs, the opponent's and the empty cells, HWC.

  Args:
    info: one (unbatched) InfoState.

  Returns:
    f32[6, 7, 3]: [the seat's discs, the opponent's, empty cells].
  """
  b = info.board.reshape(R, C)
  return jnp.stack(
      [b == info.seat, b == 1 - info.seat, b == -1], axis=-1
  ).astype(F32)


def flat_planes(info: connect_four.ConnectFourInfo) -> jax.Array:
  """Empty cells, the seat's discs and the opponent's, flattened.

  Args:
    info: one (unbatched) InfoState.

  Returns:
    f32[126]: empty ++ the seat's discs ++ the opponent's, each over 42 cells.
  """
  b = info.board
  return jnp.concatenate([
      (b == -1).astype(F32),
      (b == info.seat).astype(F32),
      (b == 1 - info.seat).astype(F32),
  ])


PLANES = registry.register(
    registry.Encoder(
        name="connect_four_planes",
        game="connect_four",
        shape=(R, C, 2),
        dtype=F32,
        fn=planes,
    )
)
PLAYER_PLANES = registry.register(
    registry.Encoder(
        name="connect_four_player_planes",
        game="connect_four",
        shape=(R, C, 3),
        dtype=F32,
        fn=player_planes,
    )
)
FLAT_PLANES = registry.register(
    registry.Encoder(
        name="connect_four_flat_planes",
        game="connect_four",
        shape=(3 * R * C,),
        dtype=F32,
        fn=flat_planes,
    )
)
