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

"""Lines of Action encoder over envs.lines_of_action.LinesOfActionInfo.

`lines_of_action_planes`: f32[8, 8, 3] HWC, row 0 = board row 0, planes
[the seat's pieces, the opponent's, all = seat]: the JAX engine's
observation_tensor layout (my, opponent, player plane).
"""

import jax
import jax.numpy as jnp

from open_spiel.python.scale.encoders import registry
from open_spiel.python.scale.envs import lines_of_action as loa

F32 = jnp.float32
R, C = loa.ROWS, loa.COLS


def planes(info: loa.LinesOfActionInfo) -> jax.Array:
  """The seat's pieces, the opponent's and the seat plane, HWC.

  Args:
    info: one (unbatched) InfoState.

  Returns:
    f32[ROWS, COLS, 3]: [the seat's pieces, the opponent's, the seat id].
  """
  b = info.board.reshape(R, C)
  seat_plane = jnp.broadcast_to(info.seat.astype(F32), (R, C))
  return jnp.stack(
      [
          (b == info.seat).astype(F32),
          (b == 1 - info.seat).astype(F32),
          seat_plane,
      ],
      axis=-1,
  )


PLANES = registry.register(
    registry.Encoder(
        name="lines_of_action_planes",
        game="lines_of_action",
        shape=(R, C, 3),
        dtype=F32,
        fn=planes,
    )
)
