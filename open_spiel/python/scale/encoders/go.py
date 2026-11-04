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

"""Go encoders over envs.go.GoInfo.

`go_planes`: f32[N, N, 6] HWC, row-major board rows, planes
  0 the seat's stones, 1 the opponent's, 2 empty,
  3 the ko point (one-hot; all zero when there is none),
  4 all ones when the seat is black (komi makes the colours unequal),
  5 all ones when the previous move was a pass (one more pass ends the game).

The shape depends on the board size, so `go_planes` is a family over
board_size ("go_planes" is 9x9, the env's default; "go_planes@board_size=19"
is 19x19). One encoding has one name.
"""

import jax
import jax.numpy as jnp

from open_spiel.python.scale.encoders import registry
from open_spiel.python.scale.envs import go

F32 = jnp.float32
NUM_PLANES = 6
FAMILY = "go_planes"


def planes(info: go.GoInfo, board_size: int) -> jax.Array:
  """The go planes at `board_size`, HWC.

  Args:
    info: one (unbatched) InfoState.
    board_size: the board size n.

  Returns:
    f32[n, n, 6]: [mine, theirs, empty, ko, the seat is black, passed].

  Raises:
    ValueError: the board is not n x n.
  """
  n = board_size
  if info.board.shape != (n * n,):
    raise ValueError(
        f"go planes for {n}x{n} got a board of shape {info.board.shape}"
    )
  b = info.board.reshape(n, n)
  sign = jnp.where(info.seat == 0, 1, -1).astype(jnp.int32)
  mine = b * sign > 0
  theirs = b * sign < 0
  empty = b == 0
  ko = (jnp.arange(n * n, dtype=jnp.int32) == info.ko_point).reshape(n, n)
  black = jnp.broadcast_to(info.seat == 0, (n, n))
  passed = jnp.broadcast_to(info.consecutive_passes > 0, (n, n))
  return jnp.stack([mine, theirs, empty, ko, black, passed], axis=-1).astype(
      F32
  )


def _encoder(name: str, n: int) -> registry.Encoder:
  """The go planes encoder of an n x n board.

  Args:
    name: The encoder name.
    n: The board size.

  Returns:
    The Encoder.
  """
  return registry.Encoder(
      name=name,
      game="go",
      shape=(n, n, NUM_PLANES),
      dtype=F32,
      fn=lambda info: planes(info, n),
  )


def _member(env_spec) -> registry.Encoder:
  """The go encoder of one board size.

  Args:
    env_spec: The env spec.

  Returns:
    The Encoder.
  """
  n = int(env_spec.kwargs["board_size"])
  return _encoder(registry.family_name(FAMILY, env_spec), n)


registry.register_family(FAMILY, "go", _member, options=("board_size",))
