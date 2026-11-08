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

"""3x3 Dark Hex encoders over envs.dark_hex InfoStates.

Classical (`dark_hex`):
  dark_hex_info_state              f32[164]: OpenSpiel's reveal-nothing
                                   information_state_tensor (view one-hot over
                                   9 cell states, 81 ++ own attempts one-hot,
                                   9 x 9) ++ one-hot(seat)
  dark_hex_observation_tensor      f32[81]: the view one-hot, OpenSpiel's
                                   observation_tensor
Abrupt, imperfect recall (`dark_hex_ir`):
  dark_hex_ir_view                 f32[29]: [empty, black, white] over the
                                   seat's view (27) ++ one-hot(seat)
  dark_hex_ir_observation_tensor   f32[81]: OpenSpiel's dark_hex_ir
                                   observation_tensor at decision nodes (cell
                                   index white 3, empty 4, black 5). At a
                                   terminal state OpenSpiel marks the winning
                                   stone (index 0 / 8), which the IR InfoState
                                   does not carry; a policy never acts there.
"""

import jax
import jax.numpy as jnp

from open_spiel.python.scale.encoders import registry
from open_spiel.python.scale.envs import dark_hex

F32 = jnp.float32
N = dark_hex.NUM_CELLS
CELL_STATES = 9  # OpenSpiel kCellStates
MIN_CELL_STATE = -4  # OpenSpiel kMinValueCellState


def _view_onehot(info: dark_hex.DarkHexInfo) -> jax.Array:
  """The one-hot cell states of the player's view.

  Args:
    info: The DarkHexInfo.

  Returns:
    f32[CELLS * CELL_STATES].
  """
  return jax.nn.one_hot(
      info.view - MIN_CELL_STATE, CELL_STATES, dtype=F32
  ).reshape(-1)


def info_state(info: dark_hex.DarkHexInfo) -> jax.Array:
  # -1 pad -> zeros
  """The dark hex information state: view, attempts and seat.

  Args:
    info: one (unbatched) InfoState.

  Returns:
    The view one-hot ++ the attempts ++ one-hot(seat), f32.
  """
  attempts = jax.nn.one_hot(info.own_actions, N, dtype=F32).reshape(-1)
  return jnp.concatenate(
      [_view_onehot(info), attempts, jax.nn.one_hot(info.seat, 2, dtype=F32)]
  )


def observation_tensor(info: dark_hex.DarkHexInfo) -> jax.Array:
  """The dark hex observation: the seat's view, one-hot per cell.

  Args:
    info: one (unbatched) InfoState.

  Returns:
    The view one-hot, f32.
  """
  return _view_onehot(info)


def ir_view(info: dark_hex.DarkHexIRInfo) -> jax.Array:
  """The imperfect-recall view: cell states and the seat.

  Args:
    info: one (unbatched) imperfect-recall InfoState.

  Returns:
    [empty, black, white] over the view ++ the seat one-hot, f32.
  """
  v = info.view
  return jnp.concatenate([
      v == dark_hex.EMPTY,
      v == dark_hex.BLACK,
      v == dark_hex.WHITE,
      jax.nn.one_hot(info.seat, 2, dtype=jnp.bool_),
  ]).astype(F32)


def ir_observation_tensor(info: dark_hex.DarkHexIRInfo) -> jax.Array:
  """The imperfect-recall observation tensor, one-hot per cell.

  Args:
    info: one (unbatched) imperfect-recall InfoState.

  Returns:
    One-hot(cell state) over the cells, flattened, f32.
  """
  v = info.view
  idx = jnp.where(v == dark_hex.BLACK, 5, jnp.where(v == dark_hex.WHITE, 3, 4))
  return jax.nn.one_hot(idx, CELL_STATES, dtype=F32).reshape(-1)


INFO_STATE = registry.register(
    registry.Encoder(
        name="dark_hex_info_state",
        game="dark_hex",
        shape=(N * CELL_STATES + N * N + 2,),
        dtype=F32,
        fn=info_state,
    )
)
OBSERVATION_TENSOR = registry.register(
    registry.Encoder(
        name="dark_hex_observation_tensor",
        game="dark_hex",
        shape=(N * CELL_STATES,),
        dtype=F32,
        fn=observation_tensor,
        reference="OpenSpiel dark_hex observation_tensor",
    )
)
IR_VIEW = registry.register(
    registry.Encoder(
        name="dark_hex_ir_view",
        game="dark_hex_ir",
        shape=(3 * N + 2,),
        dtype=F32,
        fn=ir_view,
    )
)
IR_OBSERVATION_TENSOR = registry.register(
    registry.Encoder(
        name="dark_hex_ir_observation_tensor",
        game="dark_hex_ir",
        shape=(N * CELL_STATES,),
        dtype=F32,
        fn=ir_observation_tensor,
        reference="OpenSpiel dark_hex_ir observation_tensor at decision nodes",
    )
)
