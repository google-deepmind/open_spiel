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

"""Phantom tic-tac-toe encoders: classical and abrupt imperfect recall.

Classical: envs.phantom_ttt.PhantomTTTInfo; abrupt imperfect recall:
envs.phantom_ttt_ir.PhantomTTTIRInfo (same view marks).

  phantom_ttt_info_state       f32[110]: information_state_tensor ++
                               one-hot(seat)
  phantom_ttt_observation      f32[29]: the view planes ++ one-hot(seat)
  phantom_ttt_ir_observation   f32[29]: the same, for phantom_ttt_ir
"""

import jax
import jax.numpy as jnp

from open_spiel.python.scale.encoders import registry
from open_spiel.python.scale.envs import phantom_ttt

F32 = jnp.float32


def _view_planes(info: phantom_ttt.PhantomTTTInfo) -> jax.Array:
  """[empty, O, X] over the seat's view, f32[27].

  The layout is OpenSpiel's reveal-nothing observation_tensor.

  Args:
    info: the seat's InfoState.

  Returns:
    f32[27].
  """
  v = info.view
  return jnp.concatenate([
      v == phantom_ttt.EMPTY,
      v == phantom_ttt.P1_MARK,
      v == phantom_ttt.P0_MARK,
  ]).astype(F32)


def info_state(info: phantom_ttt.PhantomTTTInfo) -> jax.Array:
  """OpenSpiel's information_state_tensor with the seat appended.

  The seat is appended because an empty view does not say whether the seat is
  X or O.

  Args:
    info: one (unbatched) InfoState.

  Returns:
    f32[110]: view planes ++ the seat's attempts one-hot (9 x 9, -1 padding as
    zeros) ++ one-hot(seat).
  """
  attempts = jax.nn.one_hot(
      info.own_actions, phantom_ttt.NUM_CELLS, dtype=F32
  ).reshape(-1)
  return jnp.concatenate(
      [_view_planes(info), attempts, jax.nn.one_hot(info.seat, 2, dtype=F32)]
  )


def observation(info: phantom_ttt.PhantomTTTInfo) -> jax.Array:
  """View planes and the seat. Not perfect recall: the attempt order is absent.

  Args:
    info: one (unbatched) InfoState.

  Returns:
    f32[29]: view planes ++ one-hot(seat).
  """
  return jnp.concatenate(
      [_view_planes(info), jax.nn.one_hot(info.seat, 2, dtype=F32)]
  )


INFO_STATE = registry.register(
    registry.Encoder(
        name="phantom_ttt_info_state",
        game="phantom_ttt",
        shape=(110,),
        dtype=F32,
        fn=info_state,
    )
)
OBSERVATION = registry.register(
    registry.Encoder(
        name="phantom_ttt_observation",
        game="phantom_ttt",
        shape=(29,),
        dtype=F32,
        fn=observation,
        reference="OpenSpiel phantom_ttt observation_tensor ++ one-hot(seat)",
    )
)


def ir_observation(info) -> jax.Array:
  """View planes and the seat, for phantom_ttt_ir.

  The game has imperfect recall: the view is the whole information state. The
  planes equal the JAX engine's / OpenSpiel's phantom_ttt_ir
  observation_tensor.

  Args:
    info: one (unbatched) InfoState.

  Returns:
    f32[29]: view planes ++ one-hot(seat).
  """
  return jnp.concatenate(
      [_view_planes(info), jax.nn.one_hot(info.seat, 2, dtype=F32)]
  )


IR_OBSERVATION = registry.register(
    registry.Encoder(
        name="phantom_ttt_ir_observation",
        game="phantom_ttt_ir",
        shape=(29,),
        dtype=F32,
        fn=ir_observation,
    )
)
