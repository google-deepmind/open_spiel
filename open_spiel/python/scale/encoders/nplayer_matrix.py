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

"""Encoder of envs.nplayer_matrix (MatrixInfo), sized by the game's variant.

nplayer_matrix_seat   f32[P]: one_hot(seat, P). A seat's whole InfoState at
    its decision (its own action is not yet played), so a policy over it
    is a mixed strategy per seat.
"""

import jax
import jax.numpy as jnp

from open_spiel.python.scale.encoders import registry
from open_spiel.python.scale.envs import nplayer_matrix

F32 = jnp.float32
GAME = "nplayer_matrix"
SEAT = "nplayer_matrix_seat"


def _member(env_spec) -> registry.Encoder:
  """The seat encoder of an env spec.

  Args:
    env_spec: The EnvSpec.

  Returns:
    The Encoder.
  """
  p = nplayer_matrix.GAMES[env_spec.kwargs["variant"]].num_players
  return registry.Encoder(
      name=registry.family_name(SEAT, env_spec),
      game=GAME,
      shape=(p,),
      dtype=F32,
      fn=lambda info: jax.nn.one_hot(info.seat, p, dtype=F32),
  )


registry.register_family(SEAT, GAME, _member, options=("variant",))
