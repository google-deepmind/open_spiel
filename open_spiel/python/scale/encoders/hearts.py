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

"""Hearts encoder over the envs.hearts InfoState.

``hearts_info_state`` is OpenSpiel hearts' information_state_tensor, f32[5088]
(the games package's hearts module doc for the layout), rebuilt from one seat's
InfoState; it equals pyspiel's tensor on every player node of random games
(envs/tests/hearts_test.py). Seats are absolute (North..West), as in pyspiel;
the shape does not depend on the rule options, so one name covers them all.
"""

import jax
import jax.numpy as jnp

from open_spiel.python.jax.games import hearts as game
from open_spiel.python.scale.encoders import registry
from open_spiel.python.scale.envs import hearts

F32 = jnp.float32
INFO_STATE_NAME = "hearts_info_state"


def info_state(info) -> jax.Array:
  """Pyspiel's hearts information-state tensor (module doc).

  Args:
    info: one seat's InfoState (envs.hearts).

  Returns:
    f32[5088].
  """
  return game.info_state_tensor(
      pass_dir=info["pass_dir"],
      dealt=info["dealt"],
      passed=info["passed"],
      received=info["received"],
      hand=info["hand"],
      points=info["points"],
      play=info["play"],
      play_seat=info["play_seat"],
      n_played=info["n_played"],
      live=True,
  )


INFO_STATE = registry.register(
    registry.Encoder(
        name=INFO_STATE_NAME,
        game=hearts.HeartsEnv.name,
        shape=(game.INFO_STATE_SIZE,),
        dtype=F32,
        fn=info_state,
        reference="OpenSpiel hearts information_state_tensor",
    )
)
