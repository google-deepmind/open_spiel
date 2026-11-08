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

"""Euchre encoder over the envs.euchre InfoState.

``euchre_info_state`` is OpenSpiel's (corrected) euchre
information_state_tensor, f32[959] (the games package's euchre module doc for
the layout), rebuilt from one seat's InfoState; it equals that tensor on
every player node of random games (envs/tests/euchre_test.py, against a
pyspiel with the corrected euchre). Seats are absolute (North..West), as in
OpenSpiel; the shape does not depend on the rule options, so one name covers
them all.
"""

import jax
import jax.numpy as jnp

from open_spiel.python.jax.games import euchre as game
from open_spiel.python.scale.encoders import registry
from open_spiel.python.scale.envs import euchre

F32 = jnp.float32
INFO_STATE_NAME = "euchre_info_state"


def info_state(info) -> jax.Array:
  """OpenSpiel's euchre information-state tensor (module doc).

  Args:
    info: one seat's InfoState (envs.euchre).

  Returns:
    f32[959].
  """
  return game.info_state_tensor(
      seat=info["seat"],
      dealer=info["dealer"],
      upcard=info["upcard"],
      num_passes=info["num_passes"],
      trump=info["trump"],
      declarer=info["declarer"],
      go_alone=info["go_alone"],
      lone_defender=info["lone_defender"],
      discard=info["discard"],
      hand=info["hand"],
      play=info["play"],
      play_seat=info["play_seat"],
      n_played=info["n_played"],
  )


INFO_STATE = registry.register(
    registry.Encoder(
        name=INFO_STATE_NAME,
        game=euchre.EuchreEnv.name,
        shape=(game.INFO_STATE_SIZE,),
        dtype=F32,
        fn=info_state,
        reference="OpenSpiel euchre information_state_tensor (corrected, 959)",
    )
)
