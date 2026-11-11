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

"""Arena test fixtures: a test-only leduc env with mid-hand chance."""

import jax.numpy as jnp

from open_spiel.python.jax.games import leduc_poker
from open_spiel.python.scale.envs import jax_game_env


class Leduc(jax_game_env.JaxGameEnv):
  """Test-only leduc (mid-hand chance): info = own card + public card."""

  name, max_len, zero_sum = "leduc_poker", 8, True

  def __init__(self):
    super().__init__(leduc_poker.LeducPoker())

  def info_state(self, s, seat):
    """The seat's own card and the public card.

    Args:
      s: the state.
      seat: the seat.

    Returns:
      {"card", "public"} as int32.
    """
    return {
        "card": s.private_cards[seat].astype(jnp.int32),
        "public": jnp.asarray(s.public_card, jnp.int32),
    }

  def labels(self, s, seat):
    """Both private cards and the public card.

    Args:
      s: the state.
      seat: the seat.

    Returns:
      {"private", "public"} as int32.
    """
    del seat
    return {
        "private": s.private_cards.astype(jnp.int32),
        "public": jnp.asarray(s.public_card, jnp.int32),
    }
