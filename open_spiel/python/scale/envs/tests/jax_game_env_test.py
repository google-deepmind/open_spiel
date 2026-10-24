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

"""The JAX-engine adapter.

Wraps games with a current_player() method (kuhn) and games with only a
state.current_player field (leduc), and the leduc wrapper passes check_env
(mid-hand chance: the public card).
"""

import sys

import jax.numpy as jnp
import pytest

from open_spiel.python.jax.games import leduc_poker
from open_spiel.python.scale.envs import jax_game_env
from open_spiel.python.scale.testing import check_env


class LeducMinimal(jax_game_env.JaxGameEnv):
  """Test-only leduc wrapper: info = own card + public card + betting round."""

  name = "leduc_poker"
  max_len = 8
  zero_sum = True

  def __init__(self):
    super().__init__(leduc_poker.LeducPoker())

  def info_state(self, s, seat):
    """What `seat` may know at `s` (the privacy boundary).

    Args:
      s: The state.
      seat: The seat (i32).

    Returns:
      The seat's InfoState; hidden information is absent.
    """
    seat = jnp.asarray(seat, jnp.int32)
    return {
        "card": s.private_cards[seat].astype(jnp.int32),
        "public": jnp.asarray(s.public_card, jnp.int32),
    }


def test_adapter_wraps_leduc():
  """Adapter wraps leduc."""
  check_env.check_env(LeducMinimal(), num_playouts=64)


def test_adapter_requires_name_and_max_len():
  """Adapter requires name and max len."""

  class NoName(jax_game_env.JaxGameEnv):
    """An adapter without a name."""

    max_len = 8

  with pytest.raises(TypeError, match="name"):
    NoName(leduc_poker.LeducPoker())


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
