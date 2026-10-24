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

"""Bargaining encoders over envs.bargaining.BargainingInfo.

  bargaining_info_state   f32[311] at max_turns 10: OpenSpiel's
                          information_state_tensor ++ one-hot(seat)

The width depends on max_turns, so the encoder is a family over that option:
"bargaining_info_state@max_turns=12" encodes a 12-turn game.
"""

import functools

import jax
import jax.numpy as jnp

from open_spiel.python.scale.encoders import registry
from open_spiel.python.scale.envs import bargaining

F32 = jnp.float32
NAME = "bargaining_info_state"
MAX_TURNS = bargaining.DEFAULT_MAX_TURNS
TOTAL_VALUE = 10  # each seat's values over the pool sum to 10
_ITEM_W = bargaining.POOL_MAX_NUM_ITEMS + 1  # 8


def width(max_turns: int) -> int:
  """The encoding width at `max_turns`.

  Args:
    max_turns: the game's max_turns option.

  Returns:
    (max_turns + 2) + 3 * 8 + 3 * 11 + max_turns * 3 * 8 + 2.
  """
  return (
      (max_turns + 2)
      + 3 * _ITEM_W
      + 3 * (TOTAL_VALUE + 1)
      + max_turns * 3 * _ITEM_W
      + 2
  )


WIDTH = width(MAX_TURNS)  # 311


def _thermometer(x: jax.Array, top: int) -> jax.Array:
  """f32[..., top + 1]: bit j is x >= j (x = -1 encodes as zeros).

  Args:
    x: The values, -1 for none.
    top: The largest value.

  Returns:
    f32[..., top + 1], the bits.
  """
  return (jnp.arange(top + 1, dtype=jnp.int32) <= x[..., None]).astype(F32)


def info_state(
    info: bargaining.BargainingInfo, max_turns: int = MAX_TURNS
) -> jax.Array:
  """The bargaining information state, with the seat appended.

  The seat is appended because the tensor names the seat only through whose
  turn it is.

  Args:
    info: one (unbatched) InfoState.
    max_turns: the game's max_turns option.

  Returns:
    f32[width(max_turns)]: one-hot(num_offers) over max_turns + 2, pool
    thermometers 3 x 8, own value thermometers 3 x 11, every offer slot's
    thermometers max_turns x 3 x 8 (zeros for offers not yet made), then
    one-hot(seat).

  Raises:
    ValueError: the offers do not match `max_turns`.
  """
  if info.offers.shape != (max_turns, 3):
    raise ValueError(
        f"{NAME} for max_turns={max_turns} got offers {info.offers.shape}"
    )
  return jnp.concatenate([
      jax.nn.one_hot(info.num_offers, max_turns + 2, dtype=F32),
      _thermometer(info.pool, bargaining.POOL_MAX_NUM_ITEMS).reshape(-1),
      _thermometer(info.values, TOTAL_VALUE).reshape(-1),
      _thermometer(info.offers, bargaining.POOL_MAX_NUM_ITEMS).reshape(-1),
      jax.nn.one_hot(info.seat, 2, dtype=F32),
  ])


_REFERENCE = "OpenSpiel bargaining information_state_tensor ++ one-hot(seat)"


def _member(env_spec) -> registry.Encoder:
  """The encoder of one bargaining variant.

  Args:
    env_spec: The env spec.

  Returns:
    The Encoder.
  """
  t = int(env_spec.kwargs["max_turns"])
  return registry.Encoder(
      name=registry.family_name(NAME, env_spec),
      game="bargaining",
      shape=(width(t),),
      dtype=F32,
      fn=functools.partial(info_state, max_turns=t),
      reference=_REFERENCE,
  )


INFO_STATE = registry.register(
    registry.Encoder(
        name=NAME,
        game="bargaining",
        shape=(WIDTH,),
        dtype=F32,
        fn=info_state,
        reference=_REFERENCE,
    )
)
registry.register_family(NAME, "bargaining", _member, options=("max_turns",))
