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

"""Board planes for breakthrough, gomoku, hex and othello.

`<game>_planes`: f32[H, W, 3] HWC, [the seat's pieces, the opponent's pieces,
a plane filled with the seat id], in the absolute board orientation for both
seats. The envs' actions are absolute cells, so a policy head's cells line up
with the planes. The seat plane matters: in hex the seats join different
edges, in breakthrough they move in opposite directions.

The shape depends on the board size, so each game's planes are a family over
its size options: "gomoku_planes" is the default 15x15 board and
"gomoku_planes@board_size=9" the 9x9 member. One encoding has one name;
`planes_name(env)` names an env's encoder.
"""

import jax
import jax.numpy as jnp

from open_spiel.python.scale.encoders import registry
from open_spiel.python.scale.envs import registry as env_registry

F32 = jnp.float32

# game -> the EnvSpec options that size its board.
SIZE_OPTIONS = {
    "breakthrough": ("rows", "columns"),
    "gomoku": ("board_size",),
    "hex": ("board_size",),
    "othello": (),
}
GAMES = tuple(SIZE_OPTIONS)


def _board_shape(game: str, options) -> tuple[int, int]:
  """The (rows, columns) of a board game.

  Args:
    game: The game.
    options: Its options.

  Returns:
    (rows, columns).

  Raises:
    KeyError: The game has no board planes encoder.
  """
  if game == "breakthrough":
    return int(options["rows"]), int(options["columns"])
  if game in ("gomoku", "hex"):
    return int(options["board_size"]), int(options["board_size"])
  if game == "othello":
    return 8, 8
  raise KeyError(f"no board planes encoder for game {game!r}")


def planes_name(env) -> str:
  """The obs_format of the planes encoder of an env of one of these games.

  Args:
    env: the Env.

  Returns:
    The family member's name.

  Raises:
    KeyError: the game has no board planes encoder.
  """
  if env.name not in SIZE_OPTIONS:
    raise KeyError(f"no board planes encoder for game {env.name!r}")
  return registry.family_name(f"{env.name}_planes", env_registry.spec_of(env))


def _planes_fn(rows: int, cols: int):
  """The planes function of a rows x cols board.

  Args:
    rows: the board's rows.
    cols: the board's columns.

  Returns:
    InfoState -> f32[rows, cols, 3].
  """

  def planes(info) -> jax.Array:
    """[the seat's stones, the opponent's, the seat plane] over the board.

    Args:
      info: one (unbatched) InfoState.

    Returns:
      f32[rows, cols, 3].
    """
    b = jnp.asarray(info.board, jnp.int32).reshape(rows, cols)
    seat = jnp.asarray(info.seat, jnp.int32)
    return jnp.stack(
        [b == seat, b == 1 - seat, jnp.full((rows, cols), seat == 1)], axis=-1
    ).astype(F32)

  return planes


def _family(game: str):
  """The family member factory of `game`'s planes encoders.

  Args:
    game: the game name.

  Returns:
    EnvSpec -> the planes Encoder of the game at those options.
  """

  def member(env_spec) -> registry.Encoder:
    """The planes encoder of the game at `env_spec`'s options.

    Args:
      env_spec: the EnvSpec.

    Returns:
      The family member Encoder.
    """
    r, c = _board_shape(game, env_spec.kwargs)
    return registry.Encoder(
        name=registry.family_name(f"{game}_planes", env_spec),
        game=game,
        shape=(r, c, 3),
        dtype=F32,
        fn=_planes_fn(r, c),
    )

  return member


for _game, _options in SIZE_OPTIONS.items():
  registry.register_family(
      f"{_game}_planes", _game, _family(_game), options=_options
  )
