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

"""Which pyspiel games are served, and the Env that plays given game parameters.

A view of the one game table (envs.registry): a game is served when its
GameSpec has a ServingSpec, which lists the pyspiel parameters it accepts
(pyspiel_params); its env is built from served_kwargs(params). Game parameters
are checked against what the env implements and anything else is refused:
an engine must never answer a variant it does not play. Parameters that only
change pyspiel's observation tensors (the env builds its own InfoState) are
accepted with any value (envs.registry.ANY). Whether a given ARTIFACT may play
the env is a separate check (envs.registry.check_serves, run by HostAgent).
"""

from typing import Any, Mapping

from open_spiel.python.scale.core import types
from open_spiel.python.scale.envs import registry as env_registry

ANY = env_registry.ANY


def served_games() -> tuple[str, ...]:
  """The pyspiel games an env serves.

  Returns:
    The game names.
  """
  return tuple(g.name for g in env_registry.games() if g.served)


def check_params(game: str, params: Mapping[str, Any]) -> None:
  """Checks that `game` is served and plays every one of `params`.

  Args:
    game: the pyspiel game's short name.
    params: pyspiel game parameters.

  Raises:
    KeyError: `game` is not served (or not registered).
    ValueError: a parameter the env does not implement, or a value it does
      not play.
  """
  serving = env_registry.game(game).serving
  if serving is None:
    raise KeyError(f"no env for game {game!r}; served: {served_games()}")
  allowed = serving.pyspiel_params
  for k, v in params.items():
    if k not in allowed:
      raise ValueError(
          f"{game}: game parameter {k}={v!r} is not implemented by the env"
      )
    if allowed[k] is not ANY and v not in allowed[k]:
      raise ValueError(
          f"{game}: game parameter {k}={v!r}; the env plays "
          f"{k} in {list(allowed[k])}"
      )


def env_spec_for(game: str, params: Mapping[str, Any]) -> env_registry.EnvSpec:
  """The EnvSpec that plays pyspiel `game` with parameters `params`.

  Args:
    game: the pyspiel game's short name.
    params: pyspiel game parameters.

  Returns:
    The EnvSpec.

  Raises:
    KeyError: `game` is not served.
  """
  if game not in served_games():
    raise KeyError(f"no env for game {game!r}; served: {served_games()}")
  check_params(game, params)
  g = env_registry.game(game)
  return env_registry.spec(game, **g.serving.served_kwargs(dict(params)))


def env_for(game: str, params: Mapping[str, Any]) -> types.Env:
  """The Env for `game` with pyspiel game parameters `params`.

  Args:
    game: the pyspiel game's short name.
    params: pyspiel game parameters.

  Returns:
    The Env.
  """
  env = env_registry.make(env_spec_for(game, params))
  assert env.name == game, (env.name, game)
  return env


def _same(a: Any, b: Any) -> bool:
  """Whether two parameter values are equal, a bool equal only to a bool.

  Args:
    a: A value.
    b: A value.

  Returns:
    True when equal.
  """
  if isinstance(a, bool) or isinstance(b, bool):
    return isinstance(a, bool) and isinstance(b, bool) and a == b
  return a == b


def game_string_params(
    game: str,
    params: Mapping[str, Any],
    parameter_specification: Mapping[str, Any],
) -> dict[str, Any]:
  """`params` as a pyspiel whose `game` has `parameter_specification` takes.

  A parameter at pyspiel's default is dropped (the same game); the others
  are kept.

  Args:
    game: the game name.
    params: the game parameters.
    parameter_specification: {parameter: default} of the target pyspiel game.

  Returns:
    The parameters to write.

  Raises:
    ValueError: a parameter the pyspiel game does not have.
  """
  out = {}
  for k in sorted(params):
    v = params[k]
    if k not in parameter_specification:
      raise ValueError(
          f"{game}: the artifact plays {k}={v!r}, but the installed pyspiel's "
          f"{game} has no {k!r} parameter"
      )
    if not _same(v, parameter_specification[k]):
      out[k] = v
  return out
