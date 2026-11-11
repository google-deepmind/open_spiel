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

Names: a request names its pyspiel game by pyspiel's short name. An env
serves the pyspiel game of its own name unless its ServingSpec names another
(pyspiel_game: an env of its own name for one variant of a broader pyspiel
game); env_name_for resolves a request to the env, pyspiel_name an env (or an
artifact's game) to its pyspiel game. Artifacts record the env's name.

Optional parameters: some served parameters are exposed only by extended
pyspiel builds (ServingSpec.upstream_values maps each to the fixed value
played when pyspiel omits that parameter). stock_params turns a parameter
set into one the installed pyspiel accepts, dropping such a parameter at
that implicit value and refusing any other value with UpstreamGapError.
"""

from typing import Any, Mapping

from open_spiel.python.scale.core import types
from open_spiel.python.scale.envs import registry as env_registry

ANY = env_registry.ANY


def served_games() -> tuple[str, ...]:
  """The envs that serve a pyspiel game.

  Returns:
    The env names (each env's pyspiel game: pyspiel_name).
  """
  return tuple(g.name for g in env_registry.games() if g.served)


def pyspiel_params_of(spec: env_registry.EnvSpec) -> Any:
  """The pyspiel parameters of the env `spec`, when its ServingSpec says.

  Args:
    spec: an EnvSpec.

  Returns:
    ServingSpec.pyspiel_params_of(spec.kwargs), or None when the env is not
    served or has no such map.
  """
  g = env_registry.game(spec.name)
  if g.serving is None or g.serving.pyspiel_params_of is None:
    return None
  return dict(g.serving.pyspiel_params_of(dict(spec.kwargs)))


def pyspiel_name(env_name: str) -> str:
  """The short name of the pyspiel game env `env_name` serves.

  Args:
    env_name: an env's (or an artifact's game's) name.

  Returns:
    Its ServingSpec.pyspiel_game, else `env_name` (also for a name that is
    not registered or not served).
  """
  if env_name not in env_registry.names():
    return env_name
  serving = env_registry.game(env_name).serving
  if serving is None or serving.pyspiel_game is None:
    return env_name
  return serving.pyspiel_game


def describe(env_name: str) -> str:
  """`env_name` for a message: repr'd, with its pyspiel game when another.

  Args:
    env_name: an env's (or an artifact's game's) name.

  Returns:
    "'name'", or "'name' (pyspiel 'game')" when pyspiel_name differs.
  """
  pyname = pyspiel_name(env_name)
  if pyname == env_name:
    return repr(env_name)
  return f"{env_name!r} (pyspiel {pyname!r})"


def served_pyspiel_games() -> tuple[str, ...]:
  """The pyspiel games an env serves, by pyspiel short name.

  Returns:
    The pyspiel game names, sorted, each once.
  """
  return tuple(sorted({pyspiel_name(n) for n in served_games()}))


def _check_env_params(env_name: str, params: Mapping[str, Any]) -> None:
  """Checks that served env `env_name` plays every one of `params`.

  Args:
    env_name: a served env.
    params: pyspiel game parameters.

  Raises:
    ValueError: a parameter the env does not implement, or a value it does
      not play.
  """
  game = pyspiel_name(env_name)
  allowed = env_registry.game(env_name).serving.pyspiel_params
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


def env_name_for(game: str, params: Mapping[str, Any]) -> str:
  """The env that serves pyspiel `game` with parameters `params`.

  Args:
    game: the pyspiel game's short name.
    params: pyspiel game parameters.

  Returns:
    The env's name.

  Raises:
    KeyError: no env serves `game`.
    ValueError: no env serving `game` plays `params` (or, a registry error,
      more than one does).
    RuntimeError: an env serving `game` raised anything but ValueError while
      checking `params` (ServingSpec.served_kwargs refuses with ValueError).
  """
  names = [n for n in served_games() if pyspiel_name(n) == game]
  if not names:
    raise KeyError(
        f"no env for game {game!r}; served: {served_pyspiel_games()}"
    )
  if len(names) == 1:
    return names[0]
  plays, refusals = [], []
  for n in names:
    try:
      _check_env_params(n, params)
      env_registry.game(n).serving.served_kwargs(dict(params))
    except ValueError as e:
      refusals.append(f"{n}: {e}")
      continue
    except Exception as e:  # pylint: disable=broad-exception-caught
      # Only ValueError is a refusal. Anything else (a KeyError above all,
      # which callers read as "no env serves the game") is a bug in env n.
      raise RuntimeError(
          f"{game}: env {n!r} failed to check parameters {dict(params)} "
          f"(a refusal must be ValueError): {type(e).__name__}: {e}"
      ) from e
    plays.append(n)
  if len(plays) != 1:
    raise ValueError(
        f"{game}: {len(plays)} envs play parameters {dict(params)} "
        f"({plays or refusals})"
    )
  return plays[0]


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
  _check_env_params(env_name_for(game, params), params)


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
  name = env_name_for(game, params)
  _check_env_params(name, params)
  g = env_registry.game(name)
  return env_registry.spec(name, **g.serving.served_kwargs(dict(params)))


def env_for(game: str, params: Mapping[str, Any]) -> types.Env:
  """The Env for `game` with pyspiel game parameters `params`.

  Args:
    game: the pyspiel game's short name.
    params: pyspiel game parameters.

  Returns:
    The Env.
  """
  spec = env_spec_for(game, params)
  env = env_registry.make(spec)
  assert env.name == spec.name, (env.name, spec.name)
  return env


class UpstreamGapError(ValueError):
  """A parameter value the installed pyspiel cannot play.

  The parameter is absent from the installed pyspiel's game and the value
  differs from what a build without that parameter plays.

  Attributes:
    game: the game name.
    param: the parameter the installed pyspiel lacks.
    value: the value the artifact needs.
  """

  def __init__(self, game: str, param: str, value: Any, message: str):
    super().__init__(message)
    self.game, self.param, self.value = game, param, value


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


def stock_params(
    game: str,
    params: Mapping[str, Any],
    parameter_specification: Mapping[str, Any],
) -> dict[str, Any]:
  """`params` as a pyspiel whose `game` has `parameter_specification` takes.

  A parameter that pyspiel knows is kept unless it equals pyspiel's default
  (then dropped: the same game). One it does not know is dropped when its
  value matches ServingSpec.upstream_values (what a pyspiel build without
  the parameter plays) and refused otherwise.

  Args:
    game: the env's name (its pyspiel game: pyspiel_name).
    params: the game parameters.
    parameter_specification: {parameter: default} of the target pyspiel game.

  Returns:
    The parameters to write.

  Raises:
    KeyError: `game` is not served.
    UpstreamGapError: a parameter the installed pyspiel lacks, at a value
      other than its implicit value (or one with no recorded implicit value).
  """
  serving = env_registry.game(game).serving
  if serving is None:
    raise KeyError(f"no env for game {game!r}; served: {served_games()}")
  upstream = serving.upstream_values
  name = pyspiel_name(game)
  out = {}
  for k in sorted(params):
    v = params[k]
    if k in parameter_specification:
      if not _same(v, parameter_specification[k]):
        out[k] = v
      continue
    if k in upstream and _same(v, upstream[k]):
      continue
    plays = (
        f"without that parameter {name} plays {k}={upstream[k]!r}"
        if k in upstream
        else f"no implicit value of {k!r} is recorded"
    )
    raise UpstreamGapError(
        name,
        k,
        v,
        f"{name}: the artifact plays {k}={v!r}, but the installed pyspiel's "
        f"{name} has no {k!r} parameter and {plays}; the installed pyspiel "
        "cannot play this artifact",
    )
  return out
