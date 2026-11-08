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

"""Envs that serve a pyspiel game of another name (ServingSpec.pyspiel_game).

Fast; needs no pyspiel.

A request names its pyspiel game; games.env_name_for resolves it to the env
that plays its parameters, and games.pyspiel_name maps an env (or an
artifact's game) back to its pyspiel game. Synthetic envs serve pyspiel's
kuhn_poker beside the builtin kuhn_poker env: `kuhn_three_four` plays
players=3 or 4 (the N-player Kuhn env) and maps its options back with
pyspiel_params_of; `kuhn_three` (registered by some tests) plays players=3
too, so a request for players=3 then has two envs.
"""

import dataclasses
import sys
import types as pytypes

import pytest

from open_spiel.python.scale.core import types
from open_spiel.python.scale.envs import registry as env_registry
from open_spiel.python.scale.serve import checkpoint_bot as checkpoint_bot_lib
from open_spiel.python.scale.serve import controller
from open_spiel.python.scale.serve import games
from open_spiel.python.scale.serve import host_agent
from open_spiel.python.scale.serve import remote_bot
from open_spiel.python.scale.serve import server as server_lib
from open_spiel.python.scale.serve import wire

_KUHN_NPLAYER_CTOR = env_registry.game("kuhn_nplayer").ctor


def _served_kwargs_of(allowed):
  """served_kwargs of an env playing kuhn_poker with `allowed` players.

  Args:
    allowed: The player counts the env plays.

  Returns:
    pyspiel params -> the env's options; ValueError for any other count.
  """

  def served_kwargs(params):
    """The env's options for pyspiel params.

    Args:
      params: the pyspiel game's parameters.

    Returns:
      {"players": n}.

    Raises:
      ValueError: the player count is not in `allowed`.
    """
    players = params.get("players")
    if players not in allowed:
      raise ValueError(f"plays players in {allowed}, not {players!r}")
    return {"players": int(players)}

  return served_kwargs


def _spec(name, allowed, pyspiel_params_of=None):
  """A GameSpec of an N-player Kuhn env served as pyspiel kuhn_poker.

  Args:
    name: The env's name.
    allowed: The player counts it plays.
    pyspiel_params_of: Its ServingSpec.pyspiel_params_of.

  Returns:
    The GameSpec.
  """
  return env_registry.GameSpec(
      name,
      _KUHN_NPLAYER_CTOR,
      default_kwargs={"players": allowed[0]},
      serving=env_registry.ServingSpec(
          pyspiel_params={"players": tuple(allowed)},
          served_kwargs=_served_kwargs_of(tuple(allowed)),
          pyspiel_game="kuhn_poker",
          pyspiel_params_of=pyspiel_params_of,
      ),
  )


def _inverse(kw):
  """kuhn_three_four's pyspiel_params_of: the options' pyspiel params.

  Args:
    kw: the env's options.

  Returns:
    The pyspiel params.
  """
  return {"players": int(kw["players"])}


@pytest.fixture(autouse=True)
def _registry(monkeypatch):
  """kuhn_three_four registered for the test, undone after it."""
  # Reaches private _GAMES of the code under test: monkeypatch restores it.
  monkeypatch.setattr(
      env_registry,
      "_GAMES",
      dict(env_registry._GAMES),  # pylint: disable=protected-access
  )
  env_registry.register(_spec("kuhn_three_four", (3, 4), _inverse))


def _register_kuhn_three():
  """A second env that plays kuhn_poker(players=3)."""
  env_registry.register(_spec("kuhn_three", (3,)))


def _manifest(game, env_spec):
  """A minimal Manifest of `game` trained on `env_spec`.

  Args:
    game: The artifact's game (an env name).
    env_spec: The EnvSpec.

  Returns:
    The Manifest.
  """
  return types.Manifest(
      schema_version=1,
      game=game,
      rules_sha="r",
      obs_format="o",
      arch_name="a",
      arch_kwargs={},
      framework="f",
      capability=None,
      default_rule=types.ARGMAX,
      params_sha256="p",
      step=1,
      parents=(),
      env_spec=env_spec.to_json(),
  )


def _pyspiel_game(short_name):
  """A stand-in pyspiel game with only a short name.

  Args:
    short_name: its game type's short_name.

  Returns:
    The stand-in.
  """
  game_type = pytypes.SimpleNamespace(short_name=short_name)
  return pytypes.SimpleNamespace(get_type=lambda: game_type)


# -- names ---------------------------------------------------------------
def test_builtins_serve_the_pyspiel_game_of_their_own_name(monkeypatch):
  """Every builtin served env's pyspiel game is its own name."""
  monkeypatch.setattr(
      env_registry,
      "_GAMES",
      {g.name: g for g in env_registry.BUILTIN},
  )
  served = games.served_games()
  assert served
  for name in served:
    assert games.pyspiel_name(name) == name
  assert games.served_pyspiel_games() == tuple(sorted(served))


def test_pyspiel_name_and_served_pyspiel_games():
  """An env of another name maps to its pyspiel game; names listed once."""
  assert games.pyspiel_name("kuhn_three_four") == "kuhn_poker"
  assert games.pyspiel_name("kuhn_poker") == "kuhn_poker"
  # Not registered: passed through (every caller then fails on the name).
  assert games.pyspiel_name("no_such_env") == "no_such_env"
  assert "kuhn_three_four" in games.served_games()
  served = games.served_pyspiel_games()
  assert "kuhn_three_four" not in served
  assert served.count("kuhn_poker") == 1
  assert list(served) == sorted(served)
  # Messages name the pyspiel game only when it is another name.
  assert games.describe("kuhn_poker") == "'kuhn_poker'"
  assert (
      games.describe("kuhn_three_four")
      == "'kuhn_three_four' (pyspiel 'kuhn_poker')"
  )


def test_pyspiel_game_must_be_a_name():
  """ServingSpec.pyspiel_game is None or a non-empty string."""
  for bad in ("", 3):
    with pytest.raises(ValueError, match="pyspiel_game"):
      env_registry.ServingSpec(pyspiel_params={}, pyspiel_game=bad)


# -- env_name_for ----------------------------------------------------------
def test_no_env_serves_the_game():
  """Zero envs: KeyError naming the served pyspiel games."""
  with pytest.raises(KeyError, match="no env for game 'no_such_game'"):
    games.env_name_for("no_such_game", {})
  # An env's own name is not a pyspiel game when it serves another one.
  with pytest.raises(KeyError):
    games.env_name_for("kuhn_three_four", {"players": 3})


def test_one_env_serves_the_game():
  """One env: it, whatever the parameters (they are checked later)."""
  assert games.env_name_for("tic_tac_toe", {}) == "tic_tac_toe"


def test_two_envs_exactly_one_plays_the_parameters():
  """kuhn_poker and kuhn_three_four: the parameters pick the env."""
  assert games.env_name_for("kuhn_poker", {}) == "kuhn_poker"
  assert games.env_name_for("kuhn_poker", {"players": 2}) == "kuhn_poker"
  assert games.env_name_for("kuhn_poker", {"players": 3}) == "kuhn_three_four"
  assert games.env_name_for("kuhn_poker", {"players": 4}) == "kuhn_three_four"
  spec = games.env_spec_for("kuhn_poker", {"players": 4})
  assert spec == env_registry.spec("kuhn_three_four", players=4)
  assert games.env_spec_for("kuhn_poker", {}) == env_registry.spec("kuhn_poker")
  games.check_params("kuhn_poker", {"players": 3})


def test_two_envs_both_play_the_parameters():
  """Two envs that both play players=3: ValueError (a registry error)."""
  _register_kuhn_three()
  with pytest.raises(ValueError, match="2 envs play"):
    games.env_name_for("kuhn_poker", {"players": 3})
  # players=4: only kuhn_three_four still.
  assert games.env_name_for("kuhn_poker", {"players": 4}) == "kuhn_three_four"


def test_no_env_of_several_plays_the_parameters():
  """Every env serving the game refuses: ValueError naming each refusal."""
  _register_kuhn_three()
  with pytest.raises(ValueError, match="0 envs play") as e:
    games.env_name_for("kuhn_poker", {"players": 9})
  for name in ("kuhn_poker", "kuhn_three_four", "kuhn_three"):
    assert f"{name}:" in str(e.value)
  with pytest.raises(ValueError):
    games.env_spec_for("kuhn_poker", {"players": 9})


def test_wire_translator_resolves_the_env():
  """The wire module resolves the env by the request's parameters."""
  assert wire.pyspiel_history_to_raw(
      "kuhn_poker", [0, 1, 2], {"players": 3}
  ) == [[0, 1, 2]]


def test_a_sibling_that_fails_otherwise_than_valueerror_is_a_bug(
    monkeypatch,
):
  """Only ValueError refuses; a KeyError is not read as "no env serves it"."""
  g = env_registry.game("kuhn_three_four")

  def served_kwargs(params):
    """A translator that fails with KeyError.

    Args:
      params: the pyspiel game's parameters.

    Returns:
      Never: it reads a key no request has.
    """
    return {"players": int(params["no_such_key"])}

  # Reaches private _GAMES (a copy, restored by the fixture's monkeypatch).
  monkeypatch.setitem(
      env_registry._GAMES,  # pylint: disable=protected-access
      "kuhn_three_four",
      dataclasses.replace(
          g, serving=dataclasses.replace(g.serving, served_kwargs=served_kwargs)
      ),
  )
  with pytest.raises(RuntimeError, match="'kuhn_three_four'.*KeyError"):
    games.env_name_for("kuhn_poker", {"players": 3})
  with pytest.raises(RuntimeError):
    games.env_spec_for("kuhn_poker", {})


# -- pyspiel_params_of and game strings ----------------------------------
def test_pyspiel_params_of():
  """The env's map, or None when it has none or is not served."""
  spec = env_registry.spec("kuhn_three_four", players=4)
  assert games.pyspiel_params_of(spec) == {"players": 4}
  assert games.pyspiel_params_of(env_registry.spec("kuhn_poker")) is None
  assert games.pyspiel_params_of(env_registry.spec("kuhn_nplayer")) is None


def test_game_string_of_writes_the_mapped_parameters():
  """game_string_of names the pyspiel game and writes pyspiel_params_of's."""
  m = _manifest(
      "kuhn_three_four", env_registry.spec("kuhn_three_four", players=4)
  )
  assert (
      checkpoint_bot_lib.game_string_of(m, {"players": 2})
      == "kuhn_poker(players=4)"
  )


def test_game_string_of_refuses_a_map_that_is_not_the_inverse(monkeypatch):
  """A pyspiel_params_of whose parameters play another env is refused."""
  g = env_registry.game("kuhn_three_four")
  bad = dataclasses.replace(
      g.serving, pyspiel_params_of=lambda kw: {"players": 3}
  )
  # Reaches private _GAMES (a copy, restored by the fixture's monkeypatch).
  monkeypatch.setitem(
      env_registry._GAMES,  # pylint: disable=protected-access
      "kuhn_three_four",
      dataclasses.replace(g, serving=bad),
  )
  m = _manifest(
      "kuhn_three_four", env_registry.spec("kuhn_three_four", players=4)
  )
  with pytest.raises(ValueError, match="not the training env"):
    checkpoint_bot_lib.game_string_of(m, {"players": 2})


# -- the serving surfaces, by pyspiel name -------------------------------
def test_controller_refusal_by_pyspiel_name():
  """The controller plays pyspiel games, never an env's own other name."""
  assert controller.refusal(_pyspiel_game("kuhn_poker")) is None
  assert "kuhn_three_four is not one" in controller.refusal(
      _pyspiel_game("kuhn_three_four")
  )


class _Artifact:
  """A stand-in artifact of game `game` (only its manifest's game)."""

  def __init__(self, game):
    self.manifest = pytypes.SimpleNamespace(game=game)


def test_server_refuses_a_checkpoint_of_another_pyspiel_game(monkeypatch):
  """PolicyServer checks the request's game against the artifact's game."""
  server = server_lib.PolicyServer()
  monkeypatch.setattr(
      server, "artifact", lambda d, s: _Artifact("kuhn_three_four")
  )
  with pytest.raises(ValueError, match=r"\(pyspiel 'kuhn_poker'\)"):
    server.agent("/ckpt", 1, 0, "leduc_poker", {})
  # The env's own name is not the request's game either.
  with pytest.raises(ValueError, match="the request's game"):
    server.agent("/ckpt", 1, 0, "kuhn_three_four", {"players": 3})


def test_remote_bot_reports_the_pyspiel_game(monkeypatch):
  """RemoteCheckpointBot.games is the default artifact's pyspiel game."""
  monkeypatch.setattr(
      remote_bot.RemoteCheckpointBot,
      "bot",
      lambda self, config: _Artifact("kuhn_three_four"),
  )
  bot = remote_bot.RemoteCheckpointBot("/ckpt@1", warmup=False)
  assert bot.games == ("kuhn_poker",)


class _BuiltError(Exception):
  """Raised by the stand-in env_registry.make once it has the spec."""


def test_host_agent_builds_a_legacy_artifacts_own_env(monkeypatch):
  """No env_spec: HostAgent builds the artifact's env, not pyspiel's game."""
  built = []

  def make(spec):
    built.append(spec)
    raise _BuiltError()

  monkeypatch.setattr(env_registry, "make", make)
  policy = pytypes.SimpleNamespace(
      spec=pytypes.SimpleNamespace(game="kuhn_three_four")
  )
  manifest = _manifest(
      "kuhn_three_four", env_registry.spec("kuhn_three_four")
  )._replace(env_spec=None)
  art = types.Artifact(
      manifest=manifest, params=None, policy=policy, path="mem://legacy"
  )
  with pytest.raises(_BuiltError):
    host_agent.HostAgent(art, 0)
  assert built == [env_registry.spec("kuhn_three_four")]


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
