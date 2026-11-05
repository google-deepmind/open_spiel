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

"""game_string_of writes game strings the installed pyspiel loads.

Fast.

A parameter at pyspiel's default is dropped; a parameter the installed pyspiel
omits (for example gin_rummy's starting_player in standard pyspiel) is
dropped at its implicit value and refused at any other, and checkpoint_bot's
check_serves and warmup then refuse with a message naming the parameter.
"""

import dataclasses
import sys
import types as pytypes

import pytest

from open_spiel.python.scale.core import types
from open_spiel.python.scale.envs import registry as env_registry
from open_spiel.python.scale.serve import checkpoint_bot as checkpoint_bot_lib
from open_spiel.python.scale.serve import games
from open_spiel.python.scale.serve import server as server_lib
import pyspiel


def _manifest(game, env_spec):
  """A minimal Manifest of `game` and its EnvSpec.

  Args:
    game: The game.
    env_spec: The EnvSpec JSON.

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
      env_spec=env_spec,
  )


def _gin_manifest(**kwargs):
  """A gin Manifest with the given env options.

  Args:
    **kwargs: The env options.

  Returns:
    The Manifest.
  """
  return _manifest(
      "gin_rummy", env_registry.spec("gin_rummy", **kwargs).to_json()
  )


# Extended gin_rummy parameters omitted by standard pyspiel.
_NOT_UPSTREAM = ("starting_player",) + tuple(env_registry.GIN_RULE_DEFAULTS)

# gin_rummy's parameter_specification when extended rule parameters are exposed.
EXTENDED_GIN_SPEC = {
    "allow_upcard_throwback": True,
    "gin_bonus": 25,
    "hand_size": 10,
    "knock_card": 10,
    "num_ranks": 13,
    "num_suits": 4,
    "oklahoma": False,
    "spade_doubles": False,
    "starting_player": 0,
    "undercut_bonus": 25,
    "wall_knock_pick": True,
}


def _stock_gin_spec():
  """gin_rummy's parameters as standard pyspiel has them.

  No starting_player and none of the rule parameters (GIN_RULE_DEFAULTS).

  Returns:
    The parameter specification.
  """
  spec = checkpoint_bot_lib.installed_parameter_specification("gin_rummy")
  for k in _NOT_UPSTREAM:
    spec.pop(k, None)
  return spec


def test_stock_params_drops_defaults_and_upstream_values():
  """Stock params drops defaults and upstream values."""
  spec = _stock_gin_spec()
  params = {
      "gin_bonus": 25,
      "hand_size": 10,
      "knock_card": 10,
      "num_ranks": 13,
      "num_suits": 4,
      "oklahoma": True,
      "starting_player": 0,
      "undercut_bonus": 25,
  }
  assert games.stock_params("gin_rummy", params, spec) == {"oklahoma": True}
  assert not games.stock_params(
      "gin_rummy", {"oklahoma": False, "starting_player": 0}, spec
  )
  # A parameter the pyspiel knows keeps a non-default value; a bool is not an
  # int.
  assert games.stock_params("gin_rummy", {"knock_card": 9}, spec) == {
      "knock_card": 9
  }
  assert games.stock_params("gin_rummy", {"oklahoma": 1}, spec) == {
      "oklahoma": 1
  }
  # A pyspiel build that exposes the parameter drops it at its default too.
  assert not games.stock_params(
      "gin_rummy", {"starting_player": 0}, {**spec, "starting_player": 0}
  )
  assert games.stock_params(
      "gin_rummy", {"starting_player": 1}, {**spec, "starting_player": 0}
  ) == {"starting_player": 1}


def test_stock_params_refuses_an_upstream_gap():
  """Stock params refuses an upstream gap."""
  with pytest.raises(games.UpstreamGapError) as e:
    games.stock_params("gin_rummy", {"starting_player": 1}, _stock_gin_spec())
  assert (e.value.game, e.value.param, e.value.value) == (
      "gin_rummy",
      "starting_player",
      1,
  )
  msg = str(e.value)
  assert (
      "starting_player=1" in msg and "has no 'starting_player' parameter" in msg
  )
  assert (
      "without that parameter gin_rummy plays starting_player=0" in msg
      and "cannot play this artifact" in msg
  )


def test_stock_params_refuses_an_unknown_parameter_with_no_upstream_value(
    monkeypatch,
):
  """Stock params refuses an unknown parameter with no upstream value."""
  g = env_registry.game("gin_rummy")
  serving = dataclasses.replace(g.serving, upstream_values={})
  # Reaches private _GAMES of the code under test: a gin row without the
  # upstream values, undone by monkeypatch.
  monkeypatch.setitem(
      env_registry._GAMES,  # pylint: disable=protected-access
      "gin_rummy",
      dataclasses.replace(g, serving=serving),
  )
  with pytest.raises(
      games.UpstreamGapError, match="no implicit value of 'starting_player'"
  ):
    games.stock_params("gin_rummy", {"starting_player": 0}, _stock_gin_spec())


def test_upstream_values_must_be_served_parameters():
  """Upstream values must be served parameters."""
  with pytest.raises(ValueError, match="does not play"):
    env_registry.ServingSpec(
        pyspiel_params={"a": (0,)}, upstream_values={"b": 0}
    )


def test_gin_oklahoma_string_loads_on_stock_pyspiel():
  """An Oklahoma gin artifact's env (Oklahoma deals, first player at chance).

  The default-valued starting_player=0 is dropped when absent from pyspiel, and
  standard pyspiel loads the string, a game in which player 0 moves first.
  """
  m = _gin_manifest(oklahoma_frac=1.0)
  s = checkpoint_bot_lib.game_string_of(m, _stock_gin_spec())
  assert s == "gin_rummy(oklahoma=true)"
  # A pyspiel with starting_player known (default 0) gets the same
  # string.
  assert (
      checkpoint_bot_lib.game_string_of(
          m, {**_stock_gin_spec(), "starting_player": 0}
      )
      == s
  )
  # And the installed pyspiel, whichever it is, loads its string.
  game = pyspiel.load_game(checkpoint_bot_lib.game_string_of(m))
  assert game.get_parameters()["oklahoma"]
  state = game.new_initial_state()
  while state.is_chance_node():
    state.apply_action(state.chance_outcomes()[0][0])
  assert state.current_player() == 0


def test_gin_default_artifact_is_the_short_name():
  """Gin default artifact is the short name."""
  assert (
      checkpoint_bot_lib.game_string_of(_gin_manifest(), _stock_gin_spec())
      == "gin_rummy"
  )


def _gap_registry(monkeypatch):
  """gin_rummy whose served env depends on starting_player.

  1 deals Oklahoma hands, 0 standard ones (a stand-in for an artifact trained
  with a fixed first player standard pyspiel does not play).

  Args:
    monkeypatch: pytest's monkeypatch.
  """
  g = env_registry.game("gin_rummy")
  serving = env_registry.ServingSpec(
      pyspiel_params={"starting_player": (0, 1)},
      served_kwargs=lambda p: {
          "oklahoma_frac": 1.0 if p.get("starting_player", 0) == 1 else None
      },
      covers=env_registry.gin_covers,
      upstream_values={"starting_player": 0},
  )
  # Reaches private _GAMES of the code under test: a gin row without the
  # upstream values, undone by monkeypatch.
  monkeypatch.setitem(
      env_registry._GAMES,  # pylint: disable=protected-access
      "gin_rummy",
      dataclasses.replace(g, serving=serving),
  )


def test_non_upstream_value_is_refused(monkeypatch):
  """Non upstream value is refused."""
  _gap_registry(monkeypatch)
  m = _gin_manifest(oklahoma_frac=1.0)
  with pytest.raises(games.UpstreamGapError, match="starting_player=1"):
    checkpoint_bot_lib.game_string_of(m, _stock_gin_spec())
  # A pyspiel with the parameter plays it.
  assert (
      checkpoint_bot_lib.game_string_of(
          m, {**_stock_gin_spec(), "starting_player": 0}
      )
      == "gin_rummy(starting_player=1)"
  )
  # The upstream value itself (standard deals) is dropped.
  assert (
      checkpoint_bot_lib.game_string_of(_gin_manifest(), _stock_gin_spec())
      == "gin_rummy"
  )


class _Game:
  """A pyspiel-like gin_rummy without starting_player (standard pyspiel's)."""

  def __init__(self, params):
    self._params = params

  def get_type(self):
    """The game type.

    Returns:
      A stand-in with short_name "gin_rummy".
    """
    return pytypes.SimpleNamespace(short_name="gin_rummy")

  def get_parameters(self):
    """The game parameters.

    Returns:
      The parameters.
    """
    return dict(self._params)

  def num_players(self):
    """Two players.

    Returns:
      2.
    """
    return 2


def test_check_serves_and_warmup_refuse_an_upstream_gap(monkeypatch):
  """Check serves and warmup refuse an upstream gap."""
  _gap_registry(monkeypatch)
  stock = _stock_gin_spec()
  monkeypatch.setattr(
      checkpoint_bot_lib,
      "installed_parameter_specification",
      lambda name: dict(stock),
  )
  m = _gin_manifest(oklahoma_frac=1.0)
  policy = pytypes.SimpleNamespace(spec=pytypes.SimpleNamespace(num_players=2))
  srv = server_lib.PolicyServer(
      loader=lambda path, step: types.Artifact(
          manifest=m, params=None, policy=policy, path=path
      )
  )
  bot = checkpoint_bot_lib.checkpoint_bot("mem://gap", step=1, server=srv)
  assert bot.game_string is None and bot.upstream_gap.param == "starting_player"
  with pytest.raises(
      ValueError,
      match=r"starting_player=1.*has no 'starting_player'.*"
      r"without that parameter gin_rummy plays starting_player=0.*"
      r"cannot play this artifact",
  ):
    bot.check_serves(_Game(stock))
  with pytest.raises(ValueError, match="cannot play this artifact"):
    bot.warmup([0])


@pytest.mark.parametrize("oklahoma", [False, True])
def test_extended_gin_parameters_map_to_the_env(oklahoma):
  """Every parameter in EXTENDED_GIN_SPEC maps to the env.

  A served game reports all its parameters (get_parameters), the rule ones
  included; they map to the env's rule options of the same name.

  Args:
    oklahoma: The parametrized case.
  """
  params = {**EXTENDED_GIN_SPEC, "oklahoma": oklahoma}
  games.check_params("gin_rummy", params)
  spec = games.env_spec_for("gin_rummy", params)
  assert spec == env_registry.spec(
      "gin_rummy", oklahoma_frac=1.0 if oklahoma else None
  )
  for k, v in env_registry.GIN_RULE_DEFAULTS.items():
    assert spec.kwargs[k] is v
  # Standard pyspiel's game (no rule parameters) is the same env.
  stock = {k: v for k, v in params.items() if k not in _NOT_UPSTREAM}
  assert games.env_spec_for("gin_rummy", stock) == spec


@pytest.mark.parametrize("oklahoma", [False, True])
def test_installed_gin_parameters_are_served(oklahoma):
  """The installed pyspiel's gin_rummy, all its parameters, is served.

  Whichever pyspiel is installed (standard, or an extended build with the rule
  parameters): a game reports every parameter it has.

  Args:
    oklahoma: The parametrized case.
  """
  game = pyspiel.load_game("gin_rummy", {"oklahoma": oklahoma})
  params = dict(game.get_parameters())
  games.check_params("gin_rummy", params)
  assert games.env_spec_for("gin_rummy", params) == env_registry.spec(
      "gin_rummy", oklahoma_frac=1.0 if oklahoma else None
  )


@pytest.mark.parametrize(
    "param,value",
    [
        ("allow_upcard_throwback", False),
        ("wall_knock_pick", False),
        ("spade_doubles", True),
    ],
)
def test_non_default_gin_rule_parameter_is_refused(param, value):
  """A rule parameter off its default is refused, naming it."""
  params = {**EXTENDED_GIN_SPEC, "oklahoma": True, param: value}
  with pytest.raises(ValueError, match=rf"{param}={value!r}; the env plays"):
    games.check_params("gin_rummy", params)
  with pytest.raises(ValueError, match=param):
    games.env_spec_for("gin_rummy", params)


def test_oklahoma_env_string_with_and_without_extended_params():
  """The env -> game string direction with and without extended parameters.

  With extended parameters the rule parameters (at their defaults) and
  starting_player=0 are dropped as defaults; standard pyspiel omits them and
  plays each at its implicit value, so they are dropped there too.
  """
  m = _gin_manifest(oklahoma_frac=1.0)
  assert (
      checkpoint_bot_lib.game_string_of(m, EXTENDED_GIN_SPEC)
      == "gin_rummy(oklahoma=true)"
  )
  stock = {k: v for k, v in EXTENDED_GIN_SPEC.items() if k not in _NOT_UPSTREAM}
  assert checkpoint_bot_lib.game_string_of(m, stock) == (
      "gin_rummy(oklahoma=true)"
  )
  assert checkpoint_bot_lib.game_string_of(_gin_manifest(), stock) == (
      "gin_rummy"
  )
  # An artifact trained under a non-default rule has no served game string.
  odd = _gin_manifest(oklahoma_frac=1.0, allow_upcard_throwback=False)
  with pytest.raises(ValueError, match="no pyspiel game string"):
    checkpoint_bot_lib.game_string_of(odd, EXTENDED_GIN_SPEC)
  # stock_params drops each rule parameter at its default and
  # refuses any other value.
  for k, v in env_registry.GIN_RULE_DEFAULTS.items():
    assert not games.stock_params("gin_rummy", {k: v}, stock)
    with pytest.raises(games.UpstreamGapError, match=k):
      games.stock_params("gin_rummy", {k: not v}, stock)


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
