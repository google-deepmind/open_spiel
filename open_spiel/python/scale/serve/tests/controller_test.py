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

"""The GUI's "checkpoint" controller, against stand-ins.

Registered into a stand-in of the GUI's controllers module and served by a
stand-in PolicyServer: nothing is loaded or compiled.

The stand-in has the interface of open_spiel.python.gui.controllers that
register_into uses (register, Kind, Setting, REQUIRED and
UnavailableError).
"""

import ast
import dataclasses
import sys
import types as pytypes
from typing import Any, Callable, Optional

import pytest

from open_spiel.python.scale.core import types
from open_spiel.python.scale.serve import controller
from open_spiel.python.scale.serve import server as server_lib
import pyspiel


class FakeUnavailableError(Exception):
  """The stand-in of controllers.UnavailableError."""


FAKE_REQUIRED = "REQUIRED"


@dataclasses.dataclass(frozen=True)
class FakeSetting:
  """The stand-in of controllers.Setting."""

  name: str
  kind: type[Any]
  default: Any
  description: str
  minimum: Optional[float] = None
  maximum: Optional[float] = None
  min_length: Optional[int] = None

  def parse(self, value):
    """The setting's value, checked.

    Args:
      value: the configured value.

    Returns:
      The value as the setting's kind.

    Raises:
      FakeUnavailableError: a value of the wrong kind, below the minimum or too
        short.
    """
    if isinstance(value, bool) != (self.kind is bool) or not isinstance(
        value, (int, float) if self.kind is float else self.kind
    ):
      raise FakeUnavailableError(f"{self.name}: {value!r}")
    if self.minimum is not None and value < self.minimum:
      raise FakeUnavailableError(f"{self.name} < {self.minimum}")
    if self.min_length is not None and len(value) < self.min_length:
      raise FakeUnavailableError(f"{self.name} shorter than {self.min_length}")
    return self.kind(value)


@dataclasses.dataclass(frozen=True)
class FakeKind:
  """The stand-in of controllers.Kind."""

  label: str
  settings: tuple[Any, ...]
  reads_full_state: bool
  build: Callable[..., Any]
  refusal: Callable[..., Any] = lambda game: None

  def parse(self, config):
    """The kind's settings from a config, checked.

    Args:
      config: {setting name: value}.

    Returns:
      {setting name: parsed value}.

    Raises:
      FakeUnavailableError: an unknown or missing required setting.
    """
    unknown = set(config) - {s.name for s in self.settings}
    if unknown:
      raise FakeUnavailableError(f"unknown {unknown}")
    out = {}
    for s in self.settings:
      if s.name not in config and s.default is FAKE_REQUIRED:
        raise FakeUnavailableError(f"setting {s.name!r} is required")
      out[s.name] = s.parse(config.get(s.name, s.default))
    return out


def fake_controllers():
  """The interface of the GUI's controllers module that register_into uses.

  Returns:
    The stand-in module.
  """
  registry = {}

  def register(name, kind):
    if name in registry:
      raise ValueError(name)
    registry[name] = kind

  return pytypes.SimpleNamespace(
      register=register,
      Kind=FakeKind,
      Setting=FakeSetting,
      REQUIRED=FAKE_REQUIRED,
      UnavailableError=FakeUnavailableError,
      registry=registry,
  )


class FakeServer:
  """A PolicyServer stand-in.

  One tic_tac_toe artifact at steps 1 and 2 that plays the last legal action; it
  records every request and every agent warmup (seat, max_plies).

  Attributes:
    requests: every request, in order.
    warmups: every agent warmup (seat, max_plies).
    manifest: the artifact's manifest stand-in.
    policy: the artifact's policy stand-in.
  """

  def __init__(self):
    self.requests = []
    self.warmups = []
    self.manifest = pytypes.SimpleNamespace(game="tic_tac_toe", env_spec=None)
    self.policy = pytypes.SimpleNamespace(
        spec=pytypes.SimpleNamespace(num_players=2)
    )

  def artifact(self, path, step):
    """The one artifact.

    Args:
      path: its directory.
      step: its step.

    Returns:
      The artifact (manifest and policy).

    Raises:
      FileNotFoundError: another path or step.
    """
    if path != "/runs/ttt" or step not in (1, 2):
      raise FileNotFoundError(f"no artifact at {path}@{step}")
    return pytypes.SimpleNamespace(manifest=self.manifest, policy=self.policy)

  def agent(self, path, step, seat, game, params, rule=None):
    """An agent whose warmup records (seat, max_plies).

    Args:
      path: the artifact's directory.
      step: the artifact's step.
      seat: the seat.
      game: the pyspiel game name.
      params: the game parameters.
      rule: the decision rule.

    Returns:
      The agent stand-in.

    Raises:
      ValueError: a game other than default tic_tac_toe.
    """
    del path, step, rule  # one artifact, one rule
    if game != "tic_tac_toe" or params:
      raise ValueError(f"cannot play {game} {params}")
    return pytypes.SimpleNamespace(
        warmup=lambda n: self.warmups.append((seat, n))
    )

  def act(self, state, *, checkpoint_dir, step, seed, rule, repair=False):
    """Records the request and plays the last legal action.

    Args:
      state: the pyspiel state.
      checkpoint_dir: the artifact's directory.
      step: the artifact's step.
      seed: the seed.
      rule: the decision rule.
      repair: whether a rules difference is played through; the checkpoint
        seat always asks for it.

    Returns:
      The Reply.
    """
    assert repair, "the checkpoint seat plays on through a rules difference"
    self.requests.append(
        (checkpoint_dir, step, seed, rule, tuple(state.history()))
    )
    legal = state.legal_actions()
    return server_lib.Reply(
        action=legal[-1],
        probs={a: 1.0 / len(legal) for a in legal},
        value=None,
        decider="network",
        artifact="ttt@2",
    )


@pytest.fixture(name="registered")
def _registered():
  """The registered checkpoint controller and its fake server.

  Returns:
    (controller, server).
  """
  ctrl, srv = fake_controllers(), FakeServer()
  controller.register_into(ctrl, server=srv)
  return ctrl.registry["checkpoint"], srv


def test_registration(registered):
  """Registration."""
  kind, _ = registered
  assert kind.label == "Checkpoint" and not kind.reads_full_state
  assert [(s.name, s.kind, s.default) for s in kind.settings] == [
      ("checkpoint", str, FAKE_REQUIRED),
      ("deterministic", bool, True),
      ("temperature", float, 1.0),
  ]
  assert kind.settings[0].min_length == 1 and kind.settings[2].minimum == 0.0
  assert kind.refusal(pyspiel.load_game("tic_tac_toe")) is None
  assert "chess" in kind.refusal(pyspiel.load_game("chess"))


def test_build_warms_up_and_plays(registered):
  """Build warms up and plays."""
  kind, srv = registered
  game = pyspiel.load_game("tic_tac_toe")
  bot = kind.build(
      game,
      kind.parse({
          "checkpoint": "/runs/ttt@2",
          "deterministic": False,
          "temperature": 0.5,
      }),
      17,
  )
  assert {r[1] for r in srv.requests} == {2}
  assert {(r[0], r[2]) for r in srv.requests} == {("/runs/ttt", 17)}
  assert {r[3] for r in srv.requests} == {types.DecisionRule("sample", 0.5)}
  movers = {len(r[4]) % 2 for r in srv.requests}
  assert movers == {0, 1}, "warmup decides for both seats"
  # Every history length of tic_tac_toe compiled, for both seats: 9 moves + the
  # end.
  assert sorted(srv.warmups) == [(0, 10), (1, 10)], srv.warmups
  srv.requests.clear()
  s = game.new_initial_state()
  s.apply_action(4)
  assert bot.choose_action(s, 1) == s.legal_actions()[-1]
  assert bot.last_probs == {a: 1.0 / 8 for a in s.legal_actions()}
  with pytest.raises(ValueError, match="not to move"):
    bot.choose_action(s, 0)


def test_build_refusals(registered):
  """Build refusals."""
  kind, _ = registered
  ttt = pyspiel.load_game("tic_tac_toe")
  with pytest.raises(FakeUnavailableError, match="required"):
    kind.parse({})
  # build checks it too
  with pytest.raises(FakeUnavailableError, match="required"):
    kind.build(
        ttt, {"checkpoint": " ", "deterministic": True, "temperature": 1.0}, 0
    )
  with pytest.raises(FakeUnavailableError, match="temperature"):
    kind.build(
        ttt,
        kind.parse({
            "checkpoint": "/runs/ttt@1",
            "deterministic": False,
            "temperature": 0.0,
        }),
        0,
    )
  with pytest.raises(FakeUnavailableError, match="no artifact"):
    kind.build(ttt, kind.parse({"checkpoint": "/runs/ttt@3"}), 0)
  with pytest.raises(
      FakeUnavailableError, match="plays 'tic_tac_toe', not 'connect_four'"
  ):
    kind.build(
        pyspiel.load_game("connect_four"),
        kind.parse({"checkpoint": "/runs/ttt@1"}),
        0,
    )


def test_resolver(registered):
  """Resolver."""
  kind, srv = registered
  controller.register_checkpoint_resolver(
      "board", lambda s: "/runs/ttt@" + s[len("board://") :]
  )
  try:
    with pytest.raises(ValueError, match="already"):
      controller.register_checkpoint_resolver("board", lambda s: s)
    kind.build(
        pyspiel.load_game("tic_tac_toe"),
        kind.parse({"checkpoint": "board://1"}),
        0,
    )
    assert {r[:2] for r in srv.requests} == {("/runs/ttt", 1)}
    assert controller.resolve_checkpoint("other://x@4") == ("other://x", 4)
  finally:
    controller.unregister_checkpoint_resolver("board")
  assert controller.resolve_checkpoint("board://1") == ("board://1", None)


def test_module_imports_only_the_package_and_the_standard_library():
  """The module imports nothing outside the package.

  The GUI is handed to register_into.
  """
  with open(controller.__file__, encoding="utf-8") as f:
    tree = ast.parse(f.read())
  roots = set()
  for node in ast.walk(tree):
    if isinstance(node, ast.Import):
      roots.update(a.name.split(".")[0] for a in node.names)
    elif isinstance(node, ast.ImportFrom):
      roots.add(node.module.split(".")[0])
  assert roots == {
      "__future__",
      "re",
      "threading",
      "typing",
      controller.__name__.split(".", maxsplit=1)[0],
  }


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
