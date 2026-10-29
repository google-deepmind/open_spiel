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

"""remote_bot over HTTP, through OpenSpiel's reference bot server and GUI.

OpenSpiel's reference bot server (open_spiel.python.gui.remote_bot_server,
its own process on 127.0.0.1) serves `--bot <this package>.serve.
remote_bot:make_bot`, and the GUI's own Remote bot controller plays whole
leduc games against it: every move is legal and is the in-process bot's
move, /info names the artifact's game and the full_state mode, a seat_view
seat is refused, and a rules-difference notice reaches the GUI's controller
and, under the GUI's analysis state policy, the seat's view. The GUI's own
full_state request is the body remote_bot takes (contract 8: FULL_STATE_KEYS,
the version in the Spiel-Protocol header, notices of at most 200
characters).
With GIN_ARTIFACT_ENV naming a trained gin checkpoint, 20 gin games are
played against a uniform random seat, every move legal.
SKIPPED, the reason named, where the installed OpenSpiel has no
remote_bot_server.
"""

import json
import os
import re
import subprocess
import sys
import time
import urllib.request

import numpy as np
import pytest

from open_spiel.python.scale.ckpt import artifact as ckpt
from open_spiel.python.scale.launch import subprocess_env
from open_spiel.python.scale.serve import checkpoint_bot as checkpoint_bot_lib
from open_spiel.python.scale.serve import remote_bot
from open_spiel.python.scale.serve.tests import remote_bot_helpers as rbh
import pyspiel

_FACTORY = f"{remote_bot.__name__}:make_bot"
_START_SECONDS = 300.0
#: A trained gin checkpoint for test_gin_checkpoint_twenty_games.
GIN_ARTIFACT_ENV = remote_bot.CHECKPOINT_ENV + "_TEST_GIN"


@pytest.fixture(name="leduc_dir", scope="module")
def _leduc_dir(tmp_path_factory):
  """A leduc artifact on disk, written once per module.

  Args:
    tmp_path_factory: pytest's temporary directory factory.

  Returns:
    The artifact's directory.
  """
  return rbh.write_leduc(tmp_path_factory.mktemp("leduc"))


def _gui():
  """The GUI's modules, or a SKIP naming what is missing.

  Returns:
    (controllers, remote_bot client, move_budget).
  """
  reason = (
      "the installed OpenSpiel has no GUI remote bot server "
      "(open_spiel.python.gui.remote_bot_server)"
  )
  pytest.importorskip("open_spiel.python.gui.remote_bot_server", reason=reason)
  return tuple(
      pytest.importorskip(f"open_spiel.python.gui.{m}", reason=reason)
      for m in ("controllers", "remote_bot", "move_budget")
  )


def _gui_protocol():
  """The GUI's contract and remote bot HTTP modules (a SKIP as _gui).

  Returns:
    (contract, remote_bot_http).
  """
  _gui()
  return tuple(
      pytest.importorskip(f"open_spiel.python.gui.{m}")
      for m in ("contract", "remote_bot_http")
  )


def start_server(factory, checkpoint):
  """Starts the reference bot server on 127.0.0.1 (a free port).

  Args:
    factory: the `--bot` MODULE:FACTORY.
    checkpoint: the CHECKPOINT_ENV value.

  Returns:
    (process, its URL).
  """
  # The child finds this package first on its PYTHONPATH, from a working
  # directory that shadows no installed package (subprocess_env).
  env = subprocess_env.child_env({remote_bot.CHECKPOINT_ENV: checkpoint})
  argv = [
      *subprocess_env.python_command(),
      "-m",
      "open_spiel.python.gui.remote_bot_server",
      "--port",
      "0",
      "--bot",
      factory,
  ]
  # The server prints its URL when it listens; its log goes to stderr.
  process = subprocess.Popen(  # pylint: disable=consider-using-with
      argv,
      stdout=subprocess.PIPE,
      stderr=None,
      text=True,
      env=env,
      cwd=subprocess_env.child_cwd(),
  )
  deadline = time.monotonic() + _START_SECONDS
  while time.monotonic() < deadline:
    line = process.stdout.readline()
    if not line:
      break
    m = re.search(r"Serving the .* bot at (http://\S+)", line)
    if m:
      return process, m.group(1)
  process.kill()
  raise AssertionError(f"{argv} did not start serving (rc {process.poll()})")


def get_json(url):
  """GETs `url` and parses its JSON body.

  Args:
    url: the URL.

  Returns:
    The parsed body.
  """
  with urllib.request.urlopen(url, timeout=30) as r:  # noqa: S310
    return json.loads(r.read())


@pytest.fixture(name="bot_server")
def _bot_server():
  """start(factory, checkpoint) -> URL; every server stops after the test.

  Yields:
    start: starts a server and returns its URL.
  """
  procs = []

  def start(factory, checkpoint):
    """Starts a reference bot server (start_server) stopped after the test.

    Args:
      factory: the `--bot` MODULE:FACTORY.
      checkpoint: the CHECKPOINT_ENV value.

    Returns:
      The server's URL.
    """
    process, url = start_server(factory, checkpoint)
    procs.append(process)
    return url

  yield start
  for p in procs:
    p.terminate()
    p.wait(timeout=30)


def full_state_spec(url, config=None):
  """The GUI's settings of a full_state Remote bot seat at `url`.

  Args:
    url: the bot server's URL.
    config: the seat's config object.

  Returns:
    The controllers.Spec.
  """
  controllers, gui_remote_bot, _ = _gui()
  return controllers.Spec(
      gui_remote_bot.NAME,
      {"url": url, "state": "full_state", "config": json.dumps(config or {})},
  )


def gui_seat(url, game, config=None):
  """The GUI's Remote bot controller for a full_state seat at `url`.

  Args:
    url: the bot server's URL.
    game: the pyspiel game.
    config: the seat's config object.

  Returns:
    choose(state, seat) -> (action, notice), through the GUI's controller.
  """
  controllers, _, _ = _gui()
  bot = controllers.create(full_state_spec(url, config), game, seed=1)
  return lambda s, seat: controllers.choose_with_notice(bot, s, seat)


def test_contract_8_is_what_remote_bot_takes(leduc_dir, bot_server):
  """The GUI's full_state request, header and notice limit are remote_bot's.

  The GUI's own Remote bot builds the request remote_bot_helpers.request_of
  builds (FULL_STATE_KEYS, in order, no `protocol` key); the version is the
  GUI's Spiel-Protocol header; the notice limit is the contract's.

  Args:
    leduc_dir: the leduc artifact the server plays.
    bot_server: starts a reference bot server (the fixture).
  """
  controllers, _, _ = _gui()
  contract, remote_bot_http = _gui_protocol()
  assert contract.CONTRACT_VERSION == 8
  assert contract.NOTICE_MAX_LENGTH == remote_bot.NOTICE_MAX_LENGTH
  assert remote_bot_http.PROTOCOL_HEADER == remote_bot.PROTOCOL_HEADER
  assert remote_bot_http.PROTOCOL == remote_bot.PROTOCOL
  url = bot_server(_FACTORY, leduc_dir)
  game = pyspiel.load_game(rbh.GAME)
  gui_bot = controllers.create(
      full_state_spec(url, {"deterministic": True}), game, seed=1
  )
  s = game.new_initial_state()
  s.apply_action(0)
  s.apply_action(3)
  path, raw = gui_bot.request(s, 0)
  body = json.loads(raw)
  assert path == "/spiel/step", path
  assert tuple(body) == remote_bot.FULL_STATE_KEYS, list(body)
  assert body == rbh.request_of(
      game, s, 0, {"deterministic": True}, seed=body["seed"]
  ), body


def test_round_trip_through_the_reference_server(leduc_dir, bot_server):
  """Whole games over HTTP: legal, and the in-process bot's moves."""
  _, _, move_budget = _gui()
  move_budget.set_seconds(60.0)
  url = bot_server(_FACTORY, leduc_dir)
  assert get_json(url + "/health")["ok"]
  info = get_json(url + "/info")
  assert info["state_modes"] == ["full_state"], info
  assert info["games"] == [rbh.GAME], info
  game = pyspiel.load_game(rbh.GAME)
  choose = gui_seat(url, game)
  ref = checkpoint_bot_lib.checkpoint_bot(leduc_dir)
  rng = np.random.default_rng(1)
  n = 0
  for g in range(10):
    for hist, action, notice in rbh.play(game, choose, bot_seat=g % 2, rng=rng):
      s = game.new_initial_state()
      for a in hist:
        s.apply_action(a)
      assert action == ref.choose_action(s, s.current_player()), hist
      assert notice is None, notice
      n += 1
  assert n >= 10, n


def test_notice_reaches_the_gui(leduc_dir, bot_server):
  """A rules-difference notice reaches the GUI's controller over HTTP."""
  _, _, move_budget = _gui()
  move_budget.set_seconds(60.0)
  url = bot_server(rbh.MISMATCH_FACTORY, leduc_dir)
  game = pyspiel.load_game(rbh.GAME)
  choose = gui_seat(url, game)
  s = game.new_initial_state()
  s.apply_action(0)
  s.apply_action(3)
  action, notice = choose(s, 0)
  assert action == 1, action
  assert notice.startswith("engine/agent rules differ on this move"), notice
  assert len(notice) <= remote_bot.NOTICE_MAX_LENGTH
  # The state policy's hook (controller_registry.full_state_notice) passes a
  # full-state seat's notice on to the seat's view under ANALYSIS.
  controllers, _, _ = _gui()
  assert controllers.seat_notice(full_state_spec(url), game, notice) == notice


def test_seat_view_seat_is_refused(leduc_dir, bot_server):
  """A seat_view seat is refused when the GUI reads /info."""
  controllers, gui_remote_bot, _ = _gui()
  url = bot_server(_FACTORY, leduc_dir)
  spec = controllers.Spec(
      gui_remote_bot.NAME, {"url": url, "state": "seat_view"}
  )
  with pytest.raises(controllers.UnavailableError, match="full_state"):
    controllers.create(spec, pyspiel.load_game(rbh.GAME), seed=1)


def test_gin_checkpoint_twenty_games(bot_server):
  """The gin checkpoint GIN_ARTIFACT_ENV names: 20 games over HTTP vs random.

  Every move the GUI's controller receives is legal (the GUI checks it, and
  so does play). SKIPPED unless GIN_ARTIFACT_ENV names a trained gin
  checkpoint.

  Args:
    bot_server: starts a reference bot server (the fixture).
  """
  path = os.environ.get(GIN_ARTIFACT_ENV, "")
  if not path:
    pytest.skip(f"{GIN_ARTIFACT_ENV} names no gin artifact")
  _, _, move_budget = _gui()
  move_budget.set_seconds(120.0)
  ckpt_dir, step = checkpoint_bot_lib.split_checkpoint(path)
  if step is None:
    step = ckpt.latest_step(ckpt_dir)
  manifest = ckpt.load_manifest(ckpt_dir, step)
  game = pyspiel.load_game(checkpoint_bot_lib.game_string_of(manifest))
  url = bot_server(_FACTORY, path)
  assert get_json(url + "/info")["games"] == ["gin_rummy"]
  choose = gui_seat(url, game)
  rng = np.random.default_rng(20)
  moves, notices = 0, []
  for g in range(20):
    for hist, _, notice in rbh.play(game, choose, bot_seat=g % 2, rng=rng):
      moves += 1
      if notice is not None:
        assert len(notice) <= remote_bot.NOTICE_MAX_LENGTH, notice
        notices.append((g, len(hist), notice))
  print(
      f"gin {game}: 20 games, {moves} bot moves over HTTP, all legal; "
      f"{len(notices)} notices: {notices}"
  )
  assert moves >= 20 * 5, moves


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
