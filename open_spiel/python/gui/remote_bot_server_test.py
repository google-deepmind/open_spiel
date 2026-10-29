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

"""Tests for the reference bot server, alone and end to end.

The end-to-end tests run the GUI server and the reference bot server as the
recipe in REMOTE_BOTS.md does, each a process of its own on this machine,
and play through the GUI server's HTTP API.
"""

import http.client
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any
import urllib.parse

from absl.testing import absltest

import pyspiel

from open_spiel.python.gui import controllers
from open_spiel.python.gui import remote_bot
from open_spiel.python.gui import remote_bot_server
from open_spiel.python.gui import remote_bot_testing

# The environment variable the recording bot logs its requests to.
_LOG_VARIABLE = "REMOTE_BOT_TEST_LOG"

# This module, as `--bot MODULE:FACTORY` names it, however the test is run.
_MODULE = "open_spiel.python.gui.remote_bot_server_test"


class RecordingBot:
  """Logs each request to a file, then plays as the request's config says.

  The config may give `sleep` (seconds to wait first) and `reply` (the
  reply to send); otherwise it plays the lowest legal action.
  """

  state_modes = (remote_bot_server.SEAT_VIEW, remote_bot_server.FULL_STATE)
  games = None

  def step(self, mode: str, request: dict[str, Any]) -> Any:
    """Logs the request, then plays as its config says."""
    with open(os.environ[_LOG_VARIABLE], "a", encoding="utf-8") as log:
      log.write(json.dumps({"mode": mode, "request": request}) + "\n")
    config = request["config"]
    time.sleep(float(config.get("sleep", 0)))
    if "reply" in config:
      return config["reply"]
    return min(remote_bot_server.legal_actions(mode, request))


def recording_bot() -> RecordingBot:
  """The factory `--bot` names: MODULE:recording_bot."""
  return RecordingBot()


def _observe(game: str, seat: int, history: list[int]) -> dict[str, Any]:
  """A seat-view request body at `history`."""
  state = pyspiel.load_game(game).new_initial_state()
  for action in history:
    state.apply_action(action)
  return {
      "observation": {
          "seat": seat,
          "legal_actions": state.legal_actions(seat),
          "seed": 4,
      },
      "config": {},
  }


def _step(game: str, seat: int, history: list[int]) -> dict[str, Any]:
  """A full-state request body at `history`."""
  state = pyspiel.load_game(game).new_initial_state()
  for action in history:
    state.apply_action(action)
  return {
      "game": game,
      "state": state.serialize(),
      "history": history,
      "seat": seat,
      "legal_actions": state.legal_actions(seat),
      "seed": 4,
      "config": {"max_simulations": 300},
  }


class BotsTest(absltest.TestCase):

  def test_random_plays_a_legal_action_in_either_mode(self) -> None:
    bot = remote_bot_server.load_bot("random")
    legal = _observe("tic_tac_toe", 1, [4])["observation"]["legal_actions"]
    self.assertIn(bot.step("seat_view", _observe("tic_tac_toe", 1, [4])), legal)
    self.assertIn(bot.step("full_state", _step("tic_tac_toe", 1, [4])), legal)

  def test_random_is_reproducible_from_the_seed(self) -> None:
    bot = remote_bot_server.load_bot("random")
    request = _observe("tic_tac_toe", 0, [])
    self.assertEqual(
        {bot.step("seat_view", request) for _ in range(5)},
        {bot.step("seat_view", request)},
    )

  def test_mcts_takes_a_winning_move(self) -> None:
    bot = remote_bot_server.load_bot("mcts")
    self.assertEqual(bot.state_modes, ("full_state",))
    # x at 0 and 1, o at 3 and 4: x wins at 2.
    self.assertEqual(
        bot.step("full_state", _step("tic_tac_toe", 0, [0, 3, 1, 4])), 2
    )

  def test_mcts_refuses_a_seat_view(self) -> None:
    with self.assertRaisesRegex(ValueError, "full state"):
      remote_bot_server.load_bot("mcts").step(
          "seat_view", _observe("tic_tac_toe", 0, [])
      )

  def test_a_factory_serves_any_bot(self) -> None:
    bot = remote_bot_server.load_bot(f"{_MODULE}:recording_bot")
    self.assertEqual(type(bot).__name__, "RecordingBot")

  def test_a_bad_bot_name_is_refused(self) -> None:
    for name in ("nope", "module_only:", ":factory"):
      with self.subTest(name):
        with self.assertRaises(ValueError):
          remote_bot_server.load_bot(name)

  def test_reply_of_an_int_or_a_dict(self) -> None:
    self.assertEqual(remote_bot_server.reply_of(3, None), {"action": 3})
    self.assertEqual(
        remote_bot_server.reply_of(3, "n"), {"action": 3, "notice": "n"}
    )
    self.assertEqual(
        remote_bot_server.reply_of({"action": 3, "notice": "own"}, "n"),
        {"action": 3, "notice": "own"},
    )
    with self.assertRaises(TypeError):
      remote_bot_server.reply_of(True, None)
    with self.assertRaises(TypeError):
      remote_bot_server.reply_of({"notice": "x"}, None)


class ServerTokenTest(absltest.TestCase):
  """The reference server's bearer token, and the GUI sending it."""

  def serve(self, token: str | None) -> str:
    """Serves the random bot with `token`; returns its URL."""
    server = remote_bot_server.Server(
        ("127.0.0.1", 0), remote_bot_server.RandomBot(), name="t", token=token
    )
    threading.Thread(target=server.serve_forever, daemon=True).start()
    self.addCleanup(server.server_close)
    self.addCleanup(server.shutdown)
    self.addCleanup(remote_bot.configure)
    return server.url

  def status(self, url: str, headers: dict[str, str]) -> int:
    """GETs /health with `headers`; returns the status."""
    parts = urllib.parse.urlsplit(url)
    connection = http.client.HTTPConnection(parts.hostname or "", parts.port)
    try:
      connection.request("GET", "/health", headers=headers)
      return connection.getresponse().status
    finally:
      connection.close()

  def test_a_request_without_the_token_is_refused(self) -> None:
    url = self.serve("s3cret")
    self.assertEqual(self.status(url, {}), 401)
    self.assertEqual(self.status(url, {"Authorization": "Bearer wrong"}), 401)
    self.assertEqual(self.status(url, {"Authorization": "Bearer s3cret"}), 200)

  def test_the_gui_sends_the_token_scoped_to_the_bot(self) -> None:
    url = self.serve("s3cret")
    game = pyspiel.load_game("tic_tac_toe")
    spec = controllers.Spec(remote_bot.NAME, {"url": url})
    with self.assertRaisesRegex(controllers.UnavailableError, "HTTP 401"):
      controllers.create(spec, game, seed=1)
    port = urllib.parse.urlsplit(url).port
    remote_bot.configure(tokens={f"127.0.0.1:{port}": "s3cret"})
    bot = controllers.create(spec, game, seed=1)
    self.assertIn(
        controllers.choose(bot, game.new_initial_state(), 0), range(9)
    )

  def test_it_will_not_listen_beyond_loopback_without_a_token(self) -> None:
    with self.assertRaisesRegex(ValueError, "no token"):
      remote_bot_server.Server(
          ("0.0.0.0", 0), remote_bot_server.RandomBot(), name="t"
      )

  def test_its_command_line_says_so_in_one_line(self) -> None:
    process = subprocess.run(
        [
            sys.executable,
            "-m",
            "open_spiel.python.gui.remote_bot_server",
            "--host",
            "0.0.0.0",
            "--port",
            "0",
        ],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    self.assertEqual(process.returncode, 1, process.stderr)
    self.assertNotIn("Traceback", process.stderr)
    self.assertEqual(
        process.stderr.strip().splitlines(),
        [
            "error: The bot server would listen on 0.0.0.0, beyond this "
            "machine, with no token; give --token-env."
        ],
    )


def call(base: str, method: str, path: str, body: object = None) -> Any:
  """Calls the GUI server's API, returning the status and the JSON body."""
  parts = urllib.parse.urlsplit(base)
  connection = http.client.HTTPConnection(
      parts.hostname or "", parts.port, timeout=120
  )
  data = None if body is None else json.dumps(body).encode()
  headers = {} if data is None else {"Content-Type": "application/json"}
  try:
    connection.request(method, f"/api/{path}", body=data, headers=headers)
    response = connection.getresponse()
    return response.status, json.loads(response.read())
  finally:
    connection.close()


class EndToEndTest(absltest.TestCase):
  """The GUI server and the reference bot server, each a process."""

  dir: str
  log: pathlib.Path
  processes: list[subprocess.Popen[str]]
  gui: str
  random: str
  recording: str
  last: tuple[int, Any]

  @classmethod
  def setUpClass(cls) -> None:
    super().setUpClass()
    cls.dir = tempfile.mkdtemp()
    cls.log = pathlib.Path(cls.dir, "requests.jsonl")
    env = dict(os.environ, **{_LOG_VARIABLE: str(cls.log)})
    cls.processes = []
    gui, cls.gui = remote_bot_testing.start(
        [
            "-m",
            "open_spiel.python.gui",
            "--port=0",
            "--move-budget-seconds=2",
            f"--client-dir={cls.dir}",
        ],
        r"Serving OpenSpiel games at (\S+)",
        env,
    )
    cls.processes.append(gui)
    for name, argv in (
        ("random", ["--bot=random", "--notice=hello from the bot"]),
        ("recording", [f"--bot={_MODULE}:recording_bot"]),
    ):
      process, url = remote_bot_testing.start(
          [
              "-m",
              "open_spiel.python.gui.remote_bot_server",
              "--port=0",
              *argv,
          ],
          r"at (http://\S+)",
          env,
      )
      cls.processes.append(process)
      setattr(cls, name, url)

  @classmethod
  def tearDownClass(cls) -> None:
    for process in cls.processes:
      process.terminate()
      process.wait(timeout=30)
    super().tearDownClass()

  def setUp(self) -> None:
    super().setUp()
    self.last = (0, None)

  def requests(self) -> list[dict[str, Any]]:
    """Returns the recording bot's requests, and forgets them."""
    if not self.log.exists():
      return []
    lines = self.log.read_text(encoding="utf-8").splitlines()
    self.log.unlink()
    return [json.loads(line) for line in lines]

  def open_kuhn(self, url: str, state: str, **config: Any) -> str:
    """Opens kuhn poker, seat 1 an autoplaying remote bot; plays seat 0."""
    self.requests()
    seat = {
        "controller": "remote",
        "config": {"url": url, "state": state, **config},
        "autoplay": True,
    }
    status, frame = call(
        self.gui,
        "POST",
        "sessions?viewer=oracle",
        {
            "game": "kuhn_poker",
            "parameters": {},
            "seats": [
                {"controller": "human", "config": {}, "autoplay": False},
                seat,
            ],
            "chance_policy": "auto",
            "seed": 11,
        },
    )
    self.assertEqual(status, 200, frame)
    for _ in range(100):
      if frame["seats"][1]["status"] == "ready":
        break
      time.sleep(0.1)
      _, frame = call(
          self.gui, "GET", f"sessions/{frame['session']['id']}?viewer=oracle"
      )
    self.assertEqual(frame["seats"][1]["status"], "ready", frame)
    session = frame["session"]["id"]
    status, frame = call(
        self.gui,
        "POST",
        f"sessions/{session}/actions?viewer=0",
        {"action": 1, "ply": None},  # Seat 0 bets.
    )
    self.addCleanup(
        call, self.gui, "DELETE", f"sessions/{session}?viewer=oracle"
    )
    self.last = (status, frame)
    return str(session)

  def test_a_seat_view_bot_receives_no_hidden_information(self) -> None:
    session = self.open_kuhn(self.recording, "seat_view", extras=True)
    self.assertEqual(self.last[0], 200, self.last[1])
    (logged,) = self.requests()
    self.assertEqual(logged["mode"], "seat_view")
    request = logged["request"]
    self.assertEqual(
        set(request), {"game", "observation", "game_encoding", "config"}
    )
    _, oracle = call(self.gui, "GET", f"sessions/{session}?viewer=oracle")
    deal = [entry["action"] for entry in oracle["history"][:2]]
    seen = request["observation"]["history"]
    # The history is as the seat's own frame shows it: only the seat's own
    # actions in full, and others' public labels, which kuhn_poker's has
    # none of. The seat reads its card and the bet in its information state.
    self.assertEqual(
        [(entry["seat"], entry["action"]) for entry in seen],
        [(None, None), (None, None), (0, None)],
    )
    self.assertNotIn(deal[0], [entry["action"] for entry in seen])
    state = pyspiel.load_game("kuhn_poker").new_initial_state()
    for action in [*deal, 1]:
      state.apply_action(action)
    self.assertEqual(
        request["observation"]["information_state_string"],
        state.information_state_string(1),
    )
    self.assertNotIn(state.serialize(), json.dumps(request))

  def test_a_full_state_bot_receives_the_state(self) -> None:
    session = self.open_kuhn(self.recording, "full_state")
    (logged,) = self.requests()
    self.assertEqual(logged["mode"], "full_state")
    _, oracle = call(self.gui, "GET", f"sessions/{session}?viewer=oracle")
    history = [entry["action"] for entry in oracle["history"][:3]]
    state = pyspiel.load_game("kuhn_poker").new_initial_state()
    for action in history:
      state.apply_action(action)
    self.assertEqual(logged["request"]["state"], state.serialize())
    self.assertEqual(logged["request"]["history"], history)

  def test_the_notice_reaches_only_the_seat_and_analysis(self) -> None:
    session = self.open_kuhn(self.random, "seat_view")
    notices = {}
    for viewer in ("oracle", "spectator", "0", "1"):
      status, frame = call(
          self.gui, "GET", f"sessions/{session}?viewer={viewer}"
      )
      self.assertEqual(status, 200, frame)
      notices[viewer] = frame["seats"][1]["notice"]
    self.assertEqual(
        notices,
        {
            "oracle": "hello from the bot",
            "1": "hello from the bot",
            "0": None,
            "spectator": None,
        },
    )

  def assert_paused(self, session: str, error: str) -> None:
    """Checks the last request paused `session`, its bot failing `error`."""
    status, body = self.last
    self.assertEqual(status, 409, body)
    self.assertEqual(body["code"], "controller_failed")
    _, oracle = call(self.gui, "GET", f"sessions/{session}?viewer=oracle")
    self.assertEqual(oracle["status"], "paused")
    self.assertEqual(oracle["seats"][1]["failure"]["error"], error)
    self.assertTrue(oracle["seats"][1]["failure"]["message"])
    _, seat = call(self.gui, "GET", f"sessions/{session}?viewer=0")
    self.assertIsNone(seat["seats"][1]["failure"]["message"])

  def test_a_bot_that_overruns_the_budget_fails_its_seat(self) -> None:
    session = self.open_kuhn(self.recording, "seat_view", config='{"sleep": 4}')
    self.assert_paused(session, "RemoteBotUnreachableError")

  def test_a_bad_reply_fails_its_seat(self) -> None:
    session = self.open_kuhn(
        self.recording, "seat_view", config='{"reply": 99}'
    )
    self.assert_paused(session, "RemoteBotReplyError")


if __name__ == "__main__":
  absltest.main()
