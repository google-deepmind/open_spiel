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


"""The one-step launcher's life: it starts both servers, opens the session
with its Remote bot seat, and stops both again, on a failure and on a
signal. The bot is the stand-in checkpoint bot, so nothing beyond OpenSpiel
is needed."""

import http.client
import io
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from typing import Any
from unittest import mock
import urllib.parse

from absl.testing import absltest

from open_spiel.python.gui import child_servers
from open_spiel.python.gui import play
from open_spiel.python.gui import stand_in_checkpoint_bot

STAND_IN = "open_spiel.python.gui.stand_in_checkpoint_bot:make_bot"
# This module, as the bot server imports a factory from it.
TEST_MODULE = "open_spiel.python.gui.play_test"


def make_second_variable_bot() -> stand_in_checkpoint_bot.StandInCheckpointBot:
  """The stand-in bot, its checkpoint read from CHECKPOINT_ENVS[1] alone."""
  return stand_in_checkpoint_bot.StandInCheckpointBot(
      os.environ.get(play.CHECKPOINT_ENVS[1])
  )


def _call(
    url: str, body: dict[str, Any] | None = None, method: str | None = None
) -> tuple[int, bytes]:
  """GETs `url`, or POSTs `body` to it as JSON; returns the status and body."""
  parts = urllib.parse.urlsplit(url)
  connection = http.client.HTTPConnection(
      parts.hostname or "", parts.port, timeout=30
  )
  try:
    connection.request(
        method or ("GET" if body is None else "POST"),
        parts.path + (f"?{parts.query}" if parts.query else ""),
        body=None if body is None else json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    response = connection.getresponse()
    return response.status, response.read()
  finally:
    connection.close()


def _get(url: str) -> Any:
  """GETs `url`; returns its JSON reply, which must be 200."""
  status, raw = _call(url)
  assert status == 200, (url, status, raw)
  return json.loads(raw)


def _post(url: str, body: dict[str, Any]) -> Any:
  """POSTs `body` to `url`; returns its JSON reply, which must be 200."""
  status, raw = _call(url, body)
  assert status == 200, (url, status, raw)
  return json.loads(raw)


def _refuses(url: str) -> bool:
  """Whether nothing listens at `url` any more."""
  try:
    _call(url)
  except ConnectionRefusedError:
    return True
  return False


class ParseTest(absltest.TestCase):

  def test_flags_after_a_double_dash_go_to_the_gui_server(self) -> None:
    options = play.parse_args([
        "--game",
        "gin_rummy(oklahoma=true)",
        "--bot",
        STAND_IN,
        "--checkpoint",
        " /tmp/c@3 ",
        "--",
        "--exclude-games",
        "backgammon",
    ])
    self.assertEqual(options.checkpoint, "/tmp/c@3")
    self.assertEqual(options.checkpoint_envs, play.CHECKPOINT_ENVS)
    self.assertEqual(
        play.CHECKPOINT_ENVS,
        (
            "OPEN_SPIEL_REMOTE_BOT_CHECKPOINT",
            "OPEN_SPIEL_SCALE_REMOTE_BOT_CHECKPOINT",
        ),
    )
    self.assertEqual(options.move_budget_seconds, 60.0)
    self.assertEqual(options.gui_flags, ("--exclude-games", "backgammon"))
    self.assertEqual(options.state, "full_state")
    self.assertEqual(options.bot_seat, 1)

  def test_a_named_checkpoint_variable_is_the_only_one_set(self) -> None:
    options = play.parse_args(
        ["--game", "kuhn_poker", "--bot", STAND_IN, "--checkpoint-env", "V"]
    )
    self.assertEqual(options.checkpoint_envs, ("V",))

  def test_a_game_string_gives_its_name_and_parameters(self) -> None:
    self.assertEqual(
        play.parse_game("gin_rummy(oklahoma=true)"),
        ("gin_rummy", {"oklahoma": True}),
    )
    self.assertEqual(play.parse_game("kuhn_poker"), ("kuhn_poker", {}))

  def test_an_unknown_game_is_refused(self) -> None:
    with self.assertRaisesRegex(play.LaunchError, "no registered game"):
      play.parse_game("no_such_game")

  def test_the_session_seats_the_bot_and_humans_elsewhere(self) -> None:
    options = play.Options(
        game="gin_rummy(oklahoma=true)", bot=STAND_IN, checkpoint="/c@3"
    )
    body = play.session_body(options, "http://127.0.0.1:9", 2)
    self.assertEqual(body["game"], "gin_rummy")
    self.assertEqual(body["parameters"], {"oklahoma": True})
    self.assertEqual(
        body["seats"],
        [
            {"controller": "human", "config": {}, "autoplay": False},
            {
                "controller": "remote",
                "config": {
                    "url": "http://127.0.0.1:9",
                    "state": "full_state",
                    "checkpoint": "/c@3",
                    "timeout_seconds": 0.0,
                },
                "autoplay": True,
            },
        ],
    )


class ChildServerTest(absltest.TestCase):
  """wait_ready's message once a child's output ends without a ready line."""

  def start(self, code: str) -> child_servers.ChildServer:
    """A child running `code` in Python, which never prints a ready line."""
    child = child_servers.ChildServer(
        "bot",
        [sys.executable, "-c", code],
        os.environ,
        re.compile(r"^listening at (\S+)$"),
        io.StringIO(),
    )
    self.addCleanup(child.stop)
    return child

  def test_a_child_that_exits_at_once_always_reports_its_exit(self) -> None:
    # A few runs only: on an idle machine this passes even without the reap,
    # so the close-then-exit test below is the guard against the race.
    for _ in range(5):
      child = self.start("import sys; print('no'); sys.exit(3)")
      with self.assertRaisesRegex(
          child_servers.LaunchError, "bot server exited with code 3"
      ):
        child.wait_ready(120)

  def test_output_ending_before_the_exit_is_reapable_reports_the_exit(
      self,
  ) -> None:
    # The race the kernel makes: a dying child's output closes before the
    # child can be reaped. Here the gap is made 1 s wide; one poll() right
    # after the output ends sees no exit and blames the 120 s budget.
    child = self.start(
        "import os, sys, time; os.close(1); os.close(2); time.sleep(1);"
        " sys.exit(4)"
    )
    started = time.monotonic()
    with self.assertRaisesRegex(
        child_servers.LaunchError, "bot server exited with code 4"
    ):
      child.wait_ready(120)
    self.assertLess(time.monotonic() - started, 60)

  def test_a_child_that_closes_its_output_and_runs_on_is_reported(
      self,
  ) -> None:
    child = self.start(
        "import os, time; os.close(1); os.close(2); time.sleep(60)"
    )
    with mock.patch.object(child_servers, "REAP_SECONDS", 0.5):
      with self.assertRaisesRegex(
          child_servers.LaunchError,
          "bot server closed its output before it listened and is still"
          " running",
      ):
        child.wait_ready(120)
    self.assertIsNone(child.process.poll())


class LaunchTest(absltest.TestCase):

  def setUp(self) -> None:
    super().setUp()
    self.directory = self.create_tempdir().full_path
    self.checkpoint = stand_in_checkpoint_bot.write(
        os.path.join(self.directory, "ckpt"), 3, "kuhn_poker"
    )
    self.out = io.StringIO()

  def options(self, **changes: Any) -> play.Options:
    """The launch of a kuhn_poker stand-in checkpoint, with `changes`."""
    values: dict[str, Any] = {
        "game": "kuhn_poker",
        "bot": STAND_IN,
        "checkpoint": self.checkpoint,
        "port": 0,
        "bot_start_seconds": 120.0,
    }
    values.update(changes)
    return play.Options(**values)

  def test_it_opens_the_session_and_the_bot_plays_its_checkpoint(self) -> None:
    running = play.launch(self.options(), self.out)
    self.addCleanup(running.stop)
    self.assertRegex(
        running.play_url,
        rf"^{re.escape(running.gui_url)}/game/kuhn_poker\?session="
        rf"{running.session}&viewer=0$",
    )
    # The bot server read the checkpoint from its environment.
    info = _get(running.bot_url + "/info")
    self.assertEqual(info["games"], ["kuhn_poker"])
    api = f"{running.gui_url}/api/sessions/{running.session}"
    frame = _get(api + "?viewer=oracle")
    seat = frame["seats"][1]
    self.assertEqual(seat["controller"], "remote")
    self.assertEqual(seat["config"]["url"], running.bot_url)
    self.assertEqual(seat["config"]["state"], "full_state")
    self.assertEqual(seat["config"]["checkpoint"], self.checkpoint)
    self.assertEqual(frame["seats"][0]["controller"], "human")
    _post(api + "/actions?viewer=0", {"action": 0, "ply": None})
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
      frame = _get(api + "?viewer=oracle")
      if any(entry["seat"] == 1 for entry in frame["history"]):
        break
      time.sleep(0.2)
    self.assertTrue(any(entry["seat"] == 1 for entry in frame["history"]))
    self.assertEqual(frame["seats"][1]["notice"], "stand-in ckpt@3")
    running.stop()
    self.assertEqual(running.exited(), "bot")
    self.assertTrue(_refuses(running.bot_url + "/health"))
    self.assertTrue(_refuses(running.gui_url + "/api/health"))

  def test_the_checkpoint_is_in_every_variable_a_factory_may_read(
      self,
  ) -> None:
    running = play.launch(
        self.options(bot=f"{TEST_MODULE}:make_second_variable_bot"),
        self.out,
    )
    self.addCleanup(running.stop)
    self.assertEqual(_get(running.bot_url + "/info")["games"], ["kuhn_poker"])

  def test_a_checkpoint_switched_in_the_seat_may_warm_up(self) -> None:
    # A checkpoint named only in the seat loads at the bot's first move. It
    # takes 12 s here, over the GUI server's own 10 s budget; the launcher's
    # budget lets the move through.
    slow = stand_in_checkpoint_bot.write(
        os.path.join(self.directory, "slow"), 4, "kuhn_poker", load_seconds=12
    )
    running = play.launch(self.options(), self.out)
    self.addCleanup(running.stop)
    api = f"{running.gui_url}/api/sessions/{running.session}"
    config = play.seat_config(self.options(checkpoint=slow), running.bot_url)
    status, raw = _call(
        api + "/seats/1?viewer=oracle",
        {"controller": "remote", "config": config, "autoplay": True},
        method="PUT",
    )
    self.assertEqual(status, 200, raw)
    started = time.monotonic()
    _post(api + "/actions?viewer=0", {"action": 0, "ply": None})
    frame: dict[str, Any] = {}
    while time.monotonic() < started + 50:
      frame = _get(api + "?viewer=oracle")
      if any(entry["seat"] == 1 for entry in frame["history"]):
        break
      self.assertIsNone(frame["seats"][1].get("failure"), frame["seats"][1])
      time.sleep(0.5)
    self.assertTrue(any(entry["seat"] == 1 for entry in frame["history"]))
    self.assertGreater(time.monotonic() - started, 10)
    self.assertEqual(frame["seats"][1]["notice"], "stand-in slow@4")
    self.assertIsNone(frame["seats"][1].get("failure"))

  def test_a_bot_that_does_not_start_stops_the_launch(self) -> None:
    with self.assertRaisesRegex(play.LaunchError, "bot server exited"):
      play.launch(self.options(bot="no_such_module:make_bot"), self.out)
    self.assertIn("[bot server] ", self.out.getvalue())
    self.assertNotIn("[gui server] ", self.out.getvalue())

  def test_a_missing_checkpoint_stops_the_launch(self) -> None:
    with self.assertRaisesRegex(play.LaunchError, "bot server exited"):
      play.launch(
          self.options(checkpoint=os.path.join(self.directory, "none@1")),
          self.out,
      )
    self.assertNotIn("[gui server] ", self.out.getvalue())

  def test_a_seat_the_game_does_not_have_starts_nothing(self) -> None:
    with self.assertRaisesRegex(play.LaunchError, "not a seat"):
      play.launch(self.options(bot_seat=2), self.out)
    self.assertEqual(self.out.getvalue(), "")


class MainTest(absltest.TestCase):
  """The launcher as a command: however it ends, both servers end with it."""

  def start(self, stdout: int) -> tuple[subprocess.Popen[str], list[str]]:
    """Starts the launcher on a kuhn_poker stand-in checkpoint.

    Args:
      stdout: Its stdout, as `subprocess.Popen` takes it.

    Returns:
      The launcher, and the lines of its stderr as they arrive.
    """
    directory = self.create_tempdir().full_path
    checkpoint = stand_in_checkpoint_bot.write(
        os.path.join(directory, "ckpt"), 7, "kuhn_poker"
    )
    process = subprocess.Popen(  # pylint: disable=consider-using-with
        [
            sys.executable,
            "-m",
            "open_spiel.python.gui.play",
            "--game",
            "kuhn_poker",
            "--bot",
            STAND_IN,
            "--checkpoint",
            checkpoint,
            "--port",
            "0",
        ],
        stdout=stdout,
        stderr=subprocess.PIPE,
        text=True,
    )
    self.addCleanup(self.reap, process)
    errors: list[str] = []
    assert process.stderr is not None
    threading.Thread(
        target=lambda: errors.extend(process.stderr or ()), daemon=True
    ).start()
    return process, errors

  def reap(self, process: subprocess.Popen[str]) -> None:
    """Kills the launcher if it still runs, and closes its pipes."""
    process.kill()
    process.wait(30)
    for pipe in (process.stdout, process.stderr):
      if pipe is not None:
        pipe.close()

  def server_urls(self, errors: list[str]) -> tuple[str, str]:
    """Returns the bot and GUI servers' URLs, from the launcher's stderr."""
    urls: dict[str, str] = {}
    deadline = time.monotonic() + 150
    while len(urls) < 2 and time.monotonic() < deadline:
      for line in list(errors):
        match = re.search(
            r"\[(bot|gui) server\] Serving .* at (http://\S+)", line
        )
        if match:
          urls[match.group(1)] = match.group(2).rstrip("/")
      time.sleep(0.1)
    self.assertEqual(sorted(urls), ["bot", "gui"], "".join(errors))
    return urls["bot"], urls["gui"]

  def assert_both_stopped(self, bot_url: str, gui_url: str) -> None:
    """Nothing listens at either server's address any more."""
    self.assertTrue(_refuses(bot_url + "/health"))
    self.assertTrue(_refuses(gui_url + "/api/health"))

  def stop_with(self, signum: signal.Signals) -> None:
    """Launches, waits for the address, sends `signum`; both servers stop."""
    process, errors = self.start(subprocess.PIPE)
    assert process.stdout is not None
    line = process.stdout.readline()
    match = re.search(r"^Play (http://\S+)/game/kuhn_poker\?", line)
    self.assertIsNotNone(match, f"{line!r} {''.join(errors)}")
    assert match is not None
    bot_url, gui_url = self.server_urls(errors)
    self.assertEqual(gui_url, match.group(1))
    self.assertTrue(_get(gui_url + "/api/health"))
    self.assertTrue(_get(bot_url + "/health")["ok"])
    process.send_signal(signum)
    self.assertEqual(process.wait(30), 0, "".join(errors))
    self.assertIn(f"Received {signum.name}; stopping.", "".join(errors))
    self.assert_both_stopped(bot_url, gui_url)

  def test_sigterm_stops_both_servers(self) -> None:
    self.stop_with(signal.SIGTERM)

  def test_sighup_stops_both_servers(self) -> None:
    # Closing the terminal: the servers, in their own sessions, do not get
    # the SIGHUP, so the launcher must stop them.
    self.stop_with(signal.SIGHUP)

  def test_sigint_stops_both_servers(self) -> None:
    self.stop_with(signal.SIGINT)

  def test_a_closed_stdout_stops_both_servers(self) -> None:
    # The reader of stdout is gone before the address is printed, so the
    # print raises BrokenPipeError; the launcher fails, loudly, and stops
    # both servers on its way out.
    read, write = os.pipe()
    os.close(read)
    try:
      process, errors = self.start(write)
    finally:
      os.close(write)
    bot_url, gui_url = self.server_urls(errors)
    code = process.wait(60)
    self.assertNotEqual(code, 0, "".join(errors))
    self.assertIn("BrokenPipeError", "".join(errors))
    self.assert_both_stopped(bot_url, gui_url)

  def test_a_signal_during_the_launch_stops_what_it_started(self) -> None:
    process, errors = self.start(subprocess.PIPE)
    deadline = time.monotonic() + 150
    bot_url = None
    while bot_url is None and time.monotonic() < deadline:
      for line in list(errors):
        match = re.search(r"\[bot server\] Serving .* at (http://\S+)", line)
        if match:
          bot_url = match.group(1).rstrip("/")
      time.sleep(0.05)
    self.assertIsNotNone(bot_url, "".join(errors))
    assert bot_url is not None
    # A second signal right after the first must not cut the stop short.
    process.send_signal(signal.SIGTERM)
    process.send_signal(signal.SIGTERM)
    code = process.wait(60)
    self.assertIn(code, (0, 1), "".join(errors))
    self.assertTrue(_refuses(bot_url + "/health"))
    for line in list(errors):
      match = re.search(r"\[gui server\] Serving .* at (http://\S+)", line)
      if match:
        self.assertTrue(_refuses(match.group(1).rstrip("/") + "/api/health"))


if __name__ == "__main__":
  absltest.main()
