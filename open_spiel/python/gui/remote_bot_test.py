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

"""Tests for the remote bot, against bot servers on this machine."""

import http
import http.server
import json
import threading
import time
from typing import Any

from absl.testing import absltest

import pyspiel

from open_spiel.python.gui import builds
from open_spiel.python.gui import contract
from open_spiel.python.gui import controllers
from open_spiel.python.gui import errors
from open_spiel.python.gui import remote_bot
from open_spiel.python.gui import remote_bot_server
from open_spiel.python.gui import session as session_lib

_ORACLE = contract.Viewer(contract.ViewerKind.ORACLE)
_SPECTATOR = contract.Viewer(contract.ViewerKind.SPECTATOR)


def _seat(index: int) -> contract.Viewer:
  """The viewer of seat `index`."""
  return contract.Viewer(contract.ViewerKind.SEAT, index)


class _Recording:
  """A bot that plays the lowest legal action and keeps every request."""

  def __init__(
      self,
      modes: tuple[str, ...] = ("seat_view", "full_state"),
      notice: str | None = None,
  ) -> None:
    self.state_modes = modes
    self.games = None
    self.notice = notice
    self.requests: list[tuple[str, dict[str, Any]]] = []
    self.reply: Any = None
    self.sleep = 0.0

  def step(self, mode: str, request: dict[str, Any]) -> Any:
    """Keeps the request, then answers with `reply` or the lowest action."""
    self.requests.append((mode, request))
    if self.sleep:
      time.sleep(self.sleep)
    if self.reply is not None:
      return self.reply
    action = min(remote_bot_server.legal_actions(mode, request))
    if self.notice is None:
      return action
    return {"action": action, "notice": self.notice}


class _RawHandler(http.server.BaseHTTPRequestHandler):
  """Answers every move with the server's fixed status and raw body."""

  server: "_RawServer"

  def log_message(self, format: str, *args: object) -> None:
    """Logs nothing."""
    del format, args

  def _answer(self, status: int, body: bytes) -> None:
    """Sends `body` with `status`."""
    self.send_response(status)
    self.send_header("Content-Type", "application/json")
    self.send_header("Content-Length", str(len(body)))
    self.end_headers()
    self.wfile.write(body)

  def do_GET(self) -> None:  # pylint: disable=invalid-name
    """Answers /health; nothing else."""
    if self.path == "/health":
      self._answer(200, b"{}")
    else:
      self._answer(404, b"{}")

  def do_POST(self) -> None:  # pylint: disable=invalid-name
    """Answers a move with the server's status and body."""
    self.rfile.read(int(self.headers["Content-Length"]))
    self._answer(self.server.status, self.server.body)


class _RawServer(http.server.ThreadingHTTPServer):
  """A bot server that answers every move the same way."""

  daemon_threads = True

  def __init__(self, status: int, body: bytes) -> None:
    super().__init__(("127.0.0.1", 0), _RawHandler)
    self.status = status
    self.body = body


class RemoteBotTest(absltest.TestCase):

  def setUp(self) -> None:
    super().setUp()
    self.addCleanup(controllers.set_move_budget, controllers.move_budget())
    self.addCleanup(builds.set_wait_seconds, builds.wait_seconds())
    builds.set_wait_seconds(10.0)
    self.addCleanup(remote_bot.configure)

  def serve(self, bot: Any, notice: str | None = None) -> str:
    """Serves `bot` on a free port for the test; returns its URL."""
    server = remote_bot_server.Server(
        ("127.0.0.1", 0), bot, name="test", notice=notice
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    self.addCleanup(server.server_close)
    self.addCleanup(server.shutdown)
    return server.url

  def serve_raw(self, status: int, body: bytes) -> str:
    """Serves a bot answering every move with `status` and `body`."""
    server = _RawServer(status, body)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    self.addCleanup(server.server_close)
    self.addCleanup(server.shutdown)
    host, port = server.server_address[:2]
    return f"http://{host!s}:{port}"

  def session(
      self, game: str, url: str, state: str, seat: int = 1, **config: Any
  ) -> session_lib.Session:
    """Opens `game` with every seat human but `seat`, a remote bot."""
    num = pyspiel.load_game(game).num_players()
    specs = [controllers.HUMAN_SPEC] * num
    specs[seat] = controllers.Spec(
        remote_bot.NAME, {"url": url, "state": state, **config}
    )
    return session_lib.Session.create(
        game,
        {},
        seed=3,
        chance_policy=contract.ChancePolicy.AUTO,
        controllers=specs,
    )

  def play_kuhn_to_bot(self, session: session_lib.Session) -> None:
    """Seat 0 passes; then the bot, seat 1, moves on an advance."""
    session.apply_action(_seat(0), 0)
    session.advance(_ORACLE)

  def test_seat_view_sends_only_what_the_seat_may_see(self) -> None:
    bot = _Recording()
    session = self.session("kuhn_poker", self.serve(bot), "seat_view")
    self.play_kuhn_to_bot(session)
    self.assertEqual(len(bot.requests), 1)
    mode, request = bot.requests[0]
    self.assertEqual(mode, "seat_view")
    # Exactly the keys a strict seat-view bot accepts.
    self.assertEqual(list(request), list(remote_bot.SEAT_VIEW_KEYS))
    observation = request["observation"]
    self.assertEqual(
        list(observation), list(remote_bot.SEAT_VIEW_OBSERVATION_KEYS)
    )
    self.assertEqual(observation["seat"], 1)
    # The rebuilt position is the one the bot was asked at, before its move.
    before = pyspiel.load_game("kuhn_poker").new_initial_state()
    for action in session.history[:3]:
      before.apply_action(action)
    self.assertEqual(
        observation["information_state_string"],
        before.information_state_string(1),
    )
    self.assertEqual(observation["legal_actions"], before.legal_actions(1))
    self.assertEqual(request["game"], "kuhn_poker()")
    self.assertEqual(request["game_encoding"], {"num_distinct_actions": 2})
    self.assertEqual(request["config"], {})
    self.assertNotIn(before.serialize(), json.dumps(request))

  def test_seat_view_extras_add_labels_and_the_seats_history(self) -> None:
    bot = _Recording()
    session = self.session(
        "kuhn_poker", self.serve(bot), "seat_view", extras=True
    )
    self.play_kuhn_to_bot(session)
    request = bot.requests[0][1]
    self.assertEqual(list(request), list(remote_bot.SEAT_VIEW_KEYS))
    observation = request["observation"]
    self.assertEqual(
        list(observation),
        list(remote_bot.SEAT_VIEW_OBSERVATION_KEYS + remote_bot.EXTRAS_KEYS),
    )
    before = pyspiel.load_game("kuhn_poker").new_initial_state()
    for action in session.history[:3]:
      before.apply_action(action)
    self.assertEqual(
        observation["legal_action_labels"],
        [before.action_to_string(1, a) for a in before.legal_actions(1)],
    )
    self.assertEqual(
        observation["observation_string"], before.observation_string(1)
    )
    # The deal: seat 0's card is withheld, seat 1's own is not.
    history = observation["history"]
    self.assertEqual([entry["seat"] for entry in history], [None, None, 0])
    self.assertIsNone(history[0]["action"])
    self.assertIsNone(history[1]["action"])
    self.assertIsNone(history[0]["label"])
    self.assertEqual(history[2]["action"], None)
    self.assertNotIn(before.serialize(), json.dumps(request))

  def test_seat_view_carries_tensors_on_request(self) -> None:
    bot = _Recording()
    session = self.session(
        "kuhn_poker", self.serve(bot), "seat_view", tensors=True
    )
    self.play_kuhn_to_bot(session)
    observation = bot.requests[0][1]["observation"]
    self.assertEqual(
        list(observation),
        list(remote_bot.SEAT_VIEW_OBSERVATION_KEYS + remote_bot.TENSORS_KEYS),
    )
    before = pyspiel.load_game("kuhn_poker").new_initial_state()
    for action in session.history[:3]:
      before.apply_action(action)
    self.assertEqual(
        observation["information_state_tensor"],
        list(before.information_state_tensor(1)),
    )

  def test_full_state_sends_the_state(self) -> None:
    bot = _Recording()
    session = self.session("kuhn_poker", self.serve(bot), "full_state")
    self.play_kuhn_to_bot(session)
    self.assertEqual(len(bot.requests), 1)
    mode, request = bot.requests[0]
    self.assertEqual(mode, "full_state")
    before = pyspiel.load_game("kuhn_poker").new_initial_state()
    for action in session.history[:3]:
      before.apply_action(action)
    self.assertEqual(request["state"], before.serialize())
    self.assertEqual(request["history"], before.history())
    self.assertEqual(request["seat"], 1)
    self.assertEqual(request["game"], "kuhn_poker()")
    self.assertNotIn("protocol", request)
    self.assertEqual(request["game_encoding"], {"num_distinct_actions": 2})

  def test_a_full_state_seat_plays_a_game_that_hides_information(
      self,
  ) -> None:
    game = pyspiel.load_game("kuhn_poker")
    kind = controllers.kinds()[remote_bot.NAME]
    self.assertIsNone(kind.unavailable_reason(game))
    self.assertFalse(kind.reads_full_state)
    config = kind.parse({"url": "http://localhost:1", "state": "full_state"})
    assert kind.config_reads_full_state is not None
    self.assertTrue(kind.config_reads_full_state(config))
    self.assertIsNone(kind.seat_refusal(game, config))

  def test_the_seat_shows_which_state_its_bot_is_sent(self) -> None:
    bot = _Recording()
    session = self.session("kuhn_poker", self.serve(bot), "full_state")
    for viewer in (_ORACLE, _SPECTATOR, _seat(0)):
      self.assertEqual(
          session.frame(viewer).seats[1].config["state"], "full_state"
      )

  def test_the_config_reaches_the_bot(self) -> None:
    bot = _Recording()
    session = self.session(
        "kuhn_poker",
        self.serve(bot),
        "seat_view",
        config='{"checkpoint": "/models/a", "temperature": 0}',
    )
    self.play_kuhn_to_bot(session)
    self.assertEqual(
        bot.requests[0][1]["config"],
        {"checkpoint": "/models/a", "temperature": 0},
    )

  def test_a_notice_reaches_only_the_seat_and_omniscient_viewers(
      self,
  ) -> None:
    bot = _Recording(notice="recovered from a legal-set mismatch")
    session = self.session("kuhn_poker", self.serve(bot), "seat_view")
    self.play_kuhn_to_bot(session)

    def notice(viewer: contract.Viewer) -> str | None:
      return session.frame(viewer).seats[1].notice

    self.assertEqual(notice(_ORACLE), "recovered from a legal-set mismatch")
    self.assertEqual(notice(_seat(1)), "recovered from a legal-set mismatch")
    self.assertIsNone(notice(_seat(0)))
    self.assertIsNone(notice(_SPECTATOR))

  def test_a_notice_has_no_effect_on_play(self) -> None:
    quiet, noisy = _Recording(), _Recording(notice="note")
    a = self.session("kuhn_poker", self.serve(quiet), "seat_view")
    b = self.session("kuhn_poker", self.serve(noisy), "seat_view")
    self.play_kuhn_to_bot(a)
    self.play_kuhn_to_bot(b)
    self.assertEqual(a.history, b.history)
    self.assertEqual(b.frame(_ORACLE).status, a.frame(_ORACLE).status)

  def test_an_undo_clears_the_notice(self) -> None:
    bot = _Recording(notice="note")
    session = self.session("kuhn_poker", self.serve(bot), "seat_view")
    self.play_kuhn_to_bot(session)
    session.undo(_ORACLE)
    self.assertIsNone(session.frame(_ORACLE).seats[1].notice)

  def test_the_server_notice_is_sent_with_every_move(self) -> None:
    bot = _Recording()
    session = self.session(
        "kuhn_poker", self.serve(bot, notice="from --notice"), "seat_view"
    )
    self.play_kuhn_to_bot(session)
    self.assertEqual(session.frame(_ORACLE).seats[1].notice, "from --notice")

  def assert_fails(self, session: session_lib.Session, error: str) -> str:
    """Plays to the bot, and returns its failure message, failing loud."""
    session.apply_action(_seat(0), 0)
    with self.assertRaises(errors.ControllerFailedError):
      session.advance(_ORACLE)
    frame = session.frame(_ORACLE)
    self.assertEqual(frame.status, contract.SessionStatus.PAUSED)
    failure = frame.seats[1].failure
    assert failure is not None
    self.assertEqual(failure.error, error)
    # Other viewers learn only that it failed.
    withheld = session.frame(_seat(0)).seats[1].failure
    assert withheld is not None
    self.assertIsNone(withheld.message)
    assert failure.message is not None
    return failure.message

  def test_a_timeout_fails_the_seat(self) -> None:
    controllers.set_move_budget(2.0)
    bot = _Recording()
    bot.sleep = 3.0
    session = self.session("kuhn_poker", self.serve(bot), "seat_view")
    message = self.assert_fails(session, "RemoteBotUnreachableError")
    self.assertIn("did not answer within 1.5 seconds", message)

  def test_the_timeout_setting_shortens_the_wait(self) -> None:
    bot = _Recording()
    bot.sleep = 1.0
    session = self.session(
        "kuhn_poker", self.serve(bot), "seat_view", timeout_seconds=0.2
    )
    message = self.assert_fails(session, "RemoteBotUnreachableError")
    self.assertIn("within 0.2 seconds", message)

  def test_an_illegal_action_fails_the_seat(self) -> None:
    bot = _Recording()
    bot.reply = 7
    session = self.session("kuhn_poker", self.serve(bot), "seat_view")
    message = self.assert_fails(session, "RemoteBotReplyError")
    self.assertIn("chose 7", message)

  def test_a_reply_without_an_action_fails_the_seat(self) -> None:
    url = self.serve_raw(200, b'{"move": 1}')
    session = self.session("kuhn_poker", url, "seat_view")
    self.assert_fails(session, "RemoteBotReplyError")

  def test_a_reply_that_is_not_json_fails_the_seat(self) -> None:
    url = self.serve_raw(200, b"<html>")
    session = self.session("kuhn_poker", url, "seat_view")
    self.assert_fails(session, "RemoteBotReplyError")

  def test_a_bad_notice_fails_the_seat(self) -> None:
    url = self.serve_raw(
        200, json.dumps({"action": 0, "notice": "x" * 201}).encode()
    )
    session = self.session("kuhn_poker", url, "seat_view")
    self.assert_fails(session, "RemoteBotReplyError")

  def test_a_refusal_fails_the_seat_with_its_reason(self) -> None:
    url = self.serve_raw(
        422, b'{"error": "checkpoint not loaded", "reason": "not_ready"}'
    )
    session = self.session("kuhn_poker", url, "seat_view")
    message = self.assert_fails(session, "RemoteBotRefusedError")
    self.assertIn("checkpoint not loaded; not_ready", message)

  def test_an_error_in_a_2xx_reply_is_a_refusal(self) -> None:
    url = self.serve_raw(200, b'{"error": "no"}')
    session = self.session("kuhn_poker", url, "seat_view")
    self.assert_fails(session, "RemoteBotRefusedError")

  def test_a_server_error_fails_the_seat(self) -> None:
    url = self.serve_raw(503, b"{}")
    session = self.session("kuhn_poker", url, "seat_view")
    message = self.assert_fails(session, "RemoteBotUnreachableError")
    self.assertIn("HTTP 503", message)

  def test_a_line_of_actions_plays_its_first(self) -> None:
    bot = _Recording()
    bot.reply = {"actions": [1, 0]}
    session = self.session("kuhn_poker", self.serve(bot), "seat_view")
    self.play_kuhn_to_bot(session)
    self.assertEqual(session.history[3], 1)

  def test_unknown_reply_fields_are_ignored(self) -> None:
    bot = _Recording()
    bot.reply = {"action": 1, "action_distribution": [], "decider": "x"}
    session = self.session("kuhn_poker", self.serve(bot), "seat_view")
    self.play_kuhn_to_bot(session)
    self.assertEqual(session.history[3], 1)

  def create(self, game: str, **config: Any) -> controllers.Controller:
    """Builds a remote bot for `game` with `config`, at once."""
    return controllers.create(
        controllers.Spec(remote_bot.NAME, config),
        pyspiel.load_game(game),
        seed=1,
    )

  def test_a_bot_that_takes_only_the_full_state_refuses_a_seat_view(
      self,
  ) -> None:
    url = self.serve(_Recording(modes=("full_state",)))
    with self.assertRaisesRegex(
        controllers.UnavailableError, "takes full_state, not seat_view"
    ):
      self.create("kuhn_poker", url=url)
    self.assertIsNotNone(self.create("kuhn_poker", url=url, state="full_state"))

  def test_a_bot_server_that_is_not_running_is_not_built(self) -> None:
    url = self.serve(_Recording())
    port = int(url.rsplit(":", 1)[1])
    self.doCleanups()
    with self.assertRaisesRegex(
        controllers.UnavailableError, "is not ready: .*cannot be reached"
    ):
      self.create("kuhn_poker", url=f"http://127.0.0.1:{port}")

  def test_a_seat_given_an_unreachable_bot_is_refused_not_a_fault(
      self,
  ) -> None:
    url = self.serve(_Recording())
    port = int(url.rsplit(":", 1)[1])
    self.doCleanups()
    session = session_lib.Session.create(
        "kuhn_poker",
        {},
        seed=3,
        chance_policy=contract.ChancePolicy.AUTO,
        controllers=[controllers.HUMAN_SPEC, controllers.HUMAN_SPEC],
    )
    with self.assertRaisesRegex(
        errors.ControllerUnavailableError, "is not ready"
    ):
      session.set_controller(
          _ORACLE,
          1,
          controllers.Spec(
              remote_bot.NAME, {"url": f"http://127.0.0.1:{port}"}
          ),
      )

  def test_another_host_is_refused_unless_allowed(self) -> None:
    with self.assertRaisesRegex(
        controllers.UnavailableError, "example.com is not one"
    ):
      controllers.builder(
          controllers.Spec(remote_bot.NAME, {"url": "http://example.com"}),
          pyspiel.load_game("kuhn_poker"),
      )
    remote_bot.configure(hosts=["example.com"])
    controllers.builder(
        controllers.Spec(remote_bot.NAME, {"url": "http://example.com:9"}),
        pyspiel.load_game("kuhn_poker"),
    )
    remote_bot.configure(hosts=["*"])
    controllers.builder(
        controllers.Spec(remote_bot.NAME, {"url": "https://bots.test/x"}),
        pyspiel.load_game("kuhn_poker"),
    )

  def test_loopback_urls_are_allowed(self) -> None:
    for url in (
        "http://localhost:9001",
        "http://127.0.0.1:9001",
        "http://[::1]:9001",
    ):
      with self.subTest(url):
        self.assertIsNone(remote_bot.url_refusal(url))

  def test_a_url_with_a_query_or_credentials_is_refused(self) -> None:
    for url in (
        "http://localhost:9001/?a=1",
        "http://user:pw@localhost:9001",
        "http://localhost:x",
        "ftp://localhost",
    ):
      with self.subTest(url):
        self.assertIsNotNone(remote_bot.url_refusal(url))

  def test_a_config_that_is_not_a_json_object_is_refused(self) -> None:
    for config in ("[1]", "{", '"a"'):
      with self.subTest(config):
        with self.assertRaisesRegex(controllers.UnavailableError, "config"):
          controllers.builder(
              controllers.Spec(
                  remote_bot.NAME,
                  {"url": "http://localhost:1", "config": config},
              ),
              pyspiel.load_game("kuhn_poker"),
          )

  def test_a_redirect_is_not_followed(self) -> None:
    url = self.serve_raw(302, b"{}")
    session = self.session("kuhn_poker", url, "seat_view")
    self.assert_fails(session, "RemoteBotUnreachableError")

  def serve_recording_headers(self) -> tuple[str, list[str | None]]:
    """Serves a bot that records each move's Authorization header."""
    seen: list[str | None] = []

    class Handler(_RawHandler):
      """Records each move request's Authorization header."""

      def do_POST(self) -> None:
        seen.append(self.headers.get("Authorization"))
        super().do_POST()

    server = _RawServer(200, b'{"action": 0}')
    server.RequestHandlerClass = Handler
    threading.Thread(target=server.serve_forever, daemon=True).start()
    self.addCleanup(server.server_close)
    self.addCleanup(server.shutdown)
    host, port = server.server_address[:2]
    return f"http://{host!s}:{port}", seen

  def test_the_token_reaches_only_its_host_and_never_the_config(
      self,
  ) -> None:
    url, seen = self.serve_recording_headers()
    port = url.rsplit(":", 1)[1]
    remote_bot.configure(tokens={f"127.0.0.1:{port}": "s3cret"})
    session = self.session("kuhn_poker", url, "seat_view")
    self.play_kuhn_to_bot(session)
    self.assertEqual(seen, ["Bearer s3cret"])
    self.assertNotIn(
        "s3cret", json.dumps(contract.to_json(session.frame(_ORACLE)))
    )

  def test_a_bot_elsewhere_gets_no_token(self) -> None:
    url, seen = self.serve_recording_headers()
    # The token is for another port of this host, and for another host.
    remote_bot.configure(tokens={"127.0.0.1:1": "a", "bots.lan": "b"})
    session = self.session("kuhn_poker", url, "seat_view")
    self.play_kuhn_to_bot(session)
    self.assertEqual(seen, [None])

  def test_a_token_for_every_host_is_refused(self) -> None:
    for where in ("*", " "):
      with self.subTest(where):
        with self.assertRaises(ValueError):
          remote_bot.configure(hosts=["*"], tokens={where: "t"})


if __name__ == "__main__":
  absltest.main()
