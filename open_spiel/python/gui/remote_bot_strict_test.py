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

"""Tests that a seat-view request is what a strict seat-view bot accepts.

A strict seat-view engine takes the acting seat's information state string,
its legal actions and a seed, and refuses, with HTTP 400, a request with any
other key: a state or a history would hand it hidden information. The fake
here enforces that schema, key for key, and the seats play it.
"""

import http.server
import json
import threading
from typing import Any

from absl.testing import absltest

import pyspiel

from open_spiel.python.gui import builds
from open_spiel.python.gui import contract
from open_spiel.python.gui import controllers
from open_spiel.python.gui import errors
from open_spiel.python.gui import remote_bot
from open_spiel.python.gui import remote_bot_http
from open_spiel.python.gui import session as session_lib

_ORACLE = contract.Viewer(contract.ViewerKind.ORACLE)


# The request schema of a strict seat-view bot: it takes the acting seat's
# information state string, never a state or a history, and refuses, with
# HTTP 400, a request with any key but these.
_STRICT_REQUEST_KEYS = frozenset(
    {"game", "observation", "config", "game_encoding", "rules_fingerprint"}
)
_STRICT_OBSERVATION_KEYS = frozenset(
    {"information_state_string", "seat", "legal_actions", "seed"}
)
_STRICT_MAX_SEED = 2**63 - 1


def _strict_refusal(request: Any) -> str | None:
  """Returns why a strict seat-view engine refuses `request`, or None."""
  if not isinstance(request, dict):
    return "the request body must be a JSON object"
  extra = sorted(set(request) - _STRICT_REQUEST_KEYS)
  if extra:
    return f"refusing request keys {extra}"
  game_string = request.get("game")
  if not isinstance(game_string, str) or not game_string:
    return "'game' must be the session's game string"
  observation = request.get("observation")
  if not isinstance(observation, dict):
    return "'observation' must be an object"
  extra = sorted(set(observation) - _STRICT_OBSERVATION_KEYS)
  if extra:
    return f"refusing observation keys {extra}"
  missing = sorted(_STRICT_OBSERVATION_KEYS - set(observation))
  if missing:
    return f"'observation' is missing {missing}"
  info = observation["information_state_string"]
  if not isinstance(info, str) or not info:
    return "information_state_string must be a non-empty string"

  def is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)

  game = pyspiel.load_game(game_string)
  seat = observation["seat"]
  if not is_int(seat) or not 0 <= seat < game.num_players():
    return "observation.seat must be a seat of the game"
  legal = observation["legal_actions"]
  if (
      not isinstance(legal, list)
      or not legal
      or not all(is_int(a) for a in legal)
      or len(set(legal)) != len(legal)
      or not all(0 <= a < game.num_distinct_actions() for a in legal)
  ):
    return "observation.legal_actions must be distinct action ids"
  seed = observation["seed"]
  if not is_int(seed) or not 0 <= seed <= _STRICT_MAX_SEED:
    return "observation.seed must be an integer in [0, 2**63 - 1]"
  config = request.get("config")
  if config is not None and not isinstance(config, dict):
    return "'config' must be a JSON object"
  if "game_encoding" in request:
    encoding = request["game_encoding"]
    if not isinstance(encoding, dict) or set(encoding) != {
        "num_distinct_actions"
    }:
      return "'game_encoding' must be {\"num_distinct_actions\": <int>}"
    if encoding["num_distinct_actions"] != game.num_distinct_actions():
      return "the action encoding differs from the engine's"
  return None


class _StrictHandler(http.server.BaseHTTPRequestHandler):
  """A strict seat-view engine: /health, and /spiel/observe only."""

  server: "_StrictServer"

  def log_message(self, format: str, *args: object) -> None:
    """Logs nothing."""
    del format, args

  def _answer(self, status: int, body: dict[str, Any]) -> None:
    """Sends `body` with `status`."""
    data = json.dumps(body).encode()
    self.send_response(status)
    self.send_header("Content-Type", "application/json")
    self.send_header("Content-Length", str(len(data)))
    self.end_headers()
    self.wfile.write(data)

  def do_GET(self) -> None:  # pylint: disable=invalid-name
    """Answers /health; it has no /info."""
    if self.path == "/health":
      self._answer(200, {"ok": True})
    else:
      self._answer(404, {"error": f"no {self.path}"})

  def do_POST(self) -> None:  # pylint: disable=invalid-name
    """Answers /spiel/observe with the lowest legal action, or a 400."""
    request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
    if self.path != "/spiel/observe":
      self._answer(404, {"error": f"no {self.path}"})
      return
    reason = _strict_refusal(request)
    if reason is not None:
      self.server.refusals.append(reason)
      self._answer(400, {"error": reason})
      return
    self.server.accepted.append(request)
    # Its replies carry more than the action, which the GUI ignores.
    self._answer(
        200,
        {
            "action": min(request["observation"]["legal_actions"]),
            "decider": "network",
            "latency_ms": 1.0,
        },
    )


class _StrictServer(http.server.ThreadingHTTPServer):
  """Serves a strict seat-view engine, keeping what it accepts and refuses."""

  daemon_threads = True

  def __init__(self) -> None:
    super().__init__(("127.0.0.1", 0), _StrictHandler)
    self.accepted: list[dict[str, Any]] = []
    self.refusals: list[str] = []


class StrictSeatViewTest(absltest.TestCase):

  def setUp(self) -> None:
    super().setUp()
    self.addCleanup(controllers.set_move_budget, controllers.move_budget())
    self.addCleanup(builds.set_wait_seconds, builds.wait_seconds())
    builds.set_wait_seconds(10.0)
    self.addCleanup(remote_bot.configure)

  def serve_strict(self) -> tuple[_StrictServer, str]:
    """Serves a strict seat-view engine; returns it and its URL."""
    server = _StrictServer()
    threading.Thread(target=server.serve_forever, daemon=True).start()
    self.addCleanup(server.server_close)
    self.addCleanup(server.shutdown)
    host, port = server.server_address[:2]
    return server, f"http://{host!s}:{port}"

  def session(self, game: str, url: str, **config: Any) -> session_lib.Session:
    """Opens `game`, every seat a seat-view remote bot at `url`."""
    num = pyspiel.load_game(game).num_players()
    spec = controllers.Spec(
        remote_bot.NAME, {"url": url, "state": "seat_view", **config}
    )
    return session_lib.Session.create(
        game,
        {},
        seed=3,
        chance_policy=contract.ChancePolicy.AUTO,
        controllers=[spec] * num,
    )

  def test_the_request_is_the_strict_shape(self) -> None:
    engine, url = self.serve_strict()
    session = self.session("kuhn_poker", url)
    session.advance(_ORACLE, max_plies=1)
    self.assertEqual(len(engine.accepted), 1)
    request = engine.accepted[0]
    self.assertEqual(list(request), list(remote_bot.SEAT_VIEW_KEYS))
    self.assertEqual(
        list(request["observation"]),
        list(remote_bot.SEAT_VIEW_OBSERVATION_KEYS),
    )
    self.assertIsNone(_strict_refusal(request))

  def test_a_strict_seat_view_engine_plays_every_seat(self) -> None:
    # A strict seat-view bot plays hearts and gin rummy here; every seat is
    # sent exactly the keys and nesting such a bot accepts.
    for game in ("hearts", "gin_rummy"):
      with self.subTest(game):
        engine, url = self.serve_strict()
        session = self.session(game, url)
        try:
          session.advance(_ORACLE, max_plies=12)
        except errors.ControllerFailedError:
          pass  # Said below, with the engine's reason.
        frame = session.frame(_ORACLE)
        self.assertEqual(engine.refusals, [])
        self.assertEqual(
            [seat.failure for seat in frame.seats], [None] * len(frame.seats)
        )
        moves = [ply for ply in frame.history if ply.seat is not None]
        # Twelve moves, or fewer to the game's end.
        self.assertEqual(len(engine.accepted), len(moves))
        if frame.status != contract.SessionStatus.TERMINAL:
          self.assertEqual(len(moves), 12)
        self.assertGreaterEqual(len(moves), 4)
        state = pyspiel.load_game(game).new_initial_state()
        asked = iter(engine.accepted)
        for ply in frame.history:
          if ply.seat is not None:
            observation = next(asked)["observation"]
            self.assertEqual(observation["seat"], ply.seat)
            self.assertEqual(
                observation["information_state_string"],
                state.information_state_string(ply.seat),
            )
            self.assertEqual(
                observation["legal_actions"], state.legal_actions(ply.seat)
            )
          state.apply_action(ply.action)
        seats = {request["observation"]["seat"] for request in engine.accepted}
        self.assertGreater(len(seats), 1)

  def test_a_strict_seat_view_engine_refuses_extras(self) -> None:
    # Extras are for bots that want them: a strict engine refuses them, and
    # the seat fails with the engine's reason.
    for extra in ("extras", "tensors"):
      with self.subTest(extra):
        engine, url = self.serve_strict()
        session = self.session("kuhn_poker", url, **{extra: True})
        with self.assertRaises(errors.ControllerFailedError):
          session.advance(_ORACLE, max_plies=1)
        self.assertEqual(engine.accepted, [])
        self.assertEqual(len(engine.refusals), 1)
        self.assertIn("refusing observation keys", engine.refusals[0])
        failure = session.frame(_ORACLE).seats[0].failure
        assert failure is not None and failure.message is not None
        self.assertEqual(failure.error, "RemoteBotRefusedError")
        self.assertIn("refusing observation keys", failure.message)

  def test_every_request_names_the_protocol_in_a_header(self) -> None:
    seen: list[str | None] = []

    class Handler(_StrictHandler):
      """The strict engine, noting each request's protocol header."""

      def do_GET(self) -> None:
        seen.append(self.headers.get(remote_bot_http.PROTOCOL_HEADER))
        super().do_GET()

      def do_POST(self) -> None:
        seen.append(self.headers.get(remote_bot_http.PROTOCOL_HEADER))
        super().do_POST()

    engine, url = self.serve_strict()
    engine.RequestHandlerClass = Handler
    session = self.session("kuhn_poker", url)
    session.advance(_ORACLE, max_plies=1)
    self.assertEqual(len(engine.accepted), 1)
    self.assertGreaterEqual(len(seen), 3)  # /health, /info, then a move.
    self.assertEqual(set(seen), {str(remote_bot.PROTOCOL)})


if __name__ == "__main__":
  absltest.main()
