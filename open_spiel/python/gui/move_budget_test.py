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


"""Tests for the move budget, alone and through a served session."""

from collections.abc import Mapping
import http
import http.client
import json
import pathlib
import threading
import time
from typing import Any
from unittest import mock

from absl.testing import absltest

import pyspiel

from open_spiel.python.gui import api
from open_spiel.python.gui import catalogue as catalogue_lib
from open_spiel.python.gui import contract
from open_spiel.python.gui import controllers
from open_spiel.python.gui import health as health_lib
from open_spiel.python.gui import mcts_controller
from open_spiel.python.gui import mcts_search
from open_spiel.python.gui import move_budget
from open_spiel.python.gui import server as server_lib
from open_spiel.python.gui import static
from open_spiel.python.gui import store as store_lib

# The simulations MCTS is capped at when the C++ search gives it no clock.
_CAP = mcts_controller.SIMULATIONS[contract.MctsLimits.SIMULATIONS_ONLY].maximum

# Short, so the tests run fast; long enough that a loopback request beats it.
_BUDGET_SECONDS = 1.0

# How soon after its budget expires a call that checks must have stopped and
# its thread exited: many simulations' time, so a slow machine passes.
_STOP_BOUND_SECONDS = 3.0

_JSON = {"Content-Type": "application/json"}


class _Stalls:
  """A controller that blocks until released, then plays its first action."""

  def __init__(self) -> None:
    self.asked = threading.Event()
    self.release = threading.Event()

  def choose_action(self, state: pyspiel.State, seat: int) -> int:
    """Waits for `release`, then returns `seat`'s first legal action."""
    self.asked.set()
    self.release.wait()
    return int(state.legal_actions(seat)[0])


class _BudgetTestCase(absltest.TestCase):
  """Sets a short move budget for the length of each test."""

  def setUp(self) -> None:
    super().setUp()
    self.addCleanup(controllers.set_move_budget, controllers.move_budget())
    controllers.set_move_budget(_BUDGET_SECONDS)

  def stalls(self) -> _Stalls:
    """Returns a stalling controller, released when the test ends."""
    controller = _Stalls()
    self.addCleanup(controller.release.set)
    return controller


def _started_since(before: set[threading.Thread]) -> list[threading.Thread]:
  """Returns the threads alive now that were not in `before`."""
  return [t for t in threading.enumerate() if t not in before]


class MoveBudgetTest(_BudgetTestCase):

  def assert_exits_promptly(self, threads: list[threading.Thread]) -> None:
    """Asserts that every one of `threads` exits within the stop bound."""
    deadline = time.monotonic() + _STOP_BOUND_SECONDS
    for thread in threads:
      thread.join(timeout=max(0.0, deadline - time.monotonic()))
    self.assertEqual([t.name for t in threads if t.is_alive()], [])

  def test_the_default_is_ten_seconds(self) -> None:
    self.assertEqual(controllers.DEFAULT_MOVE_BUDGET_SECONDS, 10.0)

  def test_refuses_a_budget_that_is_not_positive_and_finite(self) -> None:
    for value in (0.0, -1.0, float("inf"), float("nan")):
      with self.subTest(value=value):
        with self.assertRaisesRegex(ValueError, "positive and finite"):
          controllers.set_move_budget(value)
    self.assertEqual(controllers.move_budget(), _BUDGET_SECONDS)

  def test_returns_what_the_call_returns(self) -> None:
    self.assertEqual(move_budget.run(lambda: 7, name="test"), 7)

  def test_raises_what_the_call_raises(self) -> None:
    def fail() -> int:
      raise KeyError("lost")

    with self.assertRaises(KeyError):
      move_budget.run(fail, name="test")

  def test_check_passes_on_a_thread_run_did_not_start(self) -> None:
    self.assertFalse(move_budget.stop_requested())
    move_budget.check()

  def test_check_passes_within_the_budget(self) -> None:
    def checks() -> bool:
      move_budget.check()
      return move_budget.stop_requested()

    self.assertFalse(move_budget.run(checks, name="test"))

  def test_a_call_over_budget_is_asked_to_stop_and_its_thread_exits(
      self,
  ) -> None:
    stopped = threading.Event()

    def checks_forever() -> None:
      try:
        while True:
          move_budget.check()
          time.sleep(0.01)
      except move_budget.StoppedError:
        stopped.set()
        raise

    before = set(threading.enumerate())
    with self.assertRaises(move_budget.OverBudgetError):
      move_budget.run(checks_forever, name="test-checks-forever")
    self.assertTrue(stopped.wait(timeout=_STOP_BOUND_SECONDS))
    self.assert_exits_promptly(_started_since(before))

  def test_a_python_mcts_search_over_budget_stops_and_its_thread_exits(
      self,
  ) -> None:
    # The Python search, which games implemented in Python get, is made to
    # search hive: its capped simulations, each a random rollout of a whole
    # game, take several times the budget (the C++ search alone runs about
    # 94 a second).
    game = pyspiel.load_game("hive")
    spec = controllers.Spec(
        controllers.MCTS,
        {"max_simulations": _CAP, "solve": False},
    )
    with mock.patch.object(
        mcts_search, "implemented_in_python", return_value=True
    ):
      bot = controllers.create(spec, game, seed=0)
    state = game.new_initial_state()
    seat = state.current_player()
    before = set(threading.enumerate())
    with self.assertRaises(controllers.ControllerError) as raised:
      controllers.choose(bot, state, seat)
    timed_out = time.monotonic()
    self.assertEqual(raised.exception.error, controllers.TIMEOUT)
    searches = _started_since(before)
    self.assertEqual([t.name for t in searches], [f"controller-seat-{seat}"])
    self.assert_exits_promptly(searches)
    self.assertLess(time.monotonic() - timed_out, _STOP_BOUND_SECONDS)

  def test_a_controller_over_budget_fails_its_seat(self) -> None:
    state = pyspiel.load_game("tic_tac_toe").new_initial_state()
    start = time.monotonic()
    with self.assertRaises(controllers.ControllerError) as raised:
      controllers.choose(self.stalls(), state, 0)
    elapsed = time.monotonic() - start
    self.assertEqual(raised.exception.seat, 0)
    self.assertEqual(raised.exception.error, controllers.TIMEOUT)
    self.assertIn("1 second move budget", raised.exception.detail)
    self.assertIsInstance(
        raised.exception.__cause__, move_budget.OverBudgetError
    )
    self.assertGreaterEqual(elapsed, _BUDGET_SECONDS)
    self.assertLess(elapsed, _BUDGET_SECONDS + 5)
    self.assertEqual(
        raised.exception.failure(visible=False),
        contract.ControllerFailure(error=controllers.TIMEOUT, message=None),
    )

  def test_a_controller_within_budget_plays(self) -> None:
    controller = self.stalls()
    controller.release.set()
    state = pyspiel.load_game("tic_tac_toe").new_initial_state()
    self.assertEqual(controllers.choose(controller, state, 0), 0)


def _catalogue() -> catalogue_lib.Catalogue:
  """Returns an empty catalogue."""
  return catalogue_lib.Catalogue(games=())


def _health() -> health_lib.Health:
  """Returns a health report naming every registered controller."""
  return health_lib.Health(
      version="0.0.0",
      revision=None,
      capabilities=health_lib.Capabilities(
          controllers=controllers.registered(),
          chance_policies=tuple(contract.ChancePolicy),
          cpp_mcts_limits=controllers.mcts_limits(),
      ),
  )


class ServedBudgetTest(_BudgetTestCase):
  """A stalling seat fails within the budget while the server answers."""

  def setUp(self) -> None:
    super().setUp()
    self.controller = self.stalls()
    stalling = self.controller

    def build(
        game: pyspiel.Game,
        config: Mapping[str, contract.ParameterValue],
        seed: int,
    ) -> controllers.Controller:
      del game, config, seed
      return stalling

    controllers.register("stalls", controllers.Kind("Stalls", (), False, build))
    self.addCleanup(controllers.unregister, "stalls")
    root = pathlib.Path(self.create_tempdir().full_path)
    self.server = server_lib.Server(
        ("127.0.0.1", 0),
        api.Api(store_lib.SessionStore(), catalogue=_catalogue, health=_health),
        static.ClientFiles(root),
        server_lib.LOOPBACK_HOSTS,
    )
    thread = threading.Thread(target=self.server.serve_forever)
    thread.start()

    def stop() -> None:
      self.server.shutdown()
      thread.join()
      self.server.server_close()

    self.addCleanup(stop)

  def request_json(
      self, method: str, path: str, body: object = None
  ) -> tuple[int, Any]:
    """Sends a JSON request, returning the status and the decoded body."""
    host, port = self.server.server_address[:2]
    connection = http.client.HTTPConnection(str(host), port, timeout=30)
    try:
      raw = None if body is None else json.dumps(body).encode()
      connection.request(method, path, raw, _JSON)
      response = connection.getresponse()
      return response.status, json.loads(response.read())
    finally:
      connection.close()

  def test_a_stalled_seat_fails_and_the_server_answers_meanwhile(
      self,
  ) -> None:
    human = {"controller": "human", "config": {}, "autoplay": False}
    stalls = {"controller": "stalls", "config": {}, "autoplay": True}
    status, frame = self.request_json(
        "POST",
        "/api/sessions?viewer=oracle",
        {
            "game": "tic_tac_toe",
            "parameters": {},
            "seats": [human, stalls],
            "chance_policy": "auto",
            "seed": None,
        },
    )
    self.assertEqual(status, http.HTTPStatus.OK, frame)
    session = frame["session"]["id"]
    played: dict[str, tuple[int, Any]] = {}

    def play() -> None:
      # Seat 0's move hands the turn to seat 1, which then stalls.
      played["response"] = self.request_json(
          "POST",
          f"/api/sessions/{session}/actions?viewer=0",
          {"action": 4, "ply": None},
      )

    player = threading.Thread(target=play)
    player.start()
    self.assertTrue(self.controller.asked.wait(timeout=30))
    status, health = self.request_json("GET", "/api/health")
    self.assertEqual(status, http.HTTPStatus.OK)
    self.assertIn(
        "stalls", {c["name"] for c in health["capabilities"]["controllers"]}
    )
    # The health check came back while the move still waited on seat 1.
    self.assertTrue(player.is_alive())
    player.join(timeout=30)
    self.assertFalse(player.is_alive())
    status, error = played["response"]
    self.assertEqual(status, http.HTTPStatus.CONFLICT)
    self.assertEqual(error["code"], "controller_failed")
    self.assertEqual(error["seat"], 1)
    status, frame = self.request_json(
        "GET", f"/api/sessions/{session}?viewer=oracle"
    )
    self.assertEqual(status, http.HTTPStatus.OK)
    self.assertEqual(frame["seats"][1]["failure"]["error"], "Timeout")


if __name__ == "__main__":
  absltest.main()
