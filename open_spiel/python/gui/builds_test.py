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

"""Tests for controller builds, alone and through the server.

A session's builds are tested in session_builds_test.py.
"""

import http
import http.client
import json
import pathlib
import threading
import time
from typing import Any

from absl.testing import absltest

from open_spiel.python.gui import api
from open_spiel.python.gui import builds
from open_spiel.python.gui import builds_testing
from open_spiel.python.gui import catalogue as catalogue_lib
from open_spiel.python.gui import contract
from open_spiel.python.gui import controllers
from open_spiel.python.gui import health as health_lib
from open_spiel.python.gui import server as server_lib
from open_spiel.python.gui import static
from open_spiel.python.gui import store as store_lib

_JSON = {"Content-Type": "application/json"}


class BuildTest(builds_testing.BuildsTestCase):

  def test_the_defaults(self) -> None:
    self.assertEqual(builds.DEFAULT_BUDGET_SECONDS, 120.0)
    self.assertEqual(builds.DEFAULT_WAIT_SECONDS, 0.25)

  def test_refuses_settings_that_are_not_finite(self) -> None:
    for value in (0.0, -1.0, float("inf"), float("nan")):
      with self.subTest(value=value):
        with self.assertRaisesRegex(ValueError, "positive and finite"):
          builds.set_budget_seconds(value)
    for value in (-1.0, float("inf"), float("nan")):
      with self.subTest(value=value):
        with self.assertRaisesRegex(ValueError, "non-negative and finite"):
          builds.set_wait_seconds(value)
    builds.set_wait_seconds(0)
    self.assertEqual(builds.budget_seconds(), builds_testing.BUDGET_SECONDS)

  def test_finishes_with_what_the_call_returns_and_tells_once(self) -> None:
    done: list[builds.Build[int]] = []
    build = builds.Build(lambda: 7, name="test", on_done=done.append)
    self.assertFalse(build.done)
    with self.assertRaisesRegex(RuntimeError, "not finished"):
      build.outcome()
    build.start()
    self.assertTrue(build.wait(builds_testing.PATIENCE_SECONDS))
    self.assertEqual(build.outcome(), (7, None))
    builds_testing.wait_for(lambda: bool(done), "on_done")
    self.assertEqual(done, [build])
    with self.assertRaisesRegex(RuntimeError, "started already"):
      build.start()

  def test_finishes_with_what_the_call_raises(self) -> None:
    def fail() -> int:
      raise KeyError("lost")

    build = builds.Build(fail, name="test", on_done=lambda _: None)
    build.start()
    self.assertTrue(build.wait(builds_testing.PATIENCE_SECONDS))
    result, error = build.outcome()
    self.assertIsNone(result)
    self.assertIsInstance(error, KeyError)

  def test_fails_over_budget_and_drops_the_late_result(self) -> None:
    builds.set_budget_seconds(0.1)
    gate = threading.Event()
    self.addCleanup(gate.set)
    done: list[builds.Build[int]] = []

    def slow() -> int:
      gate.wait()
      return 7

    build = builds.Build(slow, name="test", on_done=done.append)
    build.start()
    self.assertTrue(build.wait(builds_testing.PATIENCE_SECONDS))
    result, error = build.outcome()
    self.assertIsNone(result)
    assert isinstance(error, builds.OverBudgetError)
    self.assertEqual(error.seconds, 0.1)
    gate.set()
    time.sleep(0.1)
    self.assertEqual(build.outcome(), (None, error))
    self.assertEqual(done, [build])


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


class ServedBuildTest(builds_testing.BuildsTestCase):
  """The server answers while a seat's controller builds."""

  def setUp(self) -> None:
    super().setUp()
    self.gate, self.built = self.register_gated("gated")
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

  def test_a_preparing_seat_is_served_and_plays_once_built(self) -> None:
    gated = {"controller": "gated", "config": {}, "autoplay": True}
    human = {"controller": "human", "config": {}, "autoplay": False}
    status, frame = self.request_json(
        "POST",
        "/api/sessions?viewer=1",
        {
            "game": "tic_tac_toe",
            "parameters": {},
            "seats": [gated, human],
            "chance_policy": "auto",
            "seed": None,
        },
    )
    self.assertEqual(status, http.HTTPStatus.OK, frame)
    self.assertEqual(frame["contract_version"], contract.CONTRACT_VERSION)
    self.assertEqual(frame["seats"][0]["status"], "preparing")
    self.assertEqual(
        frame["seats"][0]["message"], "Building Gated, for at most 30 seconds."
    )
    self.assertEqual(frame["seats"][1]["status"], "ready")
    self.assertIsNone(frame["seats"][1]["message"])
    session = frame["session"]["id"]
    status, health = self.request_json("GET", "/api/health")
    self.assertEqual(status, http.HTTPStatus.OK, health)
    status, frame = self.request_json(
        "GET", f"/api/sessions/{session}?viewer=1"
    )
    self.assertEqual(status, http.HTTPStatus.OK, frame)
    self.assertEqual(frame["seats"][0]["status"], "preparing")
    self.gate.set()

    def played() -> bool:
      status, frame = self.request_json(
          "GET", f"/api/sessions/{session}?viewer=1"
      )
      assert status == http.HTTPStatus.OK, frame
      return bool(frame["num_plies"] == 1)

    builds_testing.wait_for(played, "autoplay")
    status, frame = self.request_json(
        "GET", f"/api/sessions/{session}?viewer=1"
    )
    self.assertEqual(frame["seats"][0]["status"], "ready")
    self.assertEqual(self.built[0].moves, 1)


if __name__ == "__main__":
  absltest.main()
