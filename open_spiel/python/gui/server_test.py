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


"""Tests for the HTTP server, over a real loopback socket."""

import http
import http.client
import json
import pathlib
import socket
import threading
from typing import Any
from unittest import mock

from absl.testing import absltest

from open_spiel.python.gui import api
from open_spiel.python.gui import catalogue as catalogue_lib
from open_spiel.python.gui import contract
from open_spiel.python.gui import controllers
from open_spiel.python.gui import health as health_lib
from open_spiel.python.gui import server as server_lib
from open_spiel.python.gui import static
from open_spiel.python.gui import store as store_lib

_HUMAN_SEAT = {"controller": "human", "config": {}, "autoplay": False}
_CREATE: dict[str, object] = {
    "game": "tic_tac_toe",
    "parameters": {},
    "seats": [_HUMAN_SEAT, _HUMAN_SEAT],
    "chance_policy": "auto",
    "seed": None,
}
_JSON = {"Content-Type": "application/json"}


def _catalogue() -> catalogue_lib.Catalogue:
  """Returns an empty catalogue; the API's tests cover its content."""
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


class ServerTest(absltest.TestCase):

  def setUp(self) -> None:
    super().setUp()
    root = pathlib.Path(self.create_tempdir().full_path)
    (root / "index.html").write_text("<html></html>")
    self.store = store_lib.SessionStore()
    self.server = server_lib.Server(
        ("127.0.0.1", 0),
        api.Api(self.store, catalogue=_catalogue, health=_health),
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

  def request(
      self,
      method: str,
      path: str,
      body: object = None,
      headers: dict[str, str] | None = None,
  ) -> tuple[int, dict[str, str], bytes]:
    """Sends a request, returning the status, headers and content."""
    host, port = self.server.server_address[:2]
    connection = http.client.HTTPConnection(str(host), port)
    self.addCleanup(connection.close)
    raw = None if body is None else json.dumps(body).encode()
    connection.request(method, path, raw, headers or {})
    response = connection.getresponse()
    return response.status, dict(response.getheaders()), response.read()

  def request_json(
      self, method: str, path: str, body: object = None
  ) -> tuple[int, Any]:
    """Sends a JSON request, returning the status and the decoded body."""
    status, headers, content = self.request(method, path, body, _JSON)
    self.assertEqual(headers["Content-Type"], "application/json")
    return status, json.loads(content)

  def test_url_names_the_bound_port(self) -> None:
    port = self.server.server_address[1]
    self.assertEqual(self.server.url, f"http://127.0.0.1:{port}/")

  def test_serves_the_api_as_json(self) -> None:
    status, catalogue = self.request_json("GET", "/api/games")
    self.assertEqual(status, http.HTTPStatus.OK)
    self.assertEqual(catalogue["games"], [])

  def test_plays_a_game_end_to_end(self) -> None:
    _, frame = self.request_json("POST", "/api/sessions?viewer=0", _CREATE)
    session = frame["session"]["id"]
    # Seat 0 takes the top row while seat 1 plays the middle row.
    for ply, action in enumerate((0, 3, 1, 4, 2)):
      status, frame = self.request_json(
          "POST",
          f"/api/sessions/{session}/actions?viewer={ply % 2}",
          {"action": action, "ply": None},
      )
      self.assertEqual(status, http.HTTPStatus.OK)
    self.assertEqual(frame["status"], "terminal")
    self.assertEqual(frame["returns"], [1.0, -1.0])

  def test_serves_each_viewer_its_observation_struct(self) -> None:
    _, frame = self.request_json("POST", "/api/sessions?viewer=1", _CREATE)
    session = frame["session"]["id"]
    self.request_json(
        "POST",
        f"/api/sessions/{session}/actions?viewer=0",
        {"action": 4, "ply": None},
    )
    _, seat = self.request_json("GET", f"/api/sessions/{session}?viewer=1")
    self.assertEqual(seat["contract_version"], 7)
    self.assertEqual(seat["observation_struct"]["board"][4], "x")
    _, oracle = self.request_json(
        "GET", f"/api/sessions/{session}?viewer=oracle"
    )
    self.assertIsNone(oracle["observation_struct"])
    self.assertEqual(oracle["struct"]["board"][4], "x")

  def test_serves_the_adapter_presentation(self) -> None:
    _, frame = self.request_json("POST", "/api/sessions?viewer=1", _CREATE)
    session = frame["session"]["id"]
    self.request_json(
        "POST",
        f"/api/sessions/{session}/actions?viewer=0",
        {"action": 4, "ply": None},
    )
    _, seat = self.request_json("GET", f"/api/sessions/{session}?viewer=1")
    presentation = seat["presentation"]
    self.assertEqual(presentation["renderer"], "tic_tac_toe")
    self.assertEqual(presentation["view"]["history"], [4])
    self.assertEqual(presentation["actions"], [])

  def test_refusals_carry_the_contract_error_and_status(self) -> None:
    status, error = self.request_json("GET", "/api/sessions/none?viewer=0")
    self.assertEqual(status, http.HTTPStatus.NOT_FOUND)
    self.assertEqual(error["code"], "not_found")
    self.assertEqual(error["contract_version"], 7)

  def test_refuses_a_host_it_does_not_answer_to(self) -> None:
    status, _, content = self.request(
        "GET", "/api/games", headers={"Host": "attacker.example:8080"}
    )
    self.assertEqual(status, http.HTTPStatus.FORBIDDEN)
    self.assertIn(b"attacker.example", content)

  def test_answers_every_loopback_host_name(self) -> None:
    for host in ("localhost:80", "127.0.0.1", "[::1]:8080", "LOCALHOST"):
      with self.subTest(host=host):
        status, _, _ = self.request(
            "GET", "/api/health", headers={"Host": host}
        )
        self.assertEqual(status, http.HTTPStatus.OK)

  def test_a_change_must_be_json(self) -> None:
    status, _, content = self.request(
        "POST",
        "/api/sessions?viewer=0",
        _CREATE,
        {"Content-Type": "text/plain"},
    )
    self.assertEqual(status, http.HTTPStatus.BAD_REQUEST)
    self.assertEqual(json.loads(content)["code"], "bad_request")
    self.assertEmpty(self.store)

  def test_a_change_from_another_origin_is_refused(self) -> None:
    status, _, _ = self.request(
        "POST",
        "/api/sessions?viewer=0",
        _CREATE,
        _JSON | {"Origin": "http://attacker.example"},
    )
    self.assertEqual(status, http.HTTPStatus.BAD_REQUEST)
    self.assertEmpty(self.store)

  def test_a_change_from_its_own_origin_is_accepted(self) -> None:
    port = self.server.server_address[1]
    status, _, _ = self.request(
        "POST",
        "/api/sessions?viewer=0",
        _CREATE,
        _JSON | {"Origin": f"http://127.0.0.1:{port}"},
    )
    self.assertEqual(status, http.HTTPStatus.OK)

  def test_a_change_from_another_port_of_its_host_is_refused(self) -> None:
    port = self.server.server_address[1]
    for origin in (
        f"http://127.0.0.1:{port + 1}",
        "http://127.0.0.1",
        f"http://localhost:{port}",
        "http://127.0.0.1:port",
    ):
      with self.subTest(origin=origin):
        status, _, _ = self.request(
            "POST",
            "/api/sessions?viewer=0",
            _CREATE,
            _JSON | {"Origin": origin},
        )
        self.assertEqual(status, http.HTTPStatus.BAD_REQUEST)
    self.assertEmpty(self.store)

  def test_refuses_an_oversized_body(self) -> None:
    status, _, _ = self.request(
        "POST",
        "/api/sessions?viewer=0",
        headers=_JSON | {"Content-Length": str(server_lib.MAX_BODY_BYTES + 1)},
    )
    self.assertEqual(status, http.HTTPStatus.BAD_REQUEST)

  def raw_request(self, request: bytes) -> bytes:
    """Sends raw bytes, returning all the server sends until it closes.

    The server closes without reading the body, so the close may arrive as
    a reset once everything it sent has been received.

    Args:
      request: The raw bytes of an HTTP request.

    Returns:
      Everything the server sent before it closed the connection.
    """
    received = b""
    reset = False
    host, port = self.server.server_address[:2]
    with socket.create_connection((str(host), port)) as sock:
      sock.settimeout(10)
      sock.sendall(request)
      try:
        while chunk := sock.recv(65536):
          received += chunk
      except ConnectionResetError:
        reset = True
    if reset:
      self.assertTrue(received, "Reset before any response.")
    return received

  def test_refuses_a_length_that_is_not_ascii_digits(self) -> None:
    body = json.dumps(_CREATE).encode()
    response = self.raw_request(
        b"POST /api/sessions?viewer=0 HTTP/1.1\r\nHost: 127.0.0.1\r\n"
        b"Content-Type: application/json\r\nContent-Length: \xb2\r\n\r\n"
        + body
    )
    self.assertTrue(response.startswith(b"HTTP/1.1 400 "), response)
    self.assertIn(b'"code":"bad_request"', response)
    self.assertEmpty(self.store)

  def test_refuses_a_read_with_a_body_and_closes_the_connection(self) -> None:
    for path in (b"/api/health", b"/"):
      with self.subTest(path=path):
        # Were the body read as the next request, a second response follows.
        response = self.raw_request(
            b"GET "
            + path
            + b" HTTP/1.1\r\nHost: 127.0.0.1\r\n"
            b"Content-Length: 30\r\n\r\n"
            b"GET /api/games HTTP/1.1\r\n\r\n"
        )
        self.assertTrue(response.startswith(b"HTTP/1.1 400 "), response)
        self.assertEqual(response.count(b"HTTP/1.1 "), 1, response)

  def test_a_session_fault_is_an_internal_error(self) -> None:
    _, frame = self.request_json("POST", "/api/sessions?viewer=0", _CREATE)
    session = frame["session"]["id"]
    fault = RuntimeError("secret detail")
    with (
        mock.patch.object(self.store.get(session), "frame", side_effect=fault),
        self.assertLogs(level="ERROR"),
    ):
      status, _, content = self.request(
          "GET", f"/api/sessions/{session}?viewer=0"
      )
    self.assertEqual(status, http.HTTPStatus.INTERNAL_SERVER_ERROR)
    self.assertNotIn(b"secret detail", content)

  def test_serves_the_client_for_other_paths(self) -> None:
    status, headers, content = self.request("GET", "/games/count")
    self.assertEqual(status, http.HTTPStatus.OK)
    self.assertEqual(headers["Content-Type"], "text/html; charset=utf-8")
    self.assertEqual(headers["X-Content-Type-Options"], "nosniff")
    self.assertEqual(content, b"<html></html>")

  def test_head_omits_the_content(self) -> None:
    status, headers, content = self.request("HEAD", "/")
    self.assertEqual(status, http.HTTPStatus.OK)
    self.assertEqual(headers["Content-Length"], str(len(b"<html></html>")))
    self.assertEqual(content, b"")

  def test_the_client_accepts_only_reads(self) -> None:
    status, _, _ = self.request("POST", "/", {}, _JSON)
    self.assertEqual(status, http.HTTPStatus.METHOD_NOT_ALLOWED)

  def test_names_the_build_command_when_the_client_is_not_built(self) -> None:
    missing = pathlib.Path(self.create_tempdir().full_path) / "missing"
    self.server.client = static.ClientFiles(missing)
    status, _, content = self.request("GET", "/")
    self.assertEqual(status, http.HTTPStatus.NOT_FOUND)
    self.assertIn(b"npm run build", content)


class BacklogTest(absltest.TestCase):

  def test_holds_a_burst_of_connections_it_has_not_accepted(self) -> None:
    self.assertGreaterEqual(server_lib.Server.request_queue_size, 128)
    server = server_lib.Server(
        ("127.0.0.1", 0),
        api.Api(store_lib.SessionStore(), catalogue=_catalogue, health=_health),
        static.ClientFiles(pathlib.Path(self.create_tempdir().full_path)),
        server_lib.LOOPBACK_HOSTS,
    )
    self.addCleanup(server.server_close)
    host, port = server.server_address[:2]
    # Not serving, so nothing accepts: every connection waits in the backlog.
    # Past a full backlog the system drops the handshake, and the connection
    # times out (or is reset).
    for _ in range(100):
      connection = socket.create_connection((str(host), port), timeout=2)
      self.addCleanup(connection.close)


class HostNameTest(absltest.TestCase):

  def test_strips_the_port_and_brackets(self) -> None:
    self.assertEqual(server_lib.host_name("localhost:8080"), "localhost")
    self.assertEqual(server_lib.host_name("[::1]:8080"), "::1")
    self.assertEqual(server_lib.host_name("[::1]"), "::1")
    self.assertEqual(server_lib.host_name("::1"), "::1")
    self.assertEqual(server_lib.host_name(" Example.COM "), "example.com")


if __name__ == "__main__":
  absltest.main()
