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


"""The HTTP server: the JSON API under `/api/` and the browser client.

The server is meant for one machine. It has no authentication, and an oracle
frame shows every player's private information, so by default it listens on
the loopback interface only. It also answers only requests whose `Host`
names an allowed host, which stops a web page that rebinds its own domain to
the loopback address from reading the API, and it accepts state-changing
requests only as JSON from its own origin (the scheme, host and port the
request's `Host` names), which stops a page on another origin, another port
of the same host included, from posting to it.
"""

import email.message
import http
import http.server
import json
import re
import socket
import urllib.parse

from absl import logging

from open_spiel.python.gui import api as api_lib
from open_spiel.python.gui import contract
from open_spiel.python.gui import errors
from open_spiel.python.gui import static

# The host names that reach the loopback interface.
LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})

# The largest request body the server reads, in bytes.
MAX_BODY_BYTES = 1 << 20

_BODY_METHODS = frozenset({"POST", "PUT"})

_DIGITS = re.compile(r"[0-9]+")

_SECURITY_HEADERS = {
    "Cache-Control": "no-cache",
    "Content-Security-Policy": "frame-ancestors 'none'",
    "Referrer-Policy": "no-referrer",
    "X-Content-Type-Options": "nosniff",
}


class Server(http.server.ThreadingHTTPServer):
  """An HTTP server for the API and the client, one thread per request."""

  daemon_threads = True
  # The listen backlog: connections the system holds while the server has
  # not yet accepted them. The standard library's 5 resets a burst of
  # requests, such as a page loading while a move is played.
  request_queue_size = 128

  def __init__(
      self,
      address: tuple[str, int],
      api: api_lib.Api,
      client: static.ClientFiles,
      allowed_hosts: frozenset[str],
  ) -> None:
    """Binds the server; `serve_forever` starts it.

    Args:
      address: The host and port to listen on; port 0 picks a free port.
      api: The API that answers requests under `api_lib.PREFIX`.
      client: The client files that answer every other request.
      allowed_hosts: The host names, without ports, that a request's `Host`
        and `Origin` may name.

    Raises:
      OSError: If the address cannot be bound.
    """
    if ":" in address[0]:
      self.address_family = socket.AF_INET6
    self.api = api
    self.client = client
    self.allowed_hosts = allowed_hosts
    super().__init__(address, _Handler)

  @property
  def url(self) -> str:
    """The URL the server is reachable at."""
    host, port = self.server_address[:2]
    assert isinstance(host, str)
    return (
        f"http://[{host}]:{port}/" if ":" in host else f"http://{host}:{port}/"
    )


def host_name(authority: str) -> str:
  """Returns the host name of a URL authority, without port or brackets.

  Args:
    authority: A `Host` header value, such as `localhost:8080` or `[::1]:80`.

  Returns:
    The host name in lower case, such as `localhost` or `::1`.
  """
  authority = authority.strip().lower()
  if authority.startswith("["):
    return authority[1:].partition("]")[0]
  if authority.count(":") == 1:
    return authority.partition(":")[0]
  return authority


def _has_body(headers: email.message.Message) -> bool:
  """Whether a request's headers announce a body."""
  length = headers.get("Content-Length", "0").strip()
  return "Transfer-Encoding" in headers or length != "0"


class _Handler(http.server.BaseHTTPRequestHandler):
  """Answers one connection's requests."""

  server: Server
  protocol_version = "HTTP/1.1"
  server_version = "OpenSpielGUI"

  def do_GET(self) -> None:  # pylint: disable=invalid-name
    """Answers a GET request."""
    self._handle()

  def do_HEAD(self) -> None:  # pylint: disable=invalid-name
    """Answers a HEAD request."""
    self._handle()

  def do_POST(self) -> None:  # pylint: disable=invalid-name
    """Answers a POST request."""
    self._handle()

  def do_PUT(self) -> None:  # pylint: disable=invalid-name
    """Answers a PUT request."""
    self._handle()

  def do_DELETE(self) -> None:  # pylint: disable=invalid-name
    """Answers a DELETE request."""
    self._handle()

  def log_message(self, format: str, *args: object) -> None:
    """Logs a request through absl rather than to stderr."""
    logging.info("%s %s", self.address_string(), format % args)

  def _handle(self) -> None:
    """Answers the request, refusing it if it fails a transport check."""
    host = host_name(self.headers.get("Host", ""))
    if host not in self.server.allowed_hosts:
      self.close_connection = True
      self._send_text(http.HTTPStatus.FORBIDDEN, f"Host {host!r} is refused.")
      return
    path = urllib.parse.urlsplit(self.path).path
    is_api = path.startswith(api_lib.PREFIX)
    if self.command not in _BODY_METHODS and _has_body(self.headers):
      # The body is never read, so the connection cannot carry another
      # request: its bytes would be parsed as one.
      message = f"A {self.command} request has no body."
      if is_api:
        self._refuse(message)
      else:
        self.close_connection = True
        self._send_text(http.HTTPStatus.BAD_REQUEST, message)
      return
    if is_api:
      self._handle_api()
    else:
      self._handle_client(path)

  def _handle_api(self) -> None:
    """Answers an API request."""
    body = b""
    if self.command == "DELETE":
      refusal = self._check_origin()
      if refusal is not None:
        self._refuse(refusal)
        return
    if self.command in _BODY_METHODS:
      refusal = self._check_origin() or self._check_body()
      if refusal is not None:
        self._refuse(refusal)
        return
      body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
    try:
      response = self.server.api.handle(self.command, self.path, body)
    except Exception:  # pylint: disable=broad-exception-caught
      logging.exception("%s %s failed.", self.command, self.path)
      self._send_text(
          http.HTTPStatus.INTERNAL_SERVER_ERROR,
          "The server failed; its log has the details.",
      )
      return
    self._send_json(response.status, response.body)

  def _check_origin(self) -> str | None:
    """Returns why the request's origin is refused, or None."""
    origin = self.headers.get("Origin")
    if origin is None:
      return None
    refused = f"Requests from {origin} are refused."
    url = urllib.parse.urlsplit(origin)
    if url.scheme != "http" or url.hostname not in self.server.allowed_hosts:
      return refused
    own = urllib.parse.urlsplit("//" + self.headers.get("Host", "").strip())
    try:
      same = (url.hostname, url.port or 80) == (own.hostname, own.port or 80)
    except ValueError:  # A port that is not a number.
      return refused
    return None if same else refused

  def _check_body(self) -> str | None:
    """Returns why the request's body is refused, or None."""
    if self.headers.get_content_type() != "application/json":
      return "The body must be application/json."
    if "Transfer-Encoding" in self.headers:
      return "The body must have a Content-Length."
    length = self.headers.get("Content-Length", "0")
    if not _DIGITS.fullmatch(length) or int(length) > MAX_BODY_BYTES:
      return f"The body must be at most {MAX_BODY_BYTES} bytes."
    return None

  def _handle_client(self, path: str) -> None:
    """Answers a request for the client."""
    if self.command not in ("GET", "HEAD"):
      self.close_connection = True
      self._send_text(
          http.HTTPStatus.METHOD_NOT_ALLOWED, f"{path} accepts GET and HEAD."
      )
      return
    asset = self.server.client.find(path)
    if asset is not None:
      self._send(http.HTTPStatus.OK, asset.content, asset.content_type)
    elif self.server.client.built:
      self._send_text(http.HTTPStatus.NOT_FOUND, f"There is no file {path}.")
    else:
      self._send_text(
          http.HTTPStatus.NOT_FOUND,
          f"The client is not built in {self.server.client.root}. Run "
          "`npm ci && npm run build` in open_spiel/gui; the API is served "
          f"under {api_lib.PREFIX} either way.",
      )

  def _refuse(self, message: str) -> None:
    """Refuses a malformed API request with the contract's error body."""
    error = errors.Error(code=errors.ErrorCode.BAD_REQUEST, message=message)
    self.close_connection = True
    self._send_json(error.code.http_status, contract.to_json(error))

  def _send_json(self, status: http.HTTPStatus, body: object) -> None:
    """Sends a JSON response."""
    content = json.dumps(body, allow_nan=False, separators=(",", ":"))
    self._send(status, content.encode(), "application/json")

  def _send_text(self, status: http.HTTPStatus, text: str) -> None:
    """Sends a plain-text response, for requests outside the contract."""
    self._send(status, text.encode(), "text/plain; charset=utf-8")

  def _send(
      self, status: http.HTTPStatus, content: bytes, content_type: str
  ) -> None:
    """Sends a complete response; a `HEAD` response omits the content."""
    self.send_response(status)
    self.send_header("Content-Type", content_type)
    self.send_header("Content-Length", str(len(content)))
    for name, value in _SECURITY_HEADERS.items():
      self.send_header(name, value)
    self.end_headers()
    if self.command != "HEAD":
      self.wfile.write(content)
