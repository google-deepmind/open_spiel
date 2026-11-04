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


"""The JSON API: the contract's routes, independent of the HTTP transport.

`Api.handle` takes a request's method, target and body and returns the
response's status and JSON body. CONTRACT.md specifies every route and its
errors.
"""

from collections.abc import Callable
import dataclasses
import http
import re
from typing import Any
from typing import TypeVar
import urllib.parse

from open_spiel.python.gui import catalogue as catalogue_lib
from open_spiel.python.gui import contract
from open_spiel.python.gui import controllers
from open_spiel.python.gui import errors
from open_spiel.python.gui import health as health_lib
from open_spiel.python.gui import request
from open_spiel.python.gui import store
from open_spiel.python.gui import wire

# The path prefix of every API route.
PREFIX = "/api/"

_T = TypeVar("_T")

_NON_NEGATIVE_INTEGER = re.compile(r"0|[1-9][0-9]*")


@dataclasses.dataclass(frozen=True)
class Response:
  """An API response.

  Attributes:
    status: The HTTP status.
    body: A contract object in its wire form.
  """

  status: http.HTTPStatus
  body: contract.JsonValue


def _ok(value: object) -> Response:
  """Answers with a contract object."""
  return Response(http.HTTPStatus.OK, contract.to_json(value))


def _refusal(error: errors.Error) -> Response:
  """Answers with an error body and its status."""
  return Response(error.code.http_status, contract.to_json(error))


@dataclasses.dataclass(frozen=True)
class _Request:
  """A request, with the path parameters its route matched."""

  query: dict[str, list[str]]
  body: bytes
  path: dict[str, str]

  @property
  def session_id(self) -> str:
    """The session the request concerns."""
    return self.path["id"]

  def viewer(self) -> contract.Viewer:
    """Returns the viewer named by the `viewer` query parameter."""
    values = self.query.get("viewer", [])
    if len(values) != 1:
      raise errors.BadRequestError(
          "Name one viewer: viewer=oracle, viewer=spectator or viewer=<seat>."
      )
    value = values[0]
    if value == contract.ViewerKind.ORACLE:
      return contract.Viewer(contract.ViewerKind.ORACLE)
    if value == contract.ViewerKind.SPECTATOR:
      return contract.Viewer(contract.ViewerKind.SPECTATOR)
    if _NON_NEGATIVE_INTEGER.fullmatch(value):
      return contract.Viewer(contract.ViewerKind.SEAT, int(value))
    raise errors.BadRequestError(
        f"Viewer {value!r} is not oracle, spectator or a seat index."
    )

  def index(self, name: str) -> int:
    """Returns the path parameter `name`, a non-negative integer."""
    value = self.path[name]
    if not _NON_NEGATIVE_INTEGER.fullmatch(value):
      raise errors.BadRequestError(
          f"The {name} {value!r} is not a non-negative integer."
      )
    return int(value)

  def decode(self, cls: type[_T]) -> _T:
    """Returns the body decoded as the request dataclass `cls`."""
    return wire.decode(cls, wire.parse(self.body))


def _spec(seat: request.SeatConfig) -> controllers.Spec:
  """Returns the controller a seat asks for."""
  return controllers.Spec(seat.controller, seat.config)


_Handler = Callable[[Any, _Request], Response]
_Route = tuple[str, re.Pattern[str], _Handler]


def _route(method: str, path: str, handler: _Handler) -> _Route:
  """Compiles a route under `PREFIX`; `{name}` matches one path segment."""
  pattern = re.sub(r"{(\w+)}", r"(?P<\1>[^/]+)", PREFIX + path)
  return method, re.compile(pattern), handler


class Api:
  """Answers the contract's routes.

  Safe to call from several threads. Each route makes one call on a session,
  and a session serialises its calls under its own lock, so calls that
  concern one session are serialised and calls that concern different
  sessions run concurrently.
  """

  def __init__(
      self,
      sessions: store.SessionStore,
      *,
      catalogue: Callable[[], catalogue_lib.Catalogue],
      health: Callable[[], health_lib.Health],
  ) -> None:
    """Initializes the API.

    Args:
      sessions: Holds the sessions the API creates and serves.
      catalogue: Returns the catalogue of every game.
      health: Returns the server's health report.
    """
    self._sessions = sessions
    self._catalogue_of = catalogue
    self._health_of = health

  def handle(self, method: str, target: str, body: bytes = b"") -> Response:
    """Answers one request.

    Args:
      method: The HTTP method.
      target: The request target: an API path and its query string.
      body: The request body.

    Returns:
      The response. A request the contract refuses (an `errors.RequestError`)
      is answered with its error body. Any other exception is a fault, not a
      refusal, and propagates.
    """
    url = urllib.parse.urlsplit(target)
    query = urllib.parse.parse_qs(url.query, keep_blank_values=True)
    try:
      handler, path = self._match(method, url.path)
      return handler(self, _Request(query, body, path))
    except errors.RequestError as error:
      return _refusal(error.to_error())

  def _match(self, method: str, path: str) -> tuple[_Handler, dict[str, str]]:
    """Returns the handler for a request and its path parameters."""
    methods = []
    for route_method, pattern, handler in self._ROUTES:
      match = pattern.fullmatch(path)
      if match is None:
        continue
      if route_method == method:
        return handler, match.groupdict()
      methods.append(route_method)
    if methods:
      raise errors.MethodNotAllowedError(
          f"{path} accepts {' and '.join(methods)}, not {method}."
      )
    raise errors.NotFoundError(f"There is no route {path}.")

  def _health(self, unused_request: _Request) -> Response:
    """Answers with the health report."""
    return _ok(self._health_of())

  def _catalogue(self, unused_request: _Request) -> Response:
    """Answers with the catalogue."""
    return _ok(self._catalogue_of())

  def _create(self, req: _Request) -> Response:
    """Starts a session, keeping it only if the viewer may see it."""
    viewer, body = req.viewer(), req.decode(request.CreateSession)
    session = self._sessions.create(
        body.game,
        body.parameters,
        seed=body.seed,
        chance_policy=body.chance_policy,
        controllers=[_spec(seat) for seat in body.seats],
        autoplay=[seat.autoplay for seat in body.seats],
    )
    try:
      frame = session.frame(viewer)
    except errors.RequestError:
      self._sessions.delete(session.id)
      raise
    return _ok(frame)

  def _frame(self, req: _Request) -> Response:
    """Answers with the live position."""
    viewer = req.viewer()
    return _ok(self._sessions.get(req.session_id).frame(viewer))

  def _replay(self, req: _Request) -> Response:
    """Answers with an earlier position."""
    viewer, ply = req.viewer(), req.index("ply")
    session = self._sessions.get(req.session_id)
    return _ok(session.replay_frame(viewer, ply))

  def _action(self, req: _Request) -> Response:
    """Plays a seat's action."""
    viewer, body = req.viewer(), req.decode(request.ApplyAction)
    session = self._sessions.get(req.session_id)
    return _ok(session.apply_action(viewer, body.action, ply=body.ply))

  def _chance(self, req: _Request) -> Response:
    """Chooses a chance outcome."""
    viewer, body = req.viewer(), req.decode(request.ApplyChanceOutcome)
    session = self._sessions.get(req.session_id)
    return _ok(session.apply_chance(viewer, body.outcome, ply=body.ply))

  def _advance(self, req: _Request) -> Response:
    """Lets controllers play, at most `request.MAX_ADVANCE_PLIES` actions."""
    viewer, body = req.viewer(), req.decode(request.Advance)
    session = self._sessions.get(req.session_id)
    return _ok(session.advance(viewer, max_plies=body.limit))

  def _undo(self, req: _Request) -> Response:
    """Undoes the latest choice of a seat."""
    viewer = req.viewer()
    return _ok(self._sessions.get(req.session_id).undo(viewer))

  def _reset(self, req: _Request) -> Response:
    """Restarts the game."""
    viewer, body = req.viewer(), req.decode(request.Reset)
    session = self._sessions.get(req.session_id)
    return _ok(session.reset(viewer, history=body.history or ()))

  def _close(self, req: _Request) -> Response:
    """Forgets a session; any viewer but a spectator closes it."""
    viewer = req.viewer()
    session = self._sessions.get(req.session_id)
    session.check_closer(viewer)
    self._sessions.delete(req.session_id)
    return _ok(contract.SessionClosed(id=req.session_id))

  def _seat(self, req: _Request) -> Response:
    """Changes a seat's controller."""
    viewer, seat = req.viewer(), req.index("seat")
    body = req.decode(request.SeatConfig)
    session = self._sessions.get(req.session_id)
    return _ok(
        session.set_controller(
            viewer, seat, _spec(body), autoplay=body.autoplay
        )
    )

  _ROUTES: tuple[_Route, ...] = (
      _route("GET", "health", _health),
      _route("GET", "games", _catalogue),
      _route("POST", "sessions", _create),
      _route("GET", "sessions/{id}", _frame),
      _route("DELETE", "sessions/{id}", _close),
      _route("GET", "sessions/{id}/plies/{ply}", _replay),
      _route("POST", "sessions/{id}/actions", _action),
      _route("POST", "sessions/{id}/chance", _chance),
      _route("POST", "sessions/{id}/advance", _advance),
      _route("POST", "sessions/{id}/undo", _undo),
      _route("POST", "sessions/{id}/reset", _reset),
      _route("PUT", "sessions/{id}/seats/{seat}", _seat),
  )
