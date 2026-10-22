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

"""Refused requests: the error body, and the exceptions that carry it.

Every refusal raises a `RequestError`, one subclass per `ErrorCode`. Whatever
refuses a request (the session, the game loader, the controllers, the request
decoder) raises one of these or a subclass of one, so the server answers every
refusal the same way: `to_error()` gives the body, and its code the status.
"""

import dataclasses
import enum
import http
from typing import Any, ClassVar

from open_spiel.python.gui import contract


class ErrorCode(enum.StrEnum):
  """Why a request was refused.

  Attributes:
    BAD_REQUEST: The request is malformed.
    NOT_FOUND: The session, game, seat or ply does not exist.
    METHOD_NOT_ALLOWED: The path exists but does not accept the method.
    GAME_UNAVAILABLE: The game cannot be served; its catalogue entry says why.
    VIEWER_UNAVAILABLE: The game cannot build a frame for the viewer.
    NOT_YOUR_TURN: The viewer may not choose the next action.
    ILLEGAL_ACTION: The action is not legal at the position.
    CONTROLLER_UNAVAILABLE: The controller cannot play the game, or its
      settings do not satisfy its `config_schema`.
    CONTROLLER_FAILED: A seat's controller failed and the session is paused.
    NOTHING_TO_UNDO: The live game has no action to undo.
  """

  BAD_REQUEST = "bad_request"
  NOT_FOUND = "not_found"
  METHOD_NOT_ALLOWED = "method_not_allowed"
  GAME_UNAVAILABLE = "game_unavailable"
  VIEWER_UNAVAILABLE = "viewer_unavailable"
  NOT_YOUR_TURN = "not_your_turn"
  ILLEGAL_ACTION = "illegal_action"
  CONTROLLER_UNAVAILABLE = "controller_unavailable"
  CONTROLLER_FAILED = "controller_failed"
  NOTHING_TO_UNDO = "nothing_to_undo"

  @property
  def http_status(self) -> http.HTTPStatus:
    """The HTTP status a response with this code carries."""
    if self == ErrorCode.BAD_REQUEST:
      return http.HTTPStatus.BAD_REQUEST
    if self == ErrorCode.NOT_FOUND:
      return http.HTTPStatus.NOT_FOUND
    if self == ErrorCode.METHOD_NOT_ALLOWED:
      return http.HTTPStatus.METHOD_NOT_ALLOWED
    return http.HTTPStatus.CONFLICT


@dataclasses.dataclass(frozen=True)
class Error:
  """A refused request.

  Attributes:
    contract_version: The contract version the error was built against.
    code: Why the request was refused.
    message: A human-readable explanation, safe to show the requester: it
      never holds state the requester may not see.
    seat: The seat the refusal concerns, or null.
    ply: The index of the history entry the refusal concerns, or null.
  """

  contract_version: contract.ContractVersion = dataclasses.field(
      default=contract.CONTRACT_VERSION, kw_only=True
  )
  code: ErrorCode
  message: str
  seat: int | None = None
  ply: int | None = None

  def __post_init__(self) -> None:
    contract.require(
        self.seat is None or self.seat >= 0,
        f"An error cannot concern seat {self.seat}.",
    )
    contract.require(
        self.ply is None or self.ply >= 0,
        f"An error cannot concern ply {self.ply}.",
    )


class RequestError(Exception):
  """A refused request; subclasses fix the contract's code.

  `RequestError` itself has no code and cannot be raised; every subclass
  declares one, which is checked when the subclass is defined.

  Attributes:
    code: Why the request was refused.
    seat: The seat the refusal concerns, or None.
    ply: The index of the history entry the refusal concerns, or None.
  """

  code: ClassVar[ErrorCode]

  def __init_subclass__(cls, **kwargs: Any) -> None:
    super().__init_subclass__(**kwargs)
    if not isinstance(getattr(cls, "code", None), ErrorCode):
      raise TypeError(f"{cls.__name__} declares no ErrorCode `code`.")

  def __init__(
      self, message: str, *, seat: int | None = None, ply: int | None = None
  ) -> None:
    if not isinstance(getattr(self, "code", None), ErrorCode):
      raise TypeError("Raise a subclass of RequestError, which has a code.")
    super().__init__(message)
    self.seat = seat
    self.ply = ply

  def to_error(self) -> Error:
    """Returns the refusal as the contract's error body."""
    return Error(
        code=self.code, message=str(self), seat=self.seat, ply=self.ply
    )


class BadRequestError(RequestError):
  """The request is malformed."""

  code = ErrorCode.BAD_REQUEST


class NotFoundError(RequestError):
  """The session, game, seat or ply does not exist."""

  code = ErrorCode.NOT_FOUND


class MethodNotAllowedError(RequestError):
  """The path exists but does not accept the method."""

  code = ErrorCode.METHOD_NOT_ALLOWED


class GameUnavailableError(RequestError):
  """The game cannot be served."""

  code = ErrorCode.GAME_UNAVAILABLE


class ViewerUnavailableError(RequestError):
  """The game cannot build a frame for the viewer."""

  code = ErrorCode.VIEWER_UNAVAILABLE


class NotYourTurnError(RequestError):
  """The viewer may not choose the next action."""

  code = ErrorCode.NOT_YOUR_TURN


class IllegalActionError(RequestError):
  """The action is not legal at the position."""

  code = ErrorCode.ILLEGAL_ACTION


class ControllerUnavailableError(RequestError):
  """The controller cannot play the game with the given settings."""

  code = ErrorCode.CONTROLLER_UNAVAILABLE


class ControllerFailedError(RequestError):
  """A seat's controller failed, so the session is paused.

  The message names only the seat: what the controller raised may describe
  private state, so it reaches only an omniscient viewer, as the seat's
  `failure.message` in a frame.
  """

  code = ErrorCode.CONTROLLER_FAILED

  def __init__(self, *, seat: int) -> None:
    super().__init__(
        f"Seat {seat}'s controller failed; undo, reset or change it.",
        seat=seat,
    )


class NothingToUndoError(RequestError):
  """The live game has no action to undo."""

  code = ErrorCode.NOTHING_TO_UNDO
