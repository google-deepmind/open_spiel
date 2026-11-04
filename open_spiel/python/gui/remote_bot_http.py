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


"""The remote bot's HTTP: sending a request to a bot server, reading its reply.

`send` posts or gets, with no proxy and no redirect, and tells a refusal (HTTP
400, 409 or 422) from a failure; `parse_reply` checks a 2xx reply against the
protocol (REMOTE_BOTS.md).
"""

from collections.abc import Sequence
import http
import http.client
import json
import socket
import urllib.parse

from open_spiel.python.gui import contract
from open_spiel.python.gui import controller_settings

# The largest reply read from a bot server.
MAX_REPLY_BYTES = 1 << 20

# The protocol version, sent with every request in the `PROTOCOL_HEADER`
# header, so that a request's body is exactly the bot's input.
PROTOCOL = 1
PROTOCOL_HEADER = "Spiel-Protocol"

# A bot server's refusal: the request was understood and declined.
REFUSAL_STATUSES = frozenset({
    http.HTTPStatus.BAD_REQUEST,
    http.HTTPStatus.CONFLICT,
    http.HTTPStatus.UNPROCESSABLE_ENTITY,
})


class RemoteBotRefusedError(RuntimeError):
  """The bot server declined the request: HTTP 400, 409 or 422, or `error`."""


class RemoteBotUnreachableError(RuntimeError):
  """The bot server could not be reached, timed out, or failed (HTTP 5xx)."""


class RemoteBotReplyError(RuntimeError):
  """The bot server's reply is not the protocol's."""


def _connection(
    url: str, timeout: float
) -> tuple[http.client.HTTPConnection, str]:
  """Returns a connection to `url`'s host and the path to request there.

  `http.client` uses no proxy (a loopback bot behind an environment proxy
  would be unreachable) and follows no redirect, which could lead anywhere.
  """
  parts = urllib.parse.urlsplit(url)
  host = parts.hostname or ""
  connection = (
      http.client.HTTPSConnection(host, parts.port, timeout=timeout)
      if parts.scheme == "https"
      else http.client.HTTPConnection(host, parts.port, timeout=timeout)
  )
  return connection, parts.path or "/"


def send(
    url: str, body: bytes | None, timeout: float, token: str | None = None
) -> tuple[int, contract.JsonValue]:
  """Sends a request to a bot server and returns its status and JSON body.

  Args:
    url: The URL to request.
    body: The JSON body to POST, or None to GET.
    timeout: How long to wait for the connection and for each read.
    token: Sent as `Authorization: Bearer <token>`, unless None.

  Returns:
    The HTTP status and the parsed body: a 2xx answer, or a refusal.

  Raises:
    RemoteBotUnreachableError: If the server cannot be reached, does not
      answer within `timeout`, or answers with any status but 2xx and a
      refusal's (a redirect included, which is not followed).
    RemoteBotReplyError: If the body is not JSON, or is too large.
  """
  headers = {"Accept": "application/json", PROTOCOL_HEADER: str(PROTOCOL)}
  if body is not None:
    headers["Content-Type"] = "application/json"
  if token is not None:
    headers["Authorization"] = f"Bearer {token}"
  connection, path = _connection(url, timeout)
  try:
    connection.request(
        "GET" if body is None else "POST", path, body=body, headers=headers
    )
    response = connection.getresponse()
    status, raw = response.status, response.read(MAX_REPLY_BYTES + 1)
  except (TimeoutError, socket.timeout) as e:
    raise RemoteBotUnreachableError(
        f"The bot server at {url} did not answer within {timeout:g} seconds."
    ) from e
  except (OSError, http.client.HTTPException) as e:
    raise RemoteBotUnreachableError(
        f"The bot server at {url} cannot be reached: {e}."
    ) from e
  finally:
    connection.close()
  if not 200 <= status < 300 and status not in REFUSAL_STATUSES:
    raise RemoteBotUnreachableError(
        f"The bot server at {url} answered HTTP {status}"
        + (
            " (a redirect, which is not followed)."
            if 300 <= status < 400
            else "."
        )
    )
  if len(raw) > MAX_REPLY_BYTES:
    raise RemoteBotReplyError(
        f"The bot server's reply is over {MAX_REPLY_BYTES} bytes."
    )
  try:
    return status, json.loads(raw)
  except (json.JSONDecodeError, UnicodeDecodeError) as e:
    raise RemoteBotReplyError(
        f"The bot server's reply (HTTP {status}) is not JSON."
    ) from e


def refusal_text(reply: contract.JsonValue) -> str:
  """Says what a refusal's `error` and `reason` say."""
  if not isinstance(reply, dict):
    return "no reason given"
  error, reason = reply.get("error"), reply.get("reason")
  parts = [str(part) for part in (error, reason) if part is not None]
  return "; ".join(parts) if parts else "no reason given"


def parse_reply(
    reply: contract.JsonValue, legal: Sequence[int]
) -> tuple[int, str | None]:
  """Returns the action and the notice of a bot server's 2xx reply.

  Args:
    reply: The parsed reply.
    legal: The seat's legal actions.

  Returns:
    The action to play, and the notice, or None.

  Raises:
    RemoteBotRefusedError: If the reply sets `error`.
    RemoteBotReplyError: If it gives no action, an action that is not an
      integer or not legal, or a notice that is not text of at most
      `contract.NOTICE_MAX_LENGTH` characters.
  """
  if not isinstance(reply, dict):
    raise RemoteBotReplyError("The bot server's reply is not a JSON object.")
  if reply.get("error") is not None:
    raise RemoteBotRefusedError(
        f"The bot server refused: {refusal_text(reply)}."
    )
  if "action" in reply:
    action = reply["action"]
  elif isinstance(reply.get("actions"), list) and reply["actions"]:
    # A bot that answers with a line plays its first action, and is asked
    # again at the next position.
    action = reply["actions"][0]
  else:
    raise RemoteBotReplyError(
        "The bot server's reply has no `action` (or non-empty `actions`)."
    )
  if isinstance(action, bool) or not isinstance(action, int):
    raise RemoteBotReplyError(
        f"The bot server's action {controller_settings.shown(action)} is "
        "not an integer."
    )
  if action not in legal:
    raise RemoteBotReplyError(
        f"The bot server chose {action}, which is not one of the seat's "
        "legal actions."
    )
  notice = reply.get("notice")
  if notice is not None and (
      not isinstance(notice, str)
      or not notice.strip()
      or len(notice) > contract.NOTICE_MAX_LENGTH
  ):
    raise RemoteBotReplyError(
        f"The bot server's notice {controller_settings.shown(notice)} is "
        f"not non-blank text of at most {contract.NOTICE_MAX_LENGTH} "
        "characters."
    )
  return action, notice
