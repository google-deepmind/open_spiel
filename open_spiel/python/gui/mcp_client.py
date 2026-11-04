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


"""The MCP server's client of a GUI server, and its text summary of a frame."""

from collections.abc import Mapping
import json
from typing import Any
import http.client
import urllib.parse

# How many legal actions a summary lists before it says how many more.
_SUMMARY_ACTIONS = 60

# How long to wait for the GUI server; an advance may play many moves.
_TIMEOUT_SECONDS = 600.0


class ToolError(Exception):
  """A tool call the GUI server refused, or that was given bad arguments."""


class GuiClient:
  """Calls a GUI server's HTTP API, with no proxy and no redirect."""

  def __init__(self, server: str) -> None:
    self.server = server.rstrip("/")
    parts = urllib.parse.urlsplit(self.server)
    self._https = parts.scheme == "https"
    self._host = parts.hostname or ""
    self._port = parts.port
    self._prefix = parts.path

  def call(
      self,
      method: str,
      path: str,
      *,
      query: Mapping[str, str] | None = None,
      body: object = None,
  ) -> Any:
    """Returns the JSON body of a successful API call.

    Raises:
      ToolError: If the server answers with an error body, which says why.
      OSError: If the server cannot be reached.
    """
    target = f"{self._prefix}/api/{path}"
    if query:
      target += "?" + urllib.parse.urlencode(query)
    data = None if body is None else json.dumps(body).encode()
    headers = {"Accept": "application/json"}
    if data is not None:
      headers["Content-Type"] = "application/json"
    connection = (
        http.client.HTTPSConnection(
            self._host, self._port, timeout=_TIMEOUT_SECONDS
        )
        if self._https
        else http.client.HTTPConnection(
            self._host, self._port, timeout=_TIMEOUT_SECONDS
        )
    )
    try:
      connection.request(method, target, body=data, headers=headers)
      response = connection.getresponse()
      status, text = response.status, response.read().decode("utf-8", "replace")
    finally:
      connection.close()
    if 200 <= status < 300:
      try:
        return json.loads(text)
      except json.JSONDecodeError as e:
        raise ToolError(
            f"The GUI server's answer (HTTP {status}) is not JSON."
        ) from e
    try:
      error = json.loads(text)
    except json.JSONDecodeError as e:
      raise ToolError(f"HTTP {status}: {text.strip()}") from e
    raise ToolError(
        f"{error.get('code', status)}: {error.get('message', text)}"
    )


def arg_viewer(arguments: Mapping[str, Any]) -> str:
  """Returns the `viewer` argument: oracle (the default), spectator or seat."""
  value = arguments.get("viewer", "oracle")
  if isinstance(value, bool):
    raise ToolError("viewer is oracle, spectator or a seat index.")
  if isinstance(value, int):
    return str(value)
  if not isinstance(value, str):
    raise ToolError("viewer is oracle, spectator or a seat index.")
  return value


def arg_required(arguments: Mapping[str, Any], name: str) -> Any:
  """Returns a required argument."""
  if arguments.get(name) is None:
    raise ToolError(f"The argument {name!r} is required.")
  return arguments[name]


def arg_session(arguments: Mapping[str, Any]) -> str:
  """Returns the required `session_id`, quoted to be one path segment."""
  value = arg_required(arguments, "session_id")
  if not isinstance(value, str):
    raise ToolError("The argument 'session_id' is not a string.")
  return urllib.parse.quote(value, safe="")


def arg_str(arguments: Mapping[str, Any], name: str) -> str | None:
  """Returns an optional string argument."""
  value = arguments.get(name)
  if value is not None and not isinstance(value, str):
    raise ToolError(f"The argument {name!r} is not a string.")
  return value


def arg_object(arguments: Mapping[str, Any], name: str) -> dict[str, Any]:
  """Returns an optional JSON-object argument; {} when it is not given."""
  value = arguments.get(name)
  if value is None:
    return {}
  if not isinstance(value, dict):
    raise ToolError(f"The argument {name!r} is not an object.")
  return dict(value)


def arg_ints(arguments: Mapping[str, Any], name: str) -> list[int] | None:
  """Returns an optional list-of-integers argument."""
  value = arguments.get(name)
  if value is None:
    return None
  if not isinstance(value, list) or any(
      isinstance(item, bool) or not isinstance(item, int) for item in value
  ):
    raise ToolError(f"The argument {name!r} is not a list of integers.")
  return list(value)


def arg_seats(arguments: Mapping[str, Any]) -> list[dict[str, Any]] | None:
  """Returns the optional `seats`: {controller, config?, autoplay?} each."""
  value = arguments.get("seats")
  if value is None:
    return None
  if not isinstance(value, list):
    raise ToolError("The argument 'seats' is not a list.")
  seats = []
  for index, seat in enumerate(value):
    if not isinstance(seat, dict) or not isinstance(
        seat.get("controller"), str
    ):
      raise ToolError(
          f"seats[{index}] is not an object with a string 'controller'."
      )
    autoplay = seat.get("autoplay", False)
    if not isinstance(autoplay, bool):
      raise ToolError(f"seats[{index}].autoplay is not a boolean.")
    seats.append({
        "controller": seat["controller"],
        "config": arg_object(seat, "config"),
        "autoplay": autoplay,
    })
  return seats


def arg_bool(arguments: Mapping[str, Any], name: str) -> bool:
  """Returns an optional boolean argument; False when it is not given."""
  value = arguments.get(name, False)
  if not isinstance(value, bool):
    raise ToolError(f"The argument {name!r} is not a boolean.")
  return value


def arg_int(arguments: Mapping[str, Any], name: str) -> int | None:
  """Returns an optional integer argument."""
  value = arguments.get(name)
  if value is None:
    return None
  if isinstance(value, bool) or not isinstance(value, int):
    raise ToolError(f"The argument {name!r} is not an integer.")
  return value


def summarize(frame: Mapping[str, Any]) -> str:
  """Returns a frame as a short text an agent can read."""
  session = frame["session"]
  viewer = frame["viewer"]
  who = viewer["kind"] if viewer["seat"] is None else f"seat {viewer['seat']}"
  to_move = frame["to_move"]
  mover = (
      f"seat {to_move['seat']}"
      if to_move["kind"] == "seat"
      else to_move["kind"]
  )
  lines = [
      (
          f"session {session['id']}: {session['game']} "
          f"{json.dumps(session['parameters'])}"
      ),
      (
          f"viewer {who}; status {frame['status']}; ply {frame['ply']} of "
          f"{frame['num_plies']}; to move: {mover}"
      ),
  ]
  for seat in frame["seats"]:
    parts = [f"  seat {seat['seat']}: {seat['controller']}"]
    if seat["config"]:
      parts.append(json.dumps(seat["config"]))
    if seat["autoplay"]:
      parts.append("autoplay")
    if seat["status"] != "ready":
      parts.append(f"{seat['status']} ({seat['message']})")
    if seat["failure"] is not None:
      failure = seat["failure"]
      parts.append(
          f"FAILED {failure['error']}"
          + (f": {failure['message']}" if failure["message"] else "")
      )
    if seat.get("notice"):
      parts.append(f"notice: {seat['notice']}")
    lines.append(" ".join(parts))
  if frame["returns"] is not None:
    lines.append(f"returns {frame['returns']}")
  elif frame["rewards"] is not None:
    lines.append(f"rewards {frame['rewards']}")
  choices = frame["chance_outcomes"] or frame["legal_actions"]
  if choices:
    kind = "chance outcomes" if frame["chance_outcomes"] else "legal actions"
    lines.append(f"{kind} ({len(choices)}):")
    for choice in choices[:_SUMMARY_ACTIONS]:
      odds = (
          f" p={choice['probability']:.3f}" if "probability" in choice else ""
      )
      lines.append(f"  {choice['id']}: {choice['label']}{odds}")
    if len(choices) > _SUMMARY_ACTIONS:
      lines.append(f"  ... and {len(choices) - _SUMMARY_ACTIONS} more")
  elif frame["legal_actions"] is None:
    lines.append("legal actions: withheld from this viewer")
  for name in ("state_text", "observation_text", "information_state_text"):
    if frame[name]:
      lines.append(f"{name}:\n{frame[name]}")
  return "\n".join(lines)
