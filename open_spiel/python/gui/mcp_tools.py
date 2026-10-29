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


"""The MCP server's tools: what an agent can do with a GUI server.

`Tools` calls the GUI server's HTTP API (CONTRACT.md) through
`mcp_client.GuiClient`; `TOOLS` describes each tool as MCP's tools/list lists
it. `mcp` serves them.
"""

from collections.abc import Callable, Mapping
import dataclasses
import json
from typing import Any
import urllib.parse

from open_spiel.python.gui import mcp_client


@dataclasses.dataclass(frozen=True)
class Tool:
  """One MCP tool.

  Attributes:
    name: The tool's name.
    description: What it does, for the agent.
    properties: Its arguments' JSON Schemas, by name.
    required: The arguments it requires.
    call: Runs it on a client with its arguments, returning (text, JSON).
  """

  name: str
  description: str
  properties: Mapping[str, Any]
  required: tuple[str, ...]
  call: Callable[["Tools", Mapping[str, Any]], tuple[str, Any]]

  def describe(self) -> dict[str, Any]:
    """Returns the tool as tools/list lists it."""
    return {
        "name": self.name,
        "description": self.description,
        "inputSchema": {
            "type": "object",
            "properties": dict(self.properties),
            "required": list(self.required),
            "additionalProperties": False,
        },
    }


_SESSION = {"type": "string", "description": "The session's id."}
_VIEWER = {
    "type": ["string", "integer"],
    "description": (
        "Who the frame is for: 'oracle' (analysis; sees everything, the "
        "default), 'spectator', or a seat index (what that seat may see)."
    ),
}
_PLY = {
    "type": "integer",
    "description": (
        "The position to act at: omit for the live position; an earlier ply "
        "takes back the plies after it and plays from there."
    ),
}
_FORMAT = {
    "type": "string",
    "enum": ["summary", "json"],
    "description": "summary (readable text, the default) or json (the frame).",
}


class Tools:
  """The MCP tools, over a GUI server."""

  def __init__(self, client: mcp_client.GuiClient) -> None:
    self.client = client
    self._catalogue: Any = None

  def catalogue(self) -> Any:
    """Returns the catalogue, fetched once."""
    if self._catalogue is None:
      self._catalogue = self.client.call("GET", "games")
    return self._catalogue

  def entry(self, game: str) -> Mapping[str, Any]:
    """Returns the catalogue entry of `game`."""
    for entry in self.catalogue()["games"]:
      if entry["name"] == game:
        found: Mapping[str, Any] = entry
        return found
    raise mcp_client.ToolError(
        f"There is no game {game!r}; list_games lists them."
    )

  def frame_result(
      self, frame: Mapping[str, Any], arguments: Mapping[str, Any]
  ) -> tuple[str, Any]:
    """Returns a frame as a tool's result, in the format asked for."""
    if arguments.get("format", "summary") == "json":
      return json.dumps(frame, indent=1), frame
    return mcp_client.summarize(frame), frame

  def browser_url(self, game: str, session: str, viewer: str) -> str:
    """Returns the URL a browser opens the session at."""
    query = urllib.parse.urlencode({"session": session, "viewer": viewer})
    return (
        f"{self.client.server}/game/{urllib.parse.quote(game, safe='')}?{query}"
    )

  def health(self, arguments: Mapping[str, Any]) -> tuple[str, Any]:
    """Runs the `health` tool, as `TOOLS` describes it."""
    del arguments
    report = self.client.call("GET", "health")
    return json.dumps(report, indent=1), report

  def list_games(self, arguments: Mapping[str, Any]) -> tuple[str, Any]:
    """Runs the `list_games` tool, as `TOOLS` describes it."""
    text = (mcp_client.arg_str(arguments, "filter") or "").lower()
    games = [
        {
            "name": entry["name"],
            "long_name": entry["long_name"],
            "players": entry["num_players"],
            "information": entry["information"],
            "renderer": entry["renderer"],
            "loadable": entry["loadable"],
            "unavailable_reason": entry["unavailable_reason"],
        }
        for entry in self.catalogue()["games"]
        if text in entry["name"].lower() or text in entry["long_name"].lower()
    ]
    # The renderer and the reason a game is unavailable are separate
    # fields, in the text as in the structured result.
    lines = [
        f"{game['name']} ({game['long_name']}): {game['players']} players, "
        f"{game['information']}; renderer: {game['renderer'] or 'text'}"
        + (
            ""
            if game["loadable"]
            else f"; unavailable: {game['unavailable_reason']}"
        )
        for game in games
    ]
    return "\n".join(lines) or "No game matches.", {"games": games}

  def describe_game(self, arguments: Mapping[str, Any]) -> tuple[str, Any]:
    """Runs the `describe_game` tool, as `TOOLS` describes it."""
    entry = self.entry(str(mcp_client.arg_required(arguments, "game")))
    registered = self.client.call("GET", "health")["capabilities"][
        "controllers"
    ]
    controllers = [
        {
            "name": controller["name"],
            "label": controller["label"],
            "available": controller["name"] in entry["controllers"],
            "config_schema": controller["config_schema"],
        }
        for controller in registered
    ]
    result = {"game": entry, "controllers": controllers}
    return json.dumps(result, indent=1), result

  def open_session(self, arguments: Mapping[str, Any]) -> tuple[str, Any]:
    """Runs the `open_session` tool, as `TOOLS` describes it."""
    game = mcp_client.arg_str(arguments, "game")
    if not game:
      raise mcp_client.ToolError("The argument 'game' is required.")
    parameters = mcp_client.arg_object(arguments, "parameters")
    seats = mcp_client.arg_seats(arguments)
    if seats is None:
      count = parameters.get("players", self.entry(game)["num_players"])
      if isinstance(count, bool) or not isinstance(count, int):
        raise mcp_client.ToolError(
            f"{game} cannot be opened with default seats: give seats."
        )
      seats = [
          {"controller": "human", "config": {}, "autoplay": False}
          for _ in range(count)
      ]
    body = {
        "game": game,
        "parameters": parameters,
        "seats": seats,
        "chance_policy": (
            mcp_client.arg_str(arguments, "chance_policy") or "auto"
        ),
        "seed": mcp_client.arg_int(arguments, "seed"),
    }
    viewer = mcp_client.arg_viewer(arguments)
    frame = self.client.call(
        "POST", "sessions", query={"viewer": viewer}, body=body
    )
    url = self.browser_url(game, frame["session"]["id"], viewer)
    text, _ = self.frame_result(frame, arguments)
    return f"Open in a browser: {url}\n{text}", {"url": url, "frame": frame}

  def close_session(self, arguments: Mapping[str, Any]) -> tuple[str, Any]:
    """Runs the `close_session` tool, as `TOOLS` describes it."""
    session = mcp_client.arg_session(arguments)
    closed = self.client.call(
        "DELETE",
        f"sessions/{session}",
        query={"viewer": mcp_client.arg_viewer(arguments)},
    )
    return f"Closed session {closed['id']}.", closed

  def get_frame(self, arguments: Mapping[str, Any]) -> tuple[str, Any]:
    """Runs the `get_frame` tool, as `TOOLS` describes it."""
    session = mcp_client.arg_session(arguments)
    ply = mcp_client.arg_int(arguments, "ply")
    path = f"sessions/{session}" + ("" if ply is None else f"/plies/{ply}")
    frame = self.client.call(
        "GET", path, query={"viewer": mcp_client.arg_viewer(arguments)}
    )
    return self.frame_result(frame, arguments)

  def get_legal_actions(self, arguments: Mapping[str, Any]) -> tuple[str, Any]:
    """Runs the `get_legal_actions` tool, as `TOOLS` describes it."""
    session = mcp_client.arg_session(arguments)
    frame = self.client.call(
        "GET",
        f"sessions/{session}",
        query={"viewer": mcp_client.arg_viewer(arguments)},
    )
    choices = frame["chance_outcomes"] or frame["legal_actions"]
    if choices is None:
      return "The legal actions are withheld from this viewer.", {
          "actions": None
      }
    lines = [f"{choice['id']}: {choice['label']}" for choice in choices]
    kind = "chance_outcomes" if frame["chance_outcomes"] else "actions"
    return "\n".join(lines) or "Nothing to choose here.", {kind: choices}

  def act(self, arguments: Mapping[str, Any]) -> tuple[str, Any]:
    """Runs the `act` tool, as `TOOLS` describes it."""
    session = mcp_client.arg_session(arguments)
    viewer = mcp_client.arg_viewer(arguments)
    action = mcp_client.arg_int(arguments, "action")
    label = mcp_client.arg_str(arguments, "label")
    if (action is None) == (label is None):
      raise mcp_client.ToolError(
          "Give exactly one of action (an id) and label."
      )
    ply = mcp_client.arg_int(arguments, "ply")
    if action is None:
      path = f"sessions/{session}" + ("" if ply is None else f"/plies/{ply}")
      frame = self.client.call("GET", path, query={"viewer": viewer})
      matches = [
          choice["id"]
          for choice in frame["legal_actions"] or ()
          if choice["label"] == label
      ]
      if len(matches) != 1:
        raise mcp_client.ToolError(
            f"{len(matches)} legal actions are labelled {label!r}; give the "
            "action id instead (get_legal_actions lists them)."
        )
      action = matches[0]
    frame = self.client.call(
        "POST",
        f"sessions/{session}/actions",
        query={"viewer": viewer},
        body={"action": action, "ply": ply},
    )
    return self.frame_result(frame, arguments)

  def choose_chance(self, arguments: Mapping[str, Any]) -> tuple[str, Any]:
    """Runs the `choose_chance` tool, as `TOOLS` describes it."""
    session = mcp_client.arg_session(arguments)
    frame = self.client.call(
        "POST",
        f"sessions/{session}/chance",
        query={"viewer": mcp_client.arg_viewer(arguments)},
        body={
            "outcome": mcp_client.arg_int(arguments, "outcome"),
            "ply": mcp_client.arg_int(arguments, "ply"),
        },
    )
    return self.frame_result(frame, arguments)

  def advance(self, arguments: Mapping[str, Any]) -> tuple[str, Any]:
    """Runs the `advance` tool, as `TOOLS` describes it."""
    session = mcp_client.arg_session(arguments)
    frame = self.client.call(
        "POST",
        f"sessions/{session}/advance",
        query={"viewer": mcp_client.arg_viewer(arguments)},
        body={"max_plies": mcp_client.arg_int(arguments, "max_plies")},
    )
    return self.frame_result(frame, arguments)

  def undo(self, arguments: Mapping[str, Any]) -> tuple[str, Any]:
    """Runs the `undo` tool, as `TOOLS` describes it."""
    session = mcp_client.arg_session(arguments)
    # The route reads no fields, but the server takes every POST as JSON.
    frame = self.client.call(
        "POST",
        f"sessions/{session}/undo",
        query={"viewer": mcp_client.arg_viewer(arguments)},
        body={},
    )
    return self.frame_result(frame, arguments)

  def reset(self, arguments: Mapping[str, Any]) -> tuple[str, Any]:
    """Runs the `reset` tool, as `TOOLS` describes it."""
    session = mcp_client.arg_session(arguments)
    history = mcp_client.arg_ints(arguments, "history")
    frame = self.client.call(
        "POST",
        f"sessions/{session}/reset",
        query={"viewer": mcp_client.arg_viewer(arguments)},
        body={"history": history},
    )
    return self.frame_result(frame, arguments)

  def set_seat(self, arguments: Mapping[str, Any]) -> tuple[str, Any]:
    """Runs the `set_seat` tool, as `TOOLS` describes it."""
    session = mcp_client.arg_session(arguments)
    seat = mcp_client.arg_int(arguments, "seat")
    if seat is None:
      raise mcp_client.ToolError("The argument 'seat' is required.")
    frame = self.client.call(
        "PUT",
        f"sessions/{session}/seats/{seat}",
        query={"viewer": mcp_client.arg_viewer(arguments)},
        body={
            "controller": mcp_client.arg_str(arguments, "controller"),
            "config": mcp_client.arg_object(arguments, "config"),
            "autoplay": mcp_client.arg_bool(arguments, "autoplay"),
        },
    )
    return self.frame_result(frame, arguments)


TOOLS: tuple[Tool, ...] = (
    Tool(
        "health",
        "The GUI server's version, controllers and capabilities.",
        {},
        (),
        Tools.health,
    ),
    Tool(
        "list_games",
        "Lists the games: name, players, information, and whether the GUI "
        "draws a board for it or shows text.",
        {
            "filter": {
                "type": "string",
                "description": "Only games whose name contains this.",
            }
        },
        (),
        Tools.list_games,
    ),
    Tool(
        "describe_game",
        "A game's parameters, and each controller (human, random, mcts, "
        "remote, plugins) with whether it can play the game and its settings' "
        "JSON Schema.",
        {"game": {"type": "string", "description": "The game's short name."}},
        ("game",),
        Tools.describe_game,
    ),
    Tool(
        "open_session",
        "Opens a session, which also shows in a browser at the URL returned. "
        "Seats default to human. A seat is {controller, config?, autoplay?}; "
        "for a remote bot, controller 'remote' with config {url, state: "
        "seat_view | full_state, ...}.",
        {
            "game": {"type": "string", "description": "The game's short name."},
            "parameters": {
                "type": "object",
                "description": "Game parameters; omitted ones take defaults.",
            },
            "seats": {
                "type": "array",
                "items": {"type": "object"},
                "description": "One per seat, in seat order.",
            },
            "seed": {"type": "integer", "description": "The chance seed."},
            "chance_policy": {
                "type": "string",
                "enum": ["auto", "expose"],
                "description": (
                    "auto samples chance; expose stops at each chance node "
                    "for choose_chance."
                ),
            },
            "viewer": _VIEWER,
            "format": _FORMAT,
        },
        ("game",),
        Tools.open_session,
    ),
    Tool(
        "close_session",
        "Closes a session: the GUI server forgets it.",
        {"session_id": _SESSION, "viewer": _VIEWER},
        ("session_id",),
        Tools.close_session,
    ),
    Tool(
        "get_frame",
        "Reads a session's position for a viewer: seats, who is to move, "
        "legal actions, rewards and the game's texts. With ply, an earlier "
        "position (a replay).",
        {
            "session_id": _SESSION,
            "viewer": _VIEWER,
            "ply": {"type": "integer", "description": "An earlier position."},
            "format": _FORMAT,
        },
        ("session_id",),
        Tools.get_frame,
    ),
    Tool(
        "get_legal_actions",
        "Lists the legal actions (or chance outcomes) at the live position, "
        "as id: label, as the GUI's Legal list shows them.",
        {"session_id": _SESSION, "viewer": _VIEWER},
        ("session_id",),
        Tools.get_legal_actions,
    ),
    Tool(
        "act",
        "Plays a human seat's action, by id or by its exact label.",
        {
            "session_id": _SESSION,
            "viewer": _VIEWER,
            "action": {"type": "integer", "description": "The action id."},
            "label": {
                "type": "string",
                "description": "The action's exact label, if unique.",
            },
            "ply": _PLY,
            "format": _FORMAT,
        },
        ("session_id",),
        Tools.act,
    ),
    Tool(
        "choose_chance",
        "Chooses a chance outcome, in a session opened with chance_policy "
        "expose.",
        {
            "session_id": _SESSION,
            "viewer": _VIEWER,
            "outcome": {"type": "integer", "description": "The outcome id."},
            "ply": _PLY,
            "format": _FORMAT,
        },
        ("session_id", "outcome"),
        Tools.choose_chance,
    ),
    Tool(
        "advance",
        "Lets the bots play their seats until a human must act, the game "
        "ends, a bot fails, or max_plies moves.",
        {
            "session_id": _SESSION,
            "viewer": _VIEWER,
            "max_plies": {"type": "integer", "description": "At most 1000."},
            "format": _FORMAT,
        },
        ("session_id",),
        Tools.advance,
    ),
    Tool(
        "undo",
        "Takes back the viewer's latest action (the oracle's: any seat's).",
        {"session_id": _SESSION, "viewer": _VIEWER, "format": _FORMAT},
        ("session_id",),
        Tools.undo,
    ),
    Tool(
        "reset",
        "Restarts the game, replaying history (action ids, chance included).",
        {
            "session_id": _SESSION,
            "viewer": _VIEWER,
            "history": {"type": "array", "items": {"type": "integer"}},
            "format": _FORMAT,
        },
        ("session_id",),
        Tools.reset,
    ),
    Tool(
        "set_seat",
        "Gives a seat a controller (human, random, mcts, remote, or a plugin "
        "bot) with its config; a failed seat resumes.",
        {
            "session_id": _SESSION,
            "seat": {"type": "integer"},
            "controller": {"type": "string"},
            "config": {"type": "object"},
            "autoplay": {"type": "boolean"},
            "viewer": _VIEWER,
            "format": _FORMAT,
        },
        ("session_id", "seat", "controller"),
        Tools.set_seat,
    ),
)

BY_NAME = {tool.name: tool for tool in TOOLS}
