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

"""Tests for the MCP server, end to end against a running GUI server.

The GUI server, a reference bot server and the MCP server each run as a
process of their own, as an agent runs them; the test speaks MCP to the MCP
server over its stdin and stdout.
"""

import http.client
import io
import json
import os
import subprocess
import sys
import tempfile
from typing import Any
from unittest import mock

from absl.testing import absltest

from open_spiel.python.gui import mcp
from open_spiel.python.gui import mcp_client
from open_spiel.python.gui import mcp_tools
from open_spiel.python.gui import remote_bot_testing


# The protocol version the test's client asks for, which the server echoes.
_VERSION = "the-client's-version"


class McpUnitTest(absltest.TestCase):

  def test_initialize_requires_a_protocol_version(self) -> None:
    server = mcp.McpServer(
        mcp_tools.Tools(mcp_client.GuiClient("http://127.0.0.1:1"))
    )
    response = server.handle(
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}
    )
    assert response is not None
    self.assertEqual(response["error"]["code"], mcp.INVALID_PARAMS)

  def test_a_server_elsewhere_is_refused_unless_allowed(self) -> None:
    self.assertIsNone(mcp.server_refusal("http://127.0.0.1:8080", False))
    self.assertIsNone(mcp.server_refusal("http://localhost:8080", False))
    self.assertIn(
        "--allow-remote-server",
        mcp.server_refusal("http://example.com:8080", False) or "",
    )
    self.assertIsNone(mcp.server_refusal("http://example.com:8080", True))
    self.assertIsNotNone(mcp.server_refusal("file:///etc", True))

  def test_bad_messages_get_json_rpc_errors(self) -> None:
    server = mcp.McpServer(
        mcp_tools.Tools(mcp_client.GuiClient("http://127.0.0.1:1"))
    )
    out = io.StringIO()
    mcp.serve(
        server,
        io.StringIO(
            "not json\n"
            + json.dumps({"jsonrpc": "2.0", "id": 1, "method": "nope"})
            + "\n"
            + json.dumps({"jsonrpc": "2.0", "method": "notifications/x"})
            + "\n"
        ),
        out,
    )
    codes = [
        json.loads(line)["error"]["code"]
        for line in out.getvalue().splitlines()
    ]
    self.assertEqual(codes, [mcp.PARSE_ERROR, mcp.METHOD_NOT_FOUND])

  def test_an_unreachable_gui_server_is_a_tool_error(self) -> None:
    server = mcp.McpServer(
        mcp_tools.Tools(mcp_client.GuiClient("http://127.0.0.1:1"))
    )
    response = server.handle({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {"name": "health", "arguments": {}},
    })
    assert response is not None
    self.assertTrue(response["result"]["isError"])
    self.assertIn("cannot be reached", response["result"]["content"][0]["text"])

  def test_a_tool_that_fails_is_an_internal_error_not_an_exit(self) -> None:
    tools = mcp_tools.Tools(mcp_client.GuiClient("http://127.0.0.1:1"))
    server = mcp.McpServer(tools)
    with mock.patch.object(
        tools.client, "call", side_effect=RuntimeError("broken")
    ):
      with self.assertLogs(level="ERROR"):
        response = server.handle({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "health", "arguments": {}},
        })
    assert response is not None
    self.assertEqual(response["error"]["code"], mcp.INTERNAL_ERROR)
    self.assertIn("broken", response["error"]["message"])

  def test_an_answer_that_is_not_json_is_a_tool_error(self) -> None:
    client = mcp_client.GuiClient("http://127.0.0.1:1")
    response = mock.MagicMock(status=200)
    response.read.return_value = b"<html>"
    with mock.patch.object(
        http.client.HTTPConnection, "request"
    ), mock.patch.object(
        http.client.HTTPConnection, "getresponse", return_value=response
    ):
      with self.assertRaisesRegex(mcp_client.ToolError, "not JSON"):
        client.call("GET", "health")

  def test_list_games_keeps_the_renderer_and_the_reason_apart(self) -> None:
    tools = mcp_tools.Tools(mcp_client.GuiClient("http://127.0.0.1:1"))
    entry = {"long_name": "x", "num_players": 2, "information": "perfect"}
    tools._catalogue = {  # pylint: disable=protected-access
        "games": [
            {
                **entry,
                "name": "drawn",
                "renderer": "grid",
                "loadable": True,
                "unavailable_reason": None,
            },
            {
                **entry,
                "name": "texted",
                "renderer": None,
                "loadable": True,
                "unavailable_reason": None,
            },
            {
                **entry,
                "name": "broken",
                "renderer": None,
                "loadable": False,
                "unavailable_reason": "its positions cannot be served",
            },
        ]
    }
    text, result = tools.list_games({})
    self.assertEqual(
        text.splitlines(),
        [
            "drawn (x): 2 players, perfect; renderer: grid",
            "texted (x): 2 players, perfect; renderer: text",
            (
                "broken (x): 2 players, perfect; renderer: text; unavailable: "
                "its positions cannot be served"
            ),
        ],
    )
    self.assertEqual(
        [
            (g["renderer"], g["loadable"], g["unavailable_reason"])
            for g in result["games"]
        ],
        [
            ("grid", True, None),
            (None, True, None),
            (None, False, "its positions cannot be served"),
        ],
    )

  def test_unknown_arguments_are_refused(self) -> None:
    server = mcp.McpServer(
        mcp_tools.Tools(mcp_client.GuiClient("http://127.0.0.1:1"))
    )
    response = server.handle({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {"name": "health", "arguments": {"x": 1}},
    })
    assert response is not None
    self.assertEqual(response["error"]["code"], mcp.INVALID_PARAMS)


class McpEndToEndTest(absltest.TestCase):
  """The GUI server, a bot server and the MCP server, each a process."""

  dir: str
  processes: list[subprocess.Popen[str]]
  gui: str
  bot: str
  mcp: subprocess.Popen[str]
  next_id: int

  @classmethod
  def setUpClass(cls) -> None:
    super().setUpClass()
    cls.dir = tempfile.mkdtemp()
    cls.processes = []
    gui, cls.gui = remote_bot_testing.start(
        ["-m", "open_spiel.python.gui", "--port=0", f"--client-dir={cls.dir}"],
        r"Serving OpenSpiel games at (\S+)",
        dict(os.environ),
    )
    cls.processes.append(gui)
    bot, cls.bot = remote_bot_testing.start(
        [
            "-m",
            "open_spiel.python.gui.remote_bot_server",
            "--port=0",
            "--bot=mcts",
        ],
        r"at (http://\S+)",
        dict(os.environ),
    )
    cls.processes.append(bot)
    cls.mcp = subprocess.Popen(  # pylint: disable=consider-using-with
        [
            sys.executable,
            "-m",
            "open_spiel.python.gui.mcp",
            f"--server={cls.gui}",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    cls.processes.append(cls.mcp)
    cls.next_id = 0

  @classmethod
  def tearDownClass(cls) -> None:
    for process in cls.processes:
      process.terminate()
      process.wait(timeout=30)
    super().tearDownClass()

  def rpc(self, method: str, params: dict[str, Any] | None = None) -> Any:
    """Sends one request and returns its response."""
    type(self).next_id += 1
    message = {"jsonrpc": "2.0", "id": self.next_id, "method": method}
    if params is not None:
      message["params"] = params
    assert self.mcp.stdin is not None and self.mcp.stdout is not None
    self.mcp.stdin.write(json.dumps(message) + "\n")
    self.mcp.stdin.flush()
    response = json.loads(self.mcp.stdout.readline())
    self.assertEqual(response.get("id"), self.next_id, response)
    return response

  def tool(self, name: str, **arguments: Any) -> Any:
    """Calls a tool, failing loud on a tool error; returns its result."""
    result = self.rpc("tools/call", {"name": name, "arguments": arguments})[
        "result"
    ]
    self.assertFalse(result["isError"], result["content"][0]["text"])
    return result

  def test_initialize_and_list_the_tools(self) -> None:
    result = self.rpc(
        "initialize",
        {
            "protocolVersion": _VERSION,
            "capabilities": {},
            "clientInfo": {"name": "test", "version": "0"},
        },
    )["result"]
    self.assertEqual(result["protocolVersion"], _VERSION)
    self.assertIn("tools", result["capabilities"])
    self.assertEqual(self.rpc("ping")["result"], {})
    names = {tool["name"] for tool in self.rpc("tools/list")["result"]["tools"]}
    self.assertEqual(
        names,
        {
            "health",
            "list_games",
            "describe_game",
            "open_session",
            "close_session",
            "get_frame",
            "get_legal_actions",
            "act",
            "choose_chance",
            "advance",
            "undo",
            "reset",
            "set_seat",
        },
    )

  def test_list_and_describe_games(self) -> None:
    games = self.tool("list_games", filter="tic_tac")["structuredContent"][
        "games"
    ]
    self.assertIn("tic_tac_toe", [game["name"] for game in games])
    described = self.tool("describe_game", game="kuhn_poker")[
        "structuredContent"
    ]
    available = {c["name"]: c["available"] for c in described["controllers"]}
    self.assertTrue(available["remote"])
    self.assertFalse(available["mcts"])

  def test_play_a_game_with_every_control(self) -> None:
    opened = self.tool(
        "open_session",
        game="tic_tac_toe",
        seats=[{"controller": "human"}, {"controller": "human"}],
    )
    session = opened["structuredContent"]["frame"]["session"]["id"]
    self.assertIn(
        f"/game/tic_tac_toe?session={session}", opened["content"][0]["text"]
    )
    legal = self.tool("get_legal_actions", session_id=session)
    self.assertIn("4: x(1,1)", legal["content"][0]["text"])
    frame = self.tool("act", session_id=session, label="x(1,1)", viewer=0)
    self.assertEqual(frame["structuredContent"]["num_plies"], 1)
    frame = self.tool("act", session_id=session, action=0, viewer=1)
    self.assertEqual(frame["structuredContent"]["num_plies"], 2)
    replay = self.tool("get_frame", session_id=session, ply=1, format="json")
    self.assertEqual(json.loads(replay["content"][0]["text"])["ply"], 1)
    frame = self.tool("undo", session_id=session)
    self.assertEqual(frame["structuredContent"]["num_plies"], 1)
    # x at 0, 1 and 8; o at 4 and 3, to move, wins at 5.
    frame = self.tool("reset", session_id=session, history=[0, 4, 1, 3, 8])
    self.assertEqual(frame["structuredContent"]["num_plies"], 5)
    # The remote MCTS bot, sent the full state, takes the win.
    self.tool(
        "set_seat",
        session_id=session,
        seat=1,
        controller="remote",
        config={"url": self.bot, "state": "full_state"},
    )
    frame = self.tool("advance", session_id=session)["structuredContent"]
    self.assertEqual(frame["history"][-1]["action"], 5)
    self.assertEqual(frame["status"], "terminal")
    self.tool("close_session", session_id=session)
    gone = self.rpc(
        "tools/call",
        {"name": "get_frame", "arguments": {"session_id": session}},
    )["result"]
    self.assertTrue(gone["isError"])
    self.assertIn("not_found", gone["content"][0]["text"])

  def test_choose_chance_and_seat_views(self) -> None:
    opened = self.tool(
        "open_session", game="kuhn_poker", chance_policy="expose", seed=1
    )
    session = opened["structuredContent"]["frame"]["session"]["id"]
    self.tool("choose_chance", session_id=session, outcome=2)
    frame = self.tool("choose_chance", session_id=session, outcome=0)
    self.assertEqual(frame["structuredContent"]["num_plies"], 2)
    seat = self.tool("get_frame", session_id=session, viewer=1)[
        "structuredContent"
    ]
    self.assertIsNone(seat["state_text"])
    self.assertIsNone(seat["history"][0]["action"])
    spectator = self.tool(
        "get_legal_actions", session_id=session, viewer="spectator"
    )
    self.assertIn("withheld", spectator["content"][0]["text"])
    refused = self.rpc(
        "tools/call",
        {
            "name": "close_session",
            "arguments": {"session_id": session, "viewer": "spectator"},
        },
    )["result"]
    self.assertTrue(refused["isError"])
    self.assertIn("not_your_turn", refused["content"][0]["text"])
    self.tool("close_session", session_id=session, viewer=0)

  def test_arguments_of_the_wrong_shape_are_tool_errors(self) -> None:
    for name, arguments in (
        ("open_session", {"game": "tic_tac_toe", "seats": ["human", "human"]}),
        ("open_session", {"game": "tic_tac_toe", "parameters": "x"}),
        ("open_session", {"game": 7}),
        ("reset", {"session_id": "s", "history": ["a"]}),
        (
            "set_seat",
            {
                "session_id": "s",
                "seat": 0,
                "controller": "human",
                "autoplay": "yes",
            },
        ),
        ("get_frame", {"session_id": 7}),
    ):
      with self.subTest(name=name, arguments=arguments):
        result = self.rpc("tools/call", {"name": name, "arguments": arguments})[
            "result"
        ]
        self.assertTrue(result["isError"], result)
    # The server is still up.
    self.assertEqual(self.rpc("ping")["result"], {})

  def test_a_session_id_is_one_path_segment(self) -> None:
    result = self.rpc(
        "tools/call",
        {"name": "get_frame", "arguments": {"session_id": "../games"}},
    )["result"]
    self.assertTrue(result["isError"])
    self.assertIn("not_found", result["content"][0]["text"])

  def test_an_ambiguous_or_unknown_label_is_refused(self) -> None:
    opened = self.tool("open_session", game="tic_tac_toe")
    session = opened["structuredContent"]["frame"]["session"]["id"]
    refused = self.rpc(
        "tools/call",
        {
            "name": "act",
            "arguments": {"session_id": session, "label": "nonsense"},
        },
    )["result"]
    self.assertTrue(refused["isError"])
    self.assertIn("0 legal actions", refused["content"][0]["text"])


if __name__ == "__main__":
  absltest.main()
