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


"""An MCP server, so that an agent can drive the GUI's sessions.

  python -m open_spiel.python.gui.mcp --server http://127.0.0.1:8080

speaks the Model Context Protocol over stdin and stdout (JSON-RPC 2.0, one
message per line) to an agent that starts it, and drives a running GUI
server through its HTTP API (CONTRACT.md). Its tools list games, open and
close sessions, read a frame, its legal actions and its history for any
viewer, act, choose chance outcomes, advance, undo, reset, replay earlier
positions and change seats. A session an agent opens shows in a browser at
the URL `open_session` returns, and one a person opened in a browser can be
driven by its id.

It opens no port of its own. It drives a GUI server on this machine unless
started with `--allow-remote-server`, and holds no keys or login: it can do
exactly what a browser on this machine can.
"""

import argparse
from collections.abc import Mapping, Sequence
import http.client
import json
import sys
from typing import Any, TextIO
import urllib.parse

from absl import logging

from open_spiel.python.gui import mcp_client
from open_spiel.python.gui import mcp_tools
from open_spiel.python.gui import remote_bot

SERVER_NAME = "openspiel-gui"

# JSON-RPC error codes.
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603


class McpServer:
  """Answers MCP messages."""

  def __init__(self, tools: mcp_tools.Tools) -> None:
    self.tools = tools

  def handle(self, message: object) -> dict[str, Any] | None:
    """Returns the response to one JSON-RPC message, or None for a notice."""
    if not isinstance(message, dict) or message.get("jsonrpc") != "2.0":
      return _error(None, INVALID_REQUEST, "Not a JSON-RPC 2.0 message.")
    method = message.get("method")
    if "id" not in message:
      return None  # A notification: initialized, cancelled, ...
    ident = message["id"]
    params = message.get("params") or {}
    if not isinstance(method, str) or not isinstance(params, dict):
      return _error(ident, INVALID_REQUEST, "No method, or bad params.")
    if method == "initialize":
      # The tools use only what every published version of the protocol
      # has, so the server speaks the version the client asks for.
      version = params.get("protocolVersion")
      if not isinstance(version, str) or not version:
        return _error(
            ident, INVALID_PARAMS, "initialize names no protocolVersion."
        )
      return _result(
          ident,
          {
              "protocolVersion": version,
              "capabilities": {"tools": {"listChanged": False}},
              "serverInfo": {"name": SERVER_NAME, "version": "1"},
              "instructions": (
                  "Drives the OpenSpiel GUI server at "
                  f"{self.tools.client.server}. list_games, then "
                  "open_session; act, advance and get_frame. The viewer "
                  "argument picks whose view a frame shows (oracle by "
                  "default)."
              ),
          },
      )
    if method == "ping":
      return _result(ident, {})
    if method == "tools/list":
      return _result(
          ident, {"tools": [tool.describe() for tool in mcp_tools.TOOLS]}
      )
    if method == "tools/call":
      return self._call(ident, params)
    return _error(ident, METHOD_NOT_FOUND, f"No method {method!r}.")

  def _call(self, ident: object, params: Mapping[str, Any]) -> dict[str, Any]:
    """Runs a tool."""
    tool = mcp_tools.BY_NAME.get(str(params.get("name")))
    if tool is None:
      return _error(ident, INVALID_PARAMS, f"No tool {params.get('name')!r}.")
    arguments = params.get("arguments") or {}
    if not isinstance(arguments, dict):
      return _error(ident, INVALID_PARAMS, "arguments is not an object.")
    unknown = sorted(set(arguments) - set(tool.properties))
    if unknown:
      return _error(
          ident, INVALID_PARAMS, f"{tool.name} takes no {', '.join(unknown)}."
      )
    try:
      text, value = tool.call(self.tools, arguments)
    except mcp_client.ToolError as e:
      return _result(
          ident,
          {"content": [{"type": "text", "text": str(e)}], "isError": True},
      )
    except (OSError, http.client.HTTPException) as e:
      return _result(
          ident,
          {
              "content": [{
                  "type": "text",
                  "text": (
                      f"The GUI server at {self.tools.client.server} cannot "
                      f"be reached: {e}."
                  ),
              }],
              "isError": True,
          },
      )
    except Exception as e:  # pylint: disable=broad-exception-caught
      # A fault, not a refusal: logged in full, and answered as an internal
      # error, so that the agent keeps its server.
      logging.exception("The %s tool failed.", tool.name)
      return _error(
          ident, INTERNAL_ERROR, f"The {tool.name} tool failed: {e!r}."
      )
    result: dict[str, Any] = {
        "content": [{"type": "text", "text": text}],
        "isError": False,
    }
    if isinstance(value, dict):
      result["structuredContent"] = value
    return _result(ident, result)


def _result(ident: object, result: object) -> dict[str, Any]:
  """A JSON-RPC result."""
  return {"jsonrpc": "2.0", "id": ident, "result": result}


def _error(ident: object, code: int, message: str) -> dict[str, Any]:
  """A JSON-RPC error."""
  return {
      "jsonrpc": "2.0",
      "id": ident,
      "error": {"code": code, "message": message},
  }


def serve(server: McpServer, stdin: TextIO, stdout: TextIO) -> None:
  """Answers messages, one JSON object per line, until stdin closes."""
  for line in stdin:
    if not line.strip():
      continue
    try:
      message = json.loads(line)
    except json.JSONDecodeError as e:
      response: dict[str, Any] | None = _error(None, PARSE_ERROR, e.msg)
    else:
      response = server.handle(message)
    if response is not None:
      stdout.write(json.dumps(response) + "\n")
      stdout.flush()


def server_refusal(url: str, allow_remote: bool) -> str | None:
  """Returns why the MCP server may not drive the GUI server at `url`."""
  parts = urllib.parse.urlsplit(url)
  if parts.scheme not in ("http", "https") or not parts.hostname:
    return f"--server {url!r} is not an http URL with a host."
  if allow_remote or remote_bot.is_loopback(parts.hostname.lower()):
    return None
  return (
      f"--server {url!r} is not on this machine; pass --allow-remote-server "
      "to drive it."
  )


def main(argv: Sequence[str] | None = None) -> None:
  """Serves MCP on stdin and stdout."""
  parser = argparse.ArgumentParser(
      prog="python -m open_spiel.python.gui.mcp",
      description="An MCP server that drives an OpenSpiel GUI server.",
  )
  parser.add_argument(
      "--server",
      default="http://127.0.0.1:8080",
      help="The GUI server's URL. Default: %(default)s.",
  )
  parser.add_argument(
      "--allow-remote-server",
      action="store_true",
      help="Drive a GUI server that is not on this machine.",
  )
  args = parser.parse_args(argv)
  refusal = server_refusal(args.server, args.allow_remote_server)
  if refusal is not None:
    raise SystemExit(refusal)
  # stdout carries the protocol: logs go to stderr.
  logging.use_absl_handler()
  logging.info("MCP server for the GUI server at %s.", args.server)
  serve(
      McpServer(mcp_tools.Tools(mcp_client.GuiClient(args.server))),
      sys.stdin,
      sys.stdout,
  )


if __name__ == "__main__":
  main()
