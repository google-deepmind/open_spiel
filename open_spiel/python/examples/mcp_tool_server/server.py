# Copyright 2019 DeepMind Technologies Limited
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

from absl import app
from fastmcp import FastMCP
from open_spiel.python.examples.mcp_tool_server import open_spiel_tools


def main(_):
  server = FastMCP("Open Spiel Tool Server")
  tools = open_spiel_tools.OpenSpielTools()
  server.add_tool(tools.list_games)
  server.add_tool(tools.game_info)
  server.add_tool(tools.start_game)
  server.add_tool(tools.reset_game)
  server.add_tool(tools.current_player)
  server.add_tool(tools.legal_actions)
  server.add_tool(tools.play_action)
  server.add_tool(tools.get_return)
  server.add_tool(tools.get_observation)
  server.add_tool(tools.mcts_action_chooser)
  
  # serve via HTTP:
  server.run(transport="streamable-http", host="127.0.0.1", port=8000)
  
  # Serve via STDIO:
  # server.run()


if __name__ == '__main__':
  app.run(main)
