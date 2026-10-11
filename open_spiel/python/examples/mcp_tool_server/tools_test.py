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

from absl.testing import absltest
import numpy as np
from open_spiel.python.examples.mcp_tool_server import open_spiel_tools

SEED = 28927119
TERMINAL_PLAYER_ID = -4


class ToolsTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.os_tools = open_spiel_tools.OpenSpielTools()

  def test_play_tic_tac_toe(self):
    session_str = self.os_tools.start_game("tic_tac_toe", 0)
    while self.os_tools.current_player(session_str) != TERMINAL_PLAYER_ID:
      observation = self.os_tools.get_observation(session_str)
      print(f"Observation:\n{observation}")
      legal_actions = self.os_tools.legal_actions(session_str)
      action = np.random.choice(legal_actions)
      self.os_tools.play_action(session_str, action)
    print(f"Game over. Return: {self.os_tools.get_return(session_str)}")

  def test_play_crossword_game(self):
    session_str = self.os_tools.start_game("crossword", 0)
    observation = self.os_tools.get_observation(session_str)
    print(f"Observation:\n {observation}")
    self.os_tools.play_action(session_str, '{"clue_id": "A1", "word": "GO"}')
    observation = self.os_tools.get_observation(session_str)
    print(f"Observation:\n {observation}")


if __name__ == "__main__":
  absltest.main()
