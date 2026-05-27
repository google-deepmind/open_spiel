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

"""Tests for the game-specific functions for go_fish"""

from absl import flags
from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

import pyspiel
from open_spiel.python.utils import file_utils


FLAGS = flags.FLAGS

# From CMakeLists.txt:Python tests are run from the main binary directory which
# will be something like build/python.
flags.DEFINE_string(
    "chess960_fens_file",
    "../../open_spiel/games/chess/chess960_starting_positions.txt",
    "FENs database for chess960",
)


class GamesCrazyhouseTest(parameterized.TestCase):

  def test_bindings_sim(self):
    game = pyspiel.load_game("go_fish")
    state = game.new_initial_state()
    count = 0

    while count < 200 and not state.is_terminal():
      #print(state)
      #print()
      count += 1
      legal_actions = state.legal_actions()
      action = np.random.choice(legal_actions)
      print("action", action)
      state.apply_action(action)
    if state.is_terminal():
        print("returns", state.returns())


  # state.to_string does not save the whole state, it loses ask history.
  # but if we create a state from a string, to_string must return the same string 
  def test_string_loop(self):
      game = pyspiel.load_game("go_fish")
      state = game.new_initial_state()
      count = 0
      # make some random state
      while count < 20:
          legal_actions = state.legal_actions()
          action = np.random.choice(legal_actions)
          state.apply_action(action)
          count += 1
      ss = str(state)
      #print(ss)
      state = game.new_initial_state(ss)
      ss2 = str(state)
      print(ss2)
      assert(ss == ss2)

  def test_instant_win(self):
      """In the end_on_first_out variation a player can win at the end of the deal."""
      game = pyspiel.load_game("go_fish", {"initial_cards":4, "suits":4,
        "ranks":4, "end_on_first_out":True, "players":2})
      start = "Deal\n0\na4:0\nb3:0\nb1c4d4"
      state = game.new_initial_state(start)
      state.apply_action(2)
      assert(state.returns == [1.0, -1.0])


if __name__ == "__main__":
  np.random.seed(87375711)
  absltest.main()
