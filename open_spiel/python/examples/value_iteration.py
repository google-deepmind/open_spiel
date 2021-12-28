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

"""Python spiel example to use value iteration to solve a game."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags

from open_spiel.python.algorithms import value_iteration
import pyspiel

FLAGS = flags.FLAGS
flags.DEFINE_string("game", "tic_tac_toe", "Name of the game")


def play_tic_tac_toe():
  """Solves tic tac toe."""
  game = pyspiel.load_game("tic_tac_toe")

  print("Solving the game; depth_limit = {}".format(-1))
  values = value_iteration.value_iteration(game, -1, 0.01)

  for state, value in values.items():
    print("")
    print(str(state))
    print("Value = {}".format(value))

  initial_state = "...\n...\n..."
  cross_win_state = "...\n...\n.ox"
  naught_win_state = "x..\noo.\nxx."

  assert values[initial_state] == 0, "State should be drawn: \n" + initial_state
  assert values[cross_win_state] == 1, ("State should be won by player 0: \n" +
                                        cross_win_state)
  assert values[naught_win_state] == -1, (
      "State should be won by player 1: \n" + cross_win_state)


def main(argv):
  del argv
  if FLAGS.game == "tic_tac_toe":
    play_tic_tac_toe()
  else:
    raise NotImplementedError("This example only works for Tic-Tac-Toe.")


if __name__ == "__main__":
  app.run(main)
