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

"""Game-specific query example."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags

import pyspiel

FLAGS = flags.FLAGS

flags.DEFINE_string("game", "negotiation", "Name of the game")


def main(_):
  print("Creating game: " + FLAGS.game)
  game = pyspiel.load_game(FLAGS.game)

  state = game.new_initial_state()

  print(str(state))

  # Need to apply the first chance node for items and utilities to be generated
  state.apply_action(0)

  print("Item pool: {}".format(state.item_pool()))
  print("Player 0 utils: {}".format(state.agent_utils(0)))
  print("Player 1 utils: {}".format(state.agent_utils(1)))

  state = game.new_initial_state()

  print(str(state))

  # Need to apply the first chance node for items and utilities to be generated
  state.apply_action(0)

  print("Item pool: {}".format(state.item_pool()))
  print("Player 0 utils: {}".format(state.agent_utils(0)))
  print("Player 1 utils: {}".format(state.agent_utils(1)))


if __name__ == "__main__":
  app.run(main)
