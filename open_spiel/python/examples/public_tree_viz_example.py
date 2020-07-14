# Copyright 2019 DeepMind Technologies Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""Public tree visualization example.

Example usage:

  python public_tree_viz_example.py --game="kuhn_poker"  \
      --target_pubset="start game,Deal to player 0,Deal to player 1,Bet"
"""

from absl import app
from absl import flags
from absl import logging

import pyspiel
from open_spiel.python.visualizations import public_tree_viz
from open_spiel.python.visualizations import treeviz

FLAGS = flags.FLAGS
flags.DEFINE_string("game", "kuhn_poker", "Name of the game")
flags.DEFINE_string("out", "/tmp/gametree.png", "Name of output file, e.g., "
                    "[*.png|*.pdf].")
flags.DEFINE_enum("prog", "dot", ["dot", "neato", "circo"], "Graphviz layout.")
flags.DEFINE_string(
    "target_pubset", None,
    "Limit grouping of public states only to the specified "
    "public state.")
flags.DEFINE_boolean("verbose", False, "Whether to print verbose output.")


def _zero_sum_node_decorator(state):
  """Custom node decorator that only shows the return of the first player."""
  attrs = treeviz.default_node_decorator(state)  # get default attributes
  if state.is_terminal():
    attrs["label"] = str(int(state.returns()[0]))
  return attrs


def main(argv):
  del argv

  game = pyspiel.load_game_with_public_states(FLAGS.game)
  base_game = game.get_base_game()
  game_type = base_game.get_type()
  two_pl_zero_sum = game_type.utility == pyspiel.GameType.Utility.ZERO_SUM \
    and base_game.num_players() == 2
  node_decorator = _zero_sum_node_decorator if two_pl_zero_sum \
    else treeviz.default_node_decorator

  gametree = public_tree_viz.GamePublicTree(
      public_game=game,
      draw_world=True,
      target_public_to_base=FLAGS.target_pubset,
      node_decorator=node_decorator,
      group_pubsets=True)

  if FLAGS.verbose:
    logging.info("Game tree:\n%s", gametree.to_string())

  gametree.draw(FLAGS.out, prog=FLAGS.prog)
  logging.info("Game tree saved to file: %s", FLAGS.out)


if __name__ == "__main__":
  app.run(main)
