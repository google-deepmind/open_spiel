# Copyright 2022 DeepMind Technologies Limited
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

"""Example: MMD with dilated entropy to compute approx. Nash in Kuhn poker."""

from absl import app
from absl import flags

from open_spiel.python.algorithms import exploitability
from open_spiel.python.algorithms import mmd_dilated
import pyspiel

FLAGS = flags.FLAGS

flags.DEFINE_integer("iterations", 1000, "Number of iterations")
flags.DEFINE_string("game", "kuhn_poker", "Name of the game")
flags.DEFINE_integer("print_freq", 100, "How often to print the exploitability")


def main(_):
  game = pyspiel.load_game(FLAGS.game)
  # need to manually set stepsize if alpha = 0
  mmd = mmd_dilated.MMDDilatedEnt(game, alpha=0, stepsize=1)

  for i in range(FLAGS.iterations):
    mmd.update_sequences()
    if i % FLAGS.print_freq == 0:
      conv = exploitability.exploitability(game, mmd.get_avg_policies())
      print("Iteration {} exploitability {}".format(i, conv))


if __name__ == "__main__":
  app.run(main)
