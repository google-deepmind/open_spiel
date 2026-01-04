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

"""Example of MMD with dilated entropy to solve for QRE in Leduc Poker."""

from absl import app
from absl import flags

from open_spiel.python.algorithms import mmd_dilated
import pyspiel

FLAGS = flags.FLAGS

flags.DEFINE_integer("iterations", 100, "Number of iterations")
flags.DEFINE_float(
    "alpha", 0.05, "QRE parameter, larger value amounts to more regularization")
flags.DEFINE_string("game", "leduc_poker", "Name of the game")
flags.DEFINE_integer("print_freq", 10, "How often to print the gap")


def main(_):
  game = pyspiel.load_game(FLAGS.game)
  mmd = mmd_dilated.MMDDilatedEnt(game, FLAGS.alpha)

  for i in range(FLAGS.iterations):
    mmd.update_sequences()
    if i % FLAGS.print_freq == 0:
      conv = mmd.get_gap()
      print("Iteration {} gap {}".format(i, conv))


if __name__ == "__main__":
  app.run(main)
