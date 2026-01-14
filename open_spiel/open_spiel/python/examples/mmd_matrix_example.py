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

"""Example: using MMD with dilated entropy to solve for QRE in a matrix Game."""

from absl import app
from absl import flags

from open_spiel.python.algorithms import mmd_dilated
import pyspiel

FLAGS = flags.FLAGS

flags.DEFINE_integer("iterations", 1000, "Number of iterations")
flags.DEFINE_float(
    "alpha", 0.1, "QRE parameter, larger value amounts to more regularization")
flags.DEFINE_integer("print_freq", 100, "How often to print the gap")

# create pyspiel perturbed RPS matrix game

game = pyspiel.create_matrix_game([[0, -1, 3], [1, 0, -3], [-3, 3, 0]],
                                  [[0, 1, -3], [-1, 0, 3], [3, -3, 0]])

game = pyspiel.convert_to_turn_based(game)


def main(_):
  mmd = mmd_dilated.MMDDilatedEnt(game, FLAGS.alpha)
  for i in range(FLAGS.iterations):
    mmd.update_sequences()
    if i % FLAGS.print_freq == 0:
      conv = mmd.get_gap()
      print("Iteration {} gap {}".format(i, conv))

  # Extract policies for both players
  print(mmd.get_policies().action_probability_array)
  # Note the sequence form and behavioural-form coincide
  # for a normal-form game (sequence form has extra root value of 1)
  print(mmd.current_sequences())


if __name__ == "__main__":
  app.run(main)
