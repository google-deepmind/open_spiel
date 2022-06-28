""" Example of using MMD with dilated entropy
    to solve for QRE in Leduc Poker """

from absl import app
from absl import flags

from open_spiel.python.algorithms import mmd_dilated
import pyspiel

FLAGS = flags.FLAGS

flags.DEFINE_integer("iterations", 100, "Number of iterations")
flags.DEFINE_float("alpha", 0.05, "QRE parameter, larger value amounts to more regularization")
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