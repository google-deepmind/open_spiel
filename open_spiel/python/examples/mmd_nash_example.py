""" Example of using MMD with dilated entropy
    to compute a Nash Eq in Kuhn Poker """


from absl import app
from absl import flags

from open_spiel.python.algorithms import mmd_dilated
from open_spiel.python.algorithms import exploitability

import pyspiel

FLAGS = flags.FLAGS

flags.DEFINE_integer("iterations", 1000, "Number of iterations")
flags.DEFINE_string("game", "kuhn_poker", "Name of the game")
flags.DEFINE_integer("print_freq", 100, "How often to print the exploitability")


def main(_):
  game = pyspiel.load_game(FLAGS.game)
  mmd = mmd_dilated.MMDDilatedEnt(game, alpha=0, stepsize=1)

  for i in range(FLAGS.iterations):
    mmd.update_sequences()
    if i % FLAGS.print_freq == 0:
      conv = exploitability.exploitability(game, mmd.get_avg_policies())
      print("Iteration {} exploitability {}".format(i, conv))


if __name__ == "__main__":
  app.run(main)