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

"""Python nfg_writer example."""

from absl import app
from absl import flags

import pyspiel

FLAGS = flags.FLAGS

flags.DEFINE_string("game", "matrix_rps", "Name of the game")
flags.DEFINE_string("outfile", None, "File to send the output to.")


def main(_):
  game = pyspiel.load_game(FLAGS.game)
  nfg_text = pyspiel.game_to_nfg_string(game)

  if FLAGS.outfile is None:
    print(nfg_text)
  else:
    print("Exporting to {}".format(FLAGS.outfile))
    outfile = open(FLAGS.outfile, "w")
    outfile.write(nfg_text)
    outfile.close()


if __name__ == "__main__":
  app.run(main)
