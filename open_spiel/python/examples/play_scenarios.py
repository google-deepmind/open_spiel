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

# Lint as: python3
"""Plays a uniform random bot against the default scenarios for that game."""

import random
from absl import app
from absl import flags

from open_spiel.python.bots import scenarios
from open_spiel.python.bots import uniform_random
import pyspiel

FLAGS = flags.FLAGS
flags.DEFINE_string("game_name", "catch", "Game to play scenarios for.")


def main(argv):
  del argv
  game = pyspiel.load_game(FLAGS.game_name)

  # TODO(author1): Add support for bots from neural networks.
  bots = [
      uniform_random.UniformRandomBot(i, random)
      for i in range(game.num_players())
  ]
  scenarios.play_bot_in_scenarios(game, bots)


if __name__ == "__main__":
  app.run(main)
