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

"""Unit test for GamutGenerator."""

from absl import app
from absl.testing import absltest
import pyspiel


class GamutGeneratorTest(absltest.TestCase):

  def test_generate_game(self):
    generator = pyspiel.GamutGenerator(
        "gamut.jar")

    # Using a list of arguments.
    game = generator.generate_game([
        "-g", "RandomGame", "-players", "4", "-normalize", "-min_payoff", "0",
        "-max_payoff", "150", "-actions", "2", "4", "5", "7"
    ])
    self.assertIsNotNone(game)

    # Using a string of arguments.
    game = generator.generate_game(
        "-g RandomGame -players 4 -normalize -min_payoff 0 " +
        "-max_payoff 150 -actions 2 4 5 7")
    self.assertIsNotNone(game)


def main(_):
  absltest.main()


if __name__ == "__main__":
  # Calling main via app.run here is necessary for internal uses.
  app.run(main)
