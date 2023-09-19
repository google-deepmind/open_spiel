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

"""Unit test for GamutGenerator."""

from absl import app
from absl.testing import absltest
from absl.testing import parameterized
from open_spiel.python.egt.utils import game_payoffs_array
import pyspiel


class GamutGeneratorTest(parameterized.TestCase):

  def _gamut_generator(self):
    return pyspiel.GamutGenerator(
        "gamut.jar"
    )

  @parameterized.parameters(
      "-g BertrandOligopoly -players 2 -actions 4 -random_params",
      "-g UniformLEG-CG -players 2 -actions 4 -random_params",
      "-g PolymatrixGame-SW -players 2 -actions 4 -random_params",
      "-g GraphicalGame-SW -players 2 -actions 4 -random_params",
      "-g BidirectionalLEG-CG -players 2 -actions 4 -random_params",
      "-g CovariantGame -players 2 -actions 4 -random_params",
      "-g DispersionGame -players 2 -actions 4 -random_params",
      "-g MinimumEffortGame -players 2 -actions 4 -random_params",
      "-g RandomGame -players 2 -actions 4 -random_params",
      "-g TravelersDilemma -players 2 -actions 4 -random_params",
  )
  def test_generate_game(self, game_str):
    generator = self._gamut_generator()
    # Using a string of arguments.
    game = generator.generate_game(game_str)
    self.assertIsNotNone(game)

    payoff_tensor = game_payoffs_array(game)
    self.assertEqual(payoff_tensor.shape, (2, 4, 4))

  def test_gamut_api(self):
    generator = self._gamut_generator()

    # See the documentation at http://gamut.stanford.edu/ for the commands
    # needed to generate the various different games.

    # Using a string of arguments.
    game = generator.generate_game(
        "-g RandomGame -players 4 -normalize -min_payoff 0 "
        + "-max_payoff 150 -actions 2 4 5 7"
    )
    self.assertIsNotNone(game)

    # Using a list of arguments.
    game = generator.generate_game([
        "-g",
        "RandomGame",
        "-players",
        "4",
        "-normalize",
        "-min_payoff",
        "0",
        "-max_payoff",
        "150",
        "-actions",
        "2",
        "4",
        "5",
        "7",
    ])
    self.assertIsNotNone(game)

    # Using a list of arguments.
    matrix_game = generator.generate_matrix_game([
        "-g",
        "RandomGame",
        "-players",
        "2",
        "-normalize",
        "-min_payoff",
        "0",
        "-max_payoff",
        "150",
        "-actions",
        "10",
        "15",
    ])
    self.assertIsNotNone(matrix_game)
    print(matrix_game.new_initial_state())
    payoff_matrix = game_payoffs_array(matrix_game)
    print(payoff_matrix.shape)
    print(payoff_matrix)

    # Using a list of arguments.
    tensor_game = generator.generate_game([
        "-g",
        "RandomGame",
        "-players",
        "4",
        "-normalize",
        "-min_payoff",
        "0",
        "-max_payoff",
        "150",
        "-actions",
        "2",
        "4",
        "5",
        "7",
    ])
    self.assertIsNotNone(tensor_game)
    payoff_tensor = game_payoffs_array(tensor_game)
    print(payoff_tensor.shape)


def main(_):
  absltest.main()


if __name__ == "__main__":
  # Calling main via app.run here is necessary for internal uses.
  app.run(main)
