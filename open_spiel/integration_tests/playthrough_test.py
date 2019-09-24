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

"""Re-run playthroughs and check for differences."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl import logging
from absl.testing import absltest

from open_spiel.python.algorithms import generate_playthrough
import pyspiel

_DATA_DIR = "open_spiel/integration_tests/playthroughs/"

_OPTIONAL_GAMES = set(["hanabi"])
_AVAILABLE_GAMES = set(pyspiel.registered_names())


def _is_optional_game(basename):
  """Returns (bool, game_name or None).

  Args:
    basename: The basename of the file. It is assumed it starts with the game
      name.
  """
  for game_name in _OPTIONAL_GAMES:
    if basename.startswith(game_name):
      return True, game_name
  return False, None


class PlaythroughTest(absltest.TestCase):

  def test_rerun_playthroughs(self):
    test_srcdir = os.environ.get("TEST_SRCDIR", "")
    path = os.path.join(test_srcdir, _DATA_DIR)

    basenames = list(os.listdir(path))
    self.assertGreaterEqual(len(basenames), 40)

    for basename in basenames:
      file_path = os.path.join(path, basename)
      logging.info(basename)

      # We check whether the game is optional, and if it is, whether we do
      # have the game.
      is_optional, game_name = _is_optional_game(basename)
      if is_optional:
        if game_name not in _AVAILABLE_GAMES:
          logging.info("Skipping %s because %s is not built in.", basename,
                       game_name)
          continue

      expected, actual = generate_playthrough.replay(file_path)
      for expected_line, actual_line in zip(
          expected.split("\n"), actual.split("\n")):
        self.assertEqual(expected_line, actual_line)

      self.assertMultiLineEqual(
          expected, actual, msg="Issue with basename {}".format(basename))


if __name__ == "__main__":
  absltest.main()
