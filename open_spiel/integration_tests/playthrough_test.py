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

_DATA_DIR = "google3/third_party/open_spiel/integration_tests/playthroughs/"


class PlaythroughTest(absltest.TestCase):

  def test_rerun_playthroughs(self):
    test_srcdir = os.environ.get("TEST_SRCDIR", "")
    path = os.path.join(test_srcdir, _DATA_DIR)

    file_paths = list(os.listdir(path))
    self.assertGreaterEqual(len(file_paths), 5)

    for filename in file_paths:
      file_path = os.path.join(path, filename)
      logging.info(filename)
      expected, actual = generate_playthrough.replay(file_path)
      for expected_line, actual_line in zip(
          expected.split("\n"), actual.split("\n")):
        self.assertEqual(expected_line, actual_line)

      self.assertMultiLineEqual(
          expected, actual, msg="Issue with filename {}".format(filename))


if __name__ == "__main__":
  absltest.main()
