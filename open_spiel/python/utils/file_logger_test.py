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
"""Tests for open_spiel.python.utils.file_logger."""

import os
import tempfile

from absl.testing import absltest

from open_spiel.python.utils import file_logger


class FileLoggerTest(absltest.TestCase):

  def test_file_logger(self):
    tmp_dir = tempfile.mkdtemp()
    try:
      log_name = "test"
      log_file_name = os.path.join(tmp_dir, "log-{}.txt".format(log_name))

      self.assertTrue(os.path.isdir(tmp_dir))
      self.assertFalse(os.path.exists(log_file_name))

      with file_logger.FileLogger(tmp_dir, log_name) as logger:
        logger.print("line 1")
        logger.print("line", 2)
        logger.print("line", 3, "asdf")

      with open(log_file_name, "r") as f:
        lines = f.readlines()

        self.assertLen(lines, 3)
        self.assertIn("line 1", lines[0])
        self.assertIn("line 2", lines[1])
        self.assertIn("line 3 asdf", lines[2])
    finally:
      if os.path.exists(log_file_name):
        os.remove(log_file_name)
      os.rmdir(tmp_dir)


if __name__ == "__main__":
  absltest.main()
