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
"""A test file for run_python_test.py."""

import sys

from absl import app
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_string("print_value", "hello world", "String to print.")
flags.DEFINE_integer("return_value", 0, "Return value for the process.")


def main(argv):
  print("Num args:", len(argv))
  print("argv[0]:", argv[0])
  print("print_value:", FLAGS.print_value)
  print("return_value:", FLAGS.return_value)
  sys.exit(FLAGS.return_value)


if __name__ == "__main__":
  app.run(main)
