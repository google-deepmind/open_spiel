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
"""A class to log stuff to a file, mainly useful in parallel situations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime
import os


class FileLogger(object):
  """A logger to print stuff to a file."""

  def __init__(self, path, name, quiet=False, also_to_stdout=False):
    self._fd = open(os.path.join(path, "log-{}.txt".format(name)), "w")
    self._quiet = quiet
    self.also_to_stdout = also_to_stdout

  def print(self, *args):
    # Date/time with millisecond precision.
    date_prefix = "[{}]".format(datetime.datetime.now().isoformat(" ")[:-3])
    print(date_prefix, *args, file=self._fd, flush=True)
    if self.also_to_stdout:
      print(date_prefix, *args, flush=True)

  def opt_print(self, *args):
    if not self._quiet:
      self.print(*args)

  def __enter__(self):
    return self

  def __exit__(self, unused_exception_type, unused_exc_value, unused_traceback):
    self.close()

  def close(self):
    if self._fd:
      self._fd.close()
      self._fd = None

  def __del__(self):
    self.close()
