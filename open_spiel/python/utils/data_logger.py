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
"""Log data to a jsonl file."""

import datetime
import json
import os
import time
from typing import Any, Dict, Text

from open_spiel.python.utils import gfile


class DataLoggerJsonLines:
  """Log data to a jsonl file."""

  def __init__(self, path: str, name: str, flush=True):
    self._fd = gfile.Open(os.path.join(path, name + ".jsonl"), "w")
    self._flush = flush
    self._start_time = time.time()

  def __del__(self):
    self.close()

  def close(self):
    if hasattr(self, "_fd") and self._fd is not None:
      self._fd.flush()
      self._fd.close()
      self._fd = None

  def flush(self):
    self._fd.flush()

  def write(self, data: Dict[Text, Any]):
    now = time.time()
    data["time_abs"] = now
    data["time_rel"] = now - self._start_time
    dt_now = datetime.datetime.utcfromtimestamp(now)
    data["time_str"] = dt_now.strftime("%Y-%m-%d %H:%M:%S.%f +0000")
    self._fd.write(json.dumps(data))
    self._fd.write("\n")
    if self._flush:
      self.flush()
