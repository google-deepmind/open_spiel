# Copyright 2026 DeepMind Technologies Limited
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


"""Helpers shared by the remote bot server and MCP end-to-end tests.

remote_bot_server_test.py and mcp_test.py both start servers as processes
of their own. The helper lives in this module, not in a test, so that each
test can depend on it.
"""

import re
import subprocess
import sys
import time

# How long a process may take to start serving.
START_SECONDS = 60.0


def start(
    argv: list[str], pattern: str, env: dict[str, str]
) -> tuple[subprocess.Popen[str], str]:
  """Starts a process and returns it and the URL it prints when serving.

  Args:
    argv: The arguments after the Python interpreter, such as `-m` and a
      module.
    pattern: A regular expression whose first group is the URL, matched
      against each line the process prints.
    env: The process's environment.

  Raises:
    AssertionError: If the process exits, or prints no matching line within
      START_SECONDS.
  """
  process = subprocess.Popen(  # pylint: disable=consider-using-with
      [sys.executable, *argv],
      stdout=subprocess.PIPE,
      stderr=subprocess.DEVNULL,
      text=True,
      env=env,
  )
  deadline = time.monotonic() + START_SECONDS
  assert process.stdout is not None
  while time.monotonic() < deadline:
    line = process.stdout.readline()
    if not line:
      break
    match = re.search(pattern, line)
    if match:
      return process, match.group(1).rstrip("/")
  process.kill()
  raise AssertionError(f"{argv} did not start serving.")
