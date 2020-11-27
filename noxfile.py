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

# Lint as: python3
"""An integration test building and testing open_spiel wheel."""
import os
import sys
import sysconfig

import nox


def get_distutils_tempdir():
  return (
      f"temp.{sysconfig.get_platform()}-{sys.version_info[0]}.{sys.version_info[1]}"
  )


@nox.session(python="3")
def tests(session):
  # Explicitly pull out CC and CXX if they are specified.
  child_env = {}  
  if os.environ.get("CC") is not None:
    child_env["CC"] = os.environ.get("CC")
  if os.environ.get("CXX") is not None:
    child_env["CXX"] = os.environ.get("CXX")
  print("nox build child environment:")
  print(child_env)
  session.install("-r", "requirements.txt")
  session.run("python3", "setup.py", "build", env=child_env)
  session.run("python3", "setup.py", "install", env=child_env)
  session.cd(os.path.join("build", get_distutils_tempdir()))
  session.run(
      "ctest", f"-j{4*os.cpu_count()}", "--output-on-failure", external=True)
