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
"""The setup script for setuptools.

See https://setuptools.readthedocs.io/en/latest/setuptools.html
"""

import os
import subprocess
import sys

import setuptools
from setuptools.command.build_ext import build_ext


class CMakeExtension(setuptools.Extension):
  """An extension with no sources.

  We do not want distutils to handle any of the compilation (instead we rely
  on CMake), so we always pass an empty list to the constructor.
  """

  def __init__(self, name, sourcedir=""):
    super().__init__(name, sources=[])
    self.sourcedir = os.path.abspath(sourcedir)


class BuildExt(build_ext):
  """Our custom build_ext command.

  Uses CMake to build extensions instead of a bare compiler (e.g. gcc, clang).
  """

  def run(self):
    try:
      subprocess.check_call(["cmake", "--version"])
    except OSError as e:
      ext_names = ", ".join(e.name for e in self.extensions)
      raise RuntimeError(
          f"CMake must be installed to build the following extensions: {ext_names}"
      ) from e

    for ext in self.extensions:
      self.build_extension(ext)

  def build_extension(self, ext):
    compiler = "clang++"
    if os.environ.get("CXX") is not None:
      compiler = os.environ.get("CXX")
    extension_dir = os.path.abspath(
        os.path.dirname(self.get_ext_fullpath(ext.name)))
    cmake_args = [
        f"-DPython3_EXECUTABLE={sys.executable}",
        "-DCMAKE_CXX_COMPILER=" + compiler,
        f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extension_dir}",
    ]
    if not os.path.exists(self.build_temp):
      os.makedirs(self.build_temp)
    env = os.environ.copy()
    subprocess.check_call(
        ["cmake", ext.sourcedir] + cmake_args, cwd=self.build_temp, env=env)
    subprocess.check_call(["make", f"-j{os.cpu_count()}"],
                          cwd=self.build_temp,
                          env=env)


def _get_requirements(requirements_file):  # pylint: disable=g-doc-args
  """Returns a list of dependencies for setup() from requirements.txt.

  Currently a requirements.txt is being used to specify dependencies. In order
  to avoid specifying it in two places, we're going to use that file as the
  source of truth.
  """
  with open(requirements_file) as f:
    return [_parse_line(line) for line in f if line]


def _parse_line(s):
  """Parses a line of a requirements.txt file."""
  requirement, *_ = s.split("#")
  return requirement.strip()


setuptools.setup(
    name="pyspiel",
    version="0.0.1rc2",
    license="Apache 2.0",
    author="The OpenSpiel authors",
    author_email="open_spiel@google.com",
    description="A Framework for Reinforcement Learning in Games",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/deepmind/open_spiel",
    install_requires=_get_requirements("requirements.txt"),
    ext_modules=[CMakeExtension("pyspiel", sourcedir="open_spiel")],
    cmdclass={"build_ext": BuildExt},
    zip_safe=False,
)
