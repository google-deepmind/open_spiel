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
    self._check_build_environment()
    for ext in self.extensions:
      self.build_extension(ext)

  def _check_build_environment(self):
    """Check for required build tools: CMake, C++ compiler, and python dev."""
    try:
      subprocess.check_call(["cmake", "--version"])
    except OSError as e:
      ext_names = ", ".join(e.name for e in self.extensions)
      raise RuntimeError(
          f"CMake must be installed to build the following extensions: {ext_names}"
      ) from e
    print("Found CMake")

    cxx = "clang++"
    if os.environ.get("CXX") is not None:
      cxx = os.environ.get("CXX")
    try:
      subprocess.check_call([cxx, "--version"])
    except OSError as e:
      ext_names = ", ".join(e.name for e in self.extensions)
      raise RuntimeError(
          "A C++ compiler that supports c++17 must be installed to build the "
          + "following extensions: {}".format(ext_names)
          + ". We recommend: Clang version >= 7.0.0."
      ) from e
    print("Found C++ compiler: {}".format(cxx))

  def build_extension(self, ext):
    extension_dir = os.path.abspath(
        os.path.dirname(self.get_ext_fullpath(ext.name)))
    cxx = "clang++"
    if os.environ.get("CXX") is not None:
      cxx = os.environ.get("CXX")
    env = os.environ.copy()
    # If not specified, assume ACPC and Hanabi are built in.
    # Disable this by passing e.g. OPEN_SPIEL_BUILD_WITH_ACPC=OFF when building
    if env.get("OPEN_SPIEL_BUILD_WITH_ACPC") is None:
      env["OPEN_SPIEL_BUILD_WITH_ACPC"] = "ON"
    if env.get("OPEN_SPIEL_BUILD_WITH_HANABI") is None:
      env["OPEN_SPIEL_BUILD_WITH_HANABI"] = "ON"
    cmake_args = [
        f"-DPython3_EXECUTABLE={sys.executable}",
        f"-DCMAKE_CXX_COMPILER={cxx}",
        f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extension_dir}",
    ]
    if not os.path.exists(self.build_temp):
      os.makedirs(self.build_temp)
    subprocess.check_call(
        ["cmake", ext.sourcedir] + cmake_args, cwd=self.build_temp,
        env=env)
    if os.environ.get("OPEN_SPIEL_BUILD_ALL") is not None:
      # Build everything (necessary for nox tests)
      subprocess.check_call(["make", f"-j{os.cpu_count()}"],
                            cwd=self.build_temp,
                            env=env)
    else:
      # Build only pyspiel (for pip package)
      subprocess.check_call(["make", "pyspiel", f"-j{os.cpu_count()}"],
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


# Get the requirements from file. During nox tests, this is in the current
# directory, but when installing from pip it is in the parent directory
req_file = ""
if os.path.exists("requirements.txt"):
  req_file = "requirements.txt"
else:
  req_file = "../requirements.txt"

setuptools.setup(
    name="open_spiel",
    version="1.1.0",
    license="Apache 2.0",
    author="The OpenSpiel authors",
    author_email="open_spiel@google.com",
    description="A Framework for Reinforcement Learning in Games",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/deepmind/open_spiel",
    install_requires=_get_requirements(req_file),
    python_requires=">=3",
    ext_modules=[CMakeExtension("pyspiel", sourcedir="open_spiel")],
    cmdclass={"build_ext": BuildExt},
    zip_safe=False,
    packages=setuptools.find_packages(include=["open_spiel", "open_spiel.*"])
)
