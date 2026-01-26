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

"""The setup script for setuptools.

See https://setuptools.readthedocs.io/en/latest/setuptools.html"""

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
    """
    Check for required build tools: CMake, C++ compiler, and python dev.
    """
    try:
      subprocess.check_call(["cmake", "--version"])
    except OSError as e:
      ext_names = ", ".join(e.name for e in self.extensions)
      raise RuntimeError(
          "CMake must be installed to build" +
          f"the following extensions: {ext_names}") from e
    print("Found CMake")

    cxx = "clang++"
    if os.environ.get("CXX") is not None:
      cxx = os.environ.get("CXX")
    try:
      subprocess.check_call([cxx, "--version"])
    except OSError as e:
      ext_names = ", ".join(e.name for e in self.extensions)
      raise RuntimeError(
          f"A C++ compiler that supports c++17 must be installed to build"
          f" the following extensions: {ext_names}."
          f"We recommend: Clang version >= 7.0.0."
      ) from e
    print(f"Found C++ compiler: {cxx}")

  def build_extension(self, ext):
    extension_dir = os.path.abspath(
        os.path.dirname(self.get_ext_fullpath(ext.name))
    )
    cxx = "clang++"
    if os.environ.get("CXX") is not None:
      cxx = os.environ.get("CXX")
    env = os.environ.copy()
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

    # Build only pyspiel (for pip package)
    detected_jobs = os.cpu_count()
    jobs = max(detected_jobs or 1, 1)
    print(f"Building pyspiel with {jobs} parallel job(s)")
    subprocess.check_call(["make", "pyspiel", f"-j{jobs}"],
                          cwd=self.build_temp,
                          env=env)

setuptools.setup(
    ext_modules=[CMakeExtension("pyspiel", sourcedir="open_spiel")],
    cmdclass={"build_ext": BuildExt},
)
