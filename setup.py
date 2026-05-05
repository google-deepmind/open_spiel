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

See https://setuptools.readthedocs.io/en/latest/setuptools.html
"""

import os
import subprocess
import sys

import setuptools
from setuptools.command.build_ext import build_ext


def _get_parallel_jobs() -> int:
  env_jobs = os.environ.get("CMAKE_BUILD_PARALLEL_LEVEL")
  if env_jobs:
    return max(int(env_jobs), 1)

  # Fallback: cpu_count, but cap to avoid thrashing
  return min(os.cpu_count() or 1, 16)


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
    """Check for required build tools: CMake and C++ compiler."""
    try:
      subprocess.check_call(["cmake", "--version"])
    except OSError as e:
      ext_names = ", ".join(e.name for e in self.extensions)
      raise RuntimeError(
          "CMake must be installed to build"
          + f"the following extensions: {ext_names}"
      ) from e
    print("Found CMake")

    if not sys.platform.startswith("win"):
      cxx = "clang++"
      if os.environ.get("CXX") is not None:
        cxx = os.environ.get("CXX")
      try:
        subprocess.check_call([cxx, "--version"])
      except OSError as e:
        ext_names = ", ".join(e.name for e in self.extensions)
        raise RuntimeError(
            "A C++ compiler that supports c++20 must be installed to build the "
            + "following extensions: {}".format(ext_names)
            + ". We recommend: Clang version >= 17.0.0."
        ) from e
      print("Found C++ compiler: {}".format(cxx))

  def build_extension(self, ext):
    extension_dir = os.path.abspath(
        os.path.dirname(self.get_ext_fullpath(ext.name))
    )

    env = os.environ.copy()
    cmake_args = [
        f"-DPython3_EXECUTABLE={sys.executable}",
        f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extension_dir}",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DOPEN_SPIEL_BUILDING_WHEEL=ON",
    ]

    jobs = _get_parallel_jobs()

    if sys.platform.startswith("win"):
      cmake_args += [
          "-DCMAKE_CXX_FLAGS=/std:c++20 /utf-8 /bigobj /DWIN32 /D_WINDOWS /GR /EHsc",
          f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE={extension_dir}",
          "-A",
          "x64",
      ]
      build_args = ["--", f"/m:{jobs}"]
    # elif sys.platform.startswith("linux"):
    #   cxx = os.environ.get("CXX", "clang++")
    #   cmake_args += [
    #       f"-DCMAKE_CXX_COMPILER={cxx}",
    #       "-G",
    #       "Ninja",
    #   ]
    #   build_args = ["--parallel", str(jobs)]
    else:
      cxx = os.environ.get("CXX", "clang++")
      cmake_args.append(f"-DCMAKE_CXX_COMPILER={cxx}")
      build_args = ["--parallel", str(jobs)]

    if not os.path.exists(self.build_temp):
      os.makedirs(self.build_temp)

    # Configure
    subprocess.check_call(
        ["cmake", ext.sourcedir] + cmake_args, cwd=self.build_temp, env=env
    )

    # Build using cmake --build for better portability
    subprocess.check_call(
        ["cmake", "--build", ".", "--target", "pyspiel", "--config", "Release"]
        + build_args,
        cwd=self.build_temp,
        env=env,
    )


setuptools.setup(
    ext_modules=[CMakeExtension("pyspiel", sourcedir="open_spiel")],
    cmdclass={"build_ext": BuildExt},
)
