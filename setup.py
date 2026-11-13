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
import shlex
import subprocess
import sys
import sysconfig

import setuptools
from setuptools.command.build_ext import build_ext

# Maps distutils/setuptools Windows platform names to CMake -A generator
# platform names.
_PLAT_TO_CMAKE = {
    "win32": "Win32",
    "win-amd64": "x64",
    "win-arm32": "ARM",
    "win-arm64": "ARM64",
}


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
          "CMake must be installed to build" +
          f"the following extensions: {ext_names}") from e
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
    building_wheel = env.get("OPEN_SPIEL_BUILDING_WHEEL", "ON")
    is_win = sys.platform.startswith("win")
    host_plat = sysconfig.get_platform()
    plat_name = self.plat_name or host_plat
    is_cross_compile = is_win and plat_name != host_plat

    cmake_args = [
        f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extension_dir}",
        "-DCMAKE_BUILD_TYPE=Release",
        f"-DOPEN_SPIEL_BUILDING_WHEEL={building_wheel}",
    ]
    if is_cross_compile:
      py_version = f"{sys.version_info.major}.{sys.version_info.minor}"
      cmake_args += [
          "-DCMAKE_SYSTEM_NAME=Windows",
          f"-DOPEN_SPIEL_PYTHON_VERSION={py_version}",
      ]
    else:
      cmake_args.append(f"-DPython3_EXECUTABLE={sys.executable}")

    # Platform-specific configuration
    is_msvc_generator = False
    if is_win:
      cmake_generator = env.get("CMAKE_GENERATOR", "")
      is_msvc_generator = not cmake_generator or cmake_generator.startswith(
          "Visual Studio"
      )
      is_multi_config = is_msvc_generator or "Multi-Config" in cmake_generator
      contains_arch = any(arch in cmake_generator for arch in ("ARM", "Win64"))
      cmake_args.append(
          "-DCMAKE_CXX_FLAGS=/std:c++20 /utf-8 /bigobj /DWIN32 /D_WINDOWS"
          " /GR /EHsc"
      )
      if is_multi_config:
        cmake_args.append(
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE={extension_dir}"
        )
      if (
          is_msvc_generator
          and "CMAKE_GENERATOR_PLATFORM" not in env
          and not contains_arch
      ):
        if plat_name not in _PLAT_TO_CMAKE:
          raise ValueError(f"Unsupported Windows platform: {plat_name!r}")
        cmake_args += ["-A", _PLAT_TO_CMAKE[plat_name]]
    else:
      cxx = env.get("CXX", "clang++")
      cmake_args.append(f"-DCMAKE_CXX_COMPILER={cxx}")

    if is_msvc_generator:
      build_args = ["--", "/m"]
    else:
      detected_jobs = os.cpu_count()
      jobs = max(detected_jobs or 1, 1)
      build_args = ["--parallel", str(jobs)]

    if env.get("CMAKE_ARGS"):
      cmake_args += shlex.split(env["CMAKE_ARGS"])

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


# Get the requirements from file.
# When installing from pip it is in the parent directory
req_file = ""
if os.path.exists("requirements.txt"):
  req_file = "requirements.txt"
else:
  req_file = "../requirements.txt"

setuptools.setup(
    name="open_spiel",
    version="2.0.2",
    license="Apache 2.0",
    author="The OpenSpiel authors",
    author_email="open_spiel@google.com",
    description="A Framework for Reinforcement Learning in Games",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/deepmind/open_spiel",
    install_requires=_get_requirements(req_file),
    python_requires=">=3.11",
    ext_modules=[CMakeExtension("pyspiel", sourcedir="open_spiel")],
    cmdclass={"build_ext": BuildExt},
    zip_safe=False,
    packages=setuptools.find_packages(include=["open_spiel", "open_spiel.*"]),
)
