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


def clone_dependencies():
  """Clone required dependencies if they don't exist."""
  deps = [
      ("pybind11", "https://github.com/pybind/pybind11.git", ["-b", "master", "--single-branch", "--depth", "1"]),
      ("open_spiel/abseil-cpp", "https://github.com/abseil/abseil-cpp.git", ["-b", "20250127.1", "--single-branch", "--depth", "1"]),
      ("open_spiel/json", "https://github.com/nlohmann/json.git", ["-b", "master", "--single-branch", "--depth", "1"]),
      ("open_spiel/pybind11_json", "https://github.com/pybind/pybind11_json.git", ["-b", "master", "--single-branch", "--depth", "1"]),
      ("open_spiel/pybind11_abseil", "https://github.com/pybind/pybind11_abseil.git", ["-b", "master", "--single-branch", "--depth", "1"]),
      ("open_spiel/games/bridge/double_dummy_solver", "https://github.com/jblespiau/dds.git", ["-b", "develop", "--single-branch", "--depth", "1"])
  ]
  
  for name, url, args in deps:
    if not os.path.exists(name):
      print(f"Cloning {name}...")
      # Ensure parent directories exist
      parent_dir = os.path.dirname(name)
      if parent_dir:  # Only create if there's actually a parent directory
        os.makedirs(parent_dir, exist_ok=True)
      try:
        subprocess.check_call(["git", "clone"] + args + [url, name])
        print(f"Successfully cloned {name}")
      except subprocess.CalledProcessError as e:
        print(f"Failed to clone {name}: {e}")
        raise
  
  # Special checkouts for specific commits
  special_checkouts = [
      ("open_spiel/json", "9cca280a4d0ccf0c08f47a99aa71d1b0e52f8d03"),
      ("open_spiel/pybind11_json", "d0bf434be9d287d73a963ff28745542daf02c08f"),
      ("open_spiel/pybind11_abseil", "73992b5")
  ]
  
  for dir_name, commit_hash in special_checkouts:
    if os.path.exists(dir_name):
      print(f" Checking out specific commit for {dir_name}...")
      try:
        subprocess.check_call(["git", "checkout", commit_hash], cwd=dir_name)
        print(f" Successfully checked out {dir_name} commit")
      except subprocess.CalledProcessError as e:
        print(f" Failed to checkout {dir_name}: {e}")
        raise


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
      subprocess.check_call(["cmake", "--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except (OSError, subprocess.CalledProcessError):
      raise RuntimeError("CMake is required to build OpenSpiel extensions.")
    
    if not sys.platform.startswith("win"):
      cxx = os.environ.get("CXX", "clang++")
      try:
        subprocess.check_call([cxx, "--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
      except (OSError, subprocess.CalledProcessError):
        raise RuntimeError(f"C++ compiler {cxx} is required to build OpenSpiel extensions.")

  def build_extension(self, ext):
    extension_dir = os.path.abspath(
        os.path.dirname(self.get_ext_fullpath(ext.name)))
    
    env = os.environ.copy()
    cmake_args = [
        f"-DPython3_EXECUTABLE={sys.executable}",
        f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extension_dir}",
        "-DCMAKE_BUILD_TYPE=Release",
    ]
    
    # Platform-specific configuration
    if sys.platform.startswith("win"):
      # Windows-specific CMake arguments
      cmake_args += [
          "-DCMAKE_CXX_FLAGS=/std:c++17 /utf-8 /bigobj /DWIN32 /D_WINDOWS /GR /EHsc",
          f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE={extension_dir}",
          "-A", "x64"
      ]
      build_args = ["--config", "Release", "--", "/m"]
    else:
      # Unix-like systems
      cxx = "clang++"
      if os.environ.get("CXX") is not None:
        cxx = os.environ.get("CXX")
      cmake_args.append(f"-DCMAKE_CXX_COMPILER={cxx}")
      build_args = [f"-j{os.cpu_count()}"]
    
    if not os.path.exists(self.build_temp):
      os.makedirs(self.build_temp)
    
    # Configure
    subprocess.check_call(
        ["cmake", ext.sourcedir] + cmake_args, cwd=self.build_temp,
        env=env)

    # Build
    if sys.platform.startswith("win"):
      subprocess.check_call(["cmake", "--build", ".", "--target", "pyspiel"] + build_args,
                            cwd=self.build_temp, env=env)
    else:
      subprocess.check_call(["make", "pyspiel"] + build_args,
                            cwd=self.build_temp, env=env)
    



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


# Clone dependencies if building from source and they don't exist
if len(sys.argv) > 1 and sys.argv[1] in ['build', 'build_ext', 'bdist_wheel', 'install']:
  # Check if dependencies exist
  deps_exist = all(os.path.exists(dep[0]) for dep in [
    ("pybind11", "", []),
    ("open_spiel/abseil-cpp", "", []),
    ("open_spiel/pybind11_abseil", "", []),
    ("open_spiel/games/bridge/double_dummy_solver", "", [])
  ])
  
  if not deps_exist:
    print("Cloning required dependencies...")
    clone_dependencies()
  else:
    print("Dependencies already exist, skipping clone...")

# Get the requirements from file.
# When installing from pip it is in the parent directory
req_file = ""
if os.path.exists("requirements.txt"):
  req_file = "requirements.txt"
else:
  req_file = "../requirements.txt"

setuptools.setup(
    name="open_spiel",
    version="1.6.8",
    license="Apache 2.0",
    author="The OpenSpiel authors",
    author_email="open_spiel@google.com",
    description="A Framework for Reinforcement Learning in Games",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/deepmind/open_spiel",
    install_requires=_get_requirements(req_file),
    python_requires=">=3.9",
    ext_modules=[CMakeExtension("pyspiel", sourcedir="open_spiel")],
    cmdclass={"build_ext": BuildExt},
    zip_safe=False,
    packages=setuptools.find_packages(include=["open_spiel", "open_spiel.*"]),
)
