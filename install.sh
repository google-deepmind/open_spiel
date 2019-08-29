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

#!/usr/bin/env bash

# BEGIN GOOGLE-INTERNAL
# This comment aims at documenting the decision process.
#
# For OpenSourcing information in Google, see:
# go/big-opensource-project, go/rmi-opensource (for one shot release) and
# go/dm-opensource
# Given we aim for a long term project, our project is probably to be considered
# as a large one, not a one shot release.
#
#
# Possible options for building:
# - bazel
# - cmake
# bazel may be easier to maintain in the long run but it seems it may not
# support all features we need (e.g. cross langage dependencies). cmake is
# already setup and seems easy to update, so keeping it seems a good choice for
# now, but we may want to revisit it.
#
# Possible options for testing:
# - internally with Kokoro (e.g. done in Sonnet)
# - on Github with Travis
# The later is probably simpler at first, even though the second can detect
# breakage *before* releasing to Github. Given our manpower, let's go for easy.
# Note that Kokoro should be reasonable to setup (1/2 to 1 day).
#
# Dependencies: We want users to be able to have a one line install.
# Installing the dependencies (e.g. C++ Abseil) for the full system can be
# harder to setup and there may be version-conflicts. Thus we will have our
# specific depencencies only for OpenSpiel.
# The structure of the repo is currently:
#  ./: This is the root of the git repo.
#  open_spiel/: The code for OpenSpiel
#  pybind11/: The for for a specific dependency (here pybind11).
#
# TODO(jblespiau): For double_dummy_solver, we do not put it at the root, but
# within `open_spiel/games/bridge/double_dummy_solver`. Reassess that.
#
# To see the structure, run ./google_install_and_run_tests.sh nobuild
# END GOOGLE-INTERNAL

# The following should be easy to setup as a submodule:
# https://git-scm.com/docs/git-submodule

set -e  # exit when any command fails
set -x

if [[ "$OSTYPE" == "linux-gnu" ]]; then
  sudo apt-get update
  sudo apt-get install git virtualenv cmake python3 python3-dev python3-pip python3-setuptools python3-wheel
  if [[ "$TRAVIS" ]]; then
    sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${OS_PYTHON_VERSION} 10
  fi
elif [[ "$OSTYPE" == "darwin"* ]]; then  # Mac OSX
  brew install python3 gcc@7
  curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
  python3 get-pip.py
  pip3 install virtualenv
else
  echo "The OS '$OSTYPE' is not supported (Only Linux and MacOS is). " \
       "Feel free to contribute the install for a new OS."
  exit 1
fi

git clone -b 'v2.2.4' --single-branch --depth 1 https://github.com/pybind/pybind11.git
# TODO(jblespiau): Point to the official  https://github.com/dds-bridge/dds.git
# when pull requests are in
git clone -b 'develop' --single-branch --depth 1 https://github.com/jblespiau/dds.git  open_spiel/games/bridge/double_dummy_solver
git clone -b 'master' --single-branch --depth 1 https://github.com/abseil/abseil-cpp.git open_spiel/abseil-cpp
