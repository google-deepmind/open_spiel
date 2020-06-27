#!/usr/bin/env bash

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


# The following should be easy to setup as a submodule:
# https://git-scm.com/docs/git-submodule

die() {
  echo "$*" 1>&2
  exit 1
}

set -e  # exit when any command fails
set -x  # show evaluation trace

MYDIR="$(dirname "$(realpath "$0")")"

# Calling this file from the project root is not allowed,
# as all the paths here are hard-coded to be relative to it.
#
# So this is not allowed:
# $ ./open_spiel/scripts/install.sh
#
# Instead, just call project-root install.sh file:
# $ ./install.sh
if [[ `basename $MYDIR` == "scripts" ]]; then
  die "Please run ./install.sh from the directory where you cloned the" \
      "project, do not run $0"
fi

# Load all the build settings.
source "${MYDIR}/open_spiel/scripts/global_variables.sh"

# Specify a download cache directory for external dependencies.
DEFAULT_DOWNLOAD_CACHE_DIR="$MYDIR/download_cache"

# Use the ENV variable if defined, or the default location otherwise.
DOWNLOAD_CACHE_DIR=${DOWNLOAD_CACHE_DIR:-$DEFAULT_DOWNLOAD_CACHE_DIR}

# Create the cache directory.
[[ -d "${DOWNLOAD_CACHE_DIR}" ]] || mkdir "${DOWNLOAD_CACHE_DIR}"

# 1. Clone the external dependencies before installing systen packages, to make
# sure they are present even if later commands fail.
#
# We do not use submodules because the CL versions are stored within Git
# metadata and we do not use Git within DeepMind, so it's hard to maintain.

# Note that this needs Git intalled, so we check for that.

git --version 2>&1 >/dev/null
GIT_IS_AVAILABLE=$?
if [ $GIT_IS_AVAILABLE -ne 0 ]; then #...
  if [[ "$OSTYPE" == "linux-gnu" ]]; then
    sudo apt-get install git
  elif [[ "$OSTYPE" == "darwin"* ]]; then  # Mac OSX
    brew install git
  else
    echo "The OS '$OSTYPE' is not supported (Only Linux and MacOS is). " \
         "Feel free to contribute the install for a new OS."
    exit 1
  fi
fi

# For the external dependencies, we use fixed releases for the repositories that
# the OpenSpiel team do not control.
# Feel free to upgrade the version after having checked it works.

[[ -d "./pybind11" ]] || git clone -b 'v2.2.4' --single-branch --depth 1 https://github.com/pybind/pybind11.git
# The official https://github.com/dds-bridge/dds.git seems to not accept PR,
# so we have forked it.
[[ -d open_spiel/games/bridge/double_dummy_solver ]] || \
  git clone -b 'develop' --single-branch --depth 1 https://github.com/jblespiau/dds.git  \
  open_spiel/games/bridge/double_dummy_solver

if [[ ! -d open_spiel/abseil-cpp ]]; then
  git clone -b '20200225.1' --single-branch --depth 1 https://github.com/abseil/abseil-cpp.git open_spiel/abseil-cpp
fi

# Optional dependencies.
DIR="open_spiel/games/hanabi/hanabi-learning-environment"
if [[ ${BUILD_WITH_HANABI:-"ON"} == "ON" ]] && [[ ! -d ${DIR} ]]; then
  git clone -b 'master' --single-branch --depth 15 https://github.com/deepmind/hanabi-learning-environment.git ${DIR}
  # We checkout a specific CL to prevent future breakage due to changes upstream
  # The repository is very infrequently updated, thus the last 15 commits should
  # be ok for a long time.
  pushd ${DIR}
  git checkout  'b31c973'
  popd
fi

# This Github repository contains the raw code from the ACPC server
# http://www.computerpokercompetition.org/downloads/code/competition_server/project_acpc_server_v1.0.42.tar.bz2
# with the code compiled as C++ within a namespace.
DIR="open_spiel/games/universal_poker/acpc"
if [[ ${BUILD_WITH_ACPC:-"ON"} == "ON" ]] && [[ ! -d ${DIR} ]]; then
  git clone -b 'master' --single-branch --depth 1  https://github.com/jblespiau/project_acpc_server.git ${DIR}
fi

# Add EIGEN template library for linear algebra.
# http://eigen.tuxfamily.org/index.php?title=Main_Page
DIR="open_spiel/eigen/libeigen"
if [[ ${BUILD_WITH_EIGEN:-"ON"} == "ON" ]] && [[ ! -d ${DIR} ]]; then
  git clone -b '3.3.7' --single-branch --depth 1  https://gitlab.com/libeigen/eigen.git ${DIR}
fi

# This GitHub repository contains Nathan Sturtevant's state of the art
# Hearts program xinxin.
DIR="open_spiel/games/hearts/hearts"
if [[ ${BUILD_WITH_XINXIN:-"ON"} == "ON" ]] && [[ ! -d ${DIR} ]]; then
  git clone -b 'master' --single-branch --depth 1  https://github.com/nathansttt/hearts.git ${DIR}
fi

# Add libtorch (PyTorch C++ API).
# This downloads the precompiled binaries available from the pytorch website.
DIR="open_spiel/libtorch/libtorch"
if [[ ${BUILD_WITH_LIBTORCH:-"ON"} == "ON" ]] && [[ ! -d ${DIR} ]]; then
  # CPU-only
  DOWNLOAD_URL="https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.5.1%2Bcpu.zip"

  # Uncomment one of the following if you want GPU support with CUDA:
  # # CUDA 9.2
  # DOWNLOAD_URL="https://download.pytorch.org/libtorch/cu92/libtorch-cxx11-abi-shared-with-deps-1.5.1%2Bcu92.zip"
  # # CUDA 10.1
  # DOWNLOAD_URL="https://download.pytorch.org/libtorch/cu101/libtorch-cxx11-abi-shared-with-deps-1.5.1%2Bcu101.zip"
  # # CUDA 10.2
  # DOWNLOAD_URL="https://download.pytorch.org/libtorch/cu102/libtorch-cxx11-abi-shared-with-deps-1.5.1.zip"

  DOWNLOAD_FILE="${DOWNLOAD_CACHE_DIR}/libtorch.zip"
  [[ -f "${DOWNLOAD_FILE}" ]] || wget --show-progress -O "${DOWNLOAD_FILE}" "$DOWNLOAD_URL"
  unzip "${DOWNLOAD_FILE}" -d "open_spiel/libtorch/"
fi

# 2. Install other required system-wide dependencies

# Install Julia if required and not present already.
if [[ ${BUILD_WITH_JULIA:-"OFF"} == "ON" ]]; then
  # Check that Julia is in the path.
  if [[ ! -x `which julia` ]]
  then
    echo -e "\e[33mWarning: julia not in your PATH. Trying \$HOME/.local/bin\e[0m"
    PATH=${PATH}:${HOME}/.local/bin
  fi

  if which julia >/dev/null; then
    JULIA_VERSION_INFO=`julia --version`
    echo -e "\e[33m$JULIA_VERSION_INFO is already installed.\e[0m"
  else
    # Julia installed needs wget, make sure it's accessible.
    if [[ "$OSTYPE" == "linux-gnu" ]]
    then
      [[ -x `which wget` ]] || sudo apt-get install wget
    elif [[ "$OSTYPE" == "darwin"* ]]
    then
      [[ -x `which wget` ]] || brew install wget
    fi
    # Now install Julia
    JULIA_INSTALLER="open_spiel/scripts/jill.sh"
    if [[ ! -f $JULIA_INSTALLER ]]; then
    curl https://raw.githubusercontent.com/abelsiqueira/jill/master/jill.sh -o jill.sh
    mv jill.sh $JULIA_INSTALLER
    fi
    JULIA_VERSION=1.3.1 bash $JULIA_INSTALLER -y
    # Should install in $HOME/.local/bin which was added to the path above
    [[ -x `which julia` ]] || die "julia not found PATH after install."
  fi

  # Install dependencies.
  julia --project="${MYDIR}/open_spiel/julia" -e 'using Pkg; Pkg.instantiate();'
fi

# Install other system-wide packages.
if [[ "$OSTYPE" == "linux-gnu" ]]; then
  EXT_DEPS="virtualenv clang cmake curl python3 python3-dev python3-pip python3-setuptools python3-wheel python3-tk"
  APT_GET=`which apt-get`
  if [ "$APT_GET" = "" ]
  then
     echo "This script assumes a Debian-based Linux distribution. Please install these packages manually or using your distribution's package manager:"
     echo "$EXT_DEPS"
     exit 1
  fi

  # We install the packages only if they are not present yet.
  # See https://stackoverflow.com/questions/18621990/bash-get-exit-status-of-command-when-set-e-is-active
  already_installed=0
  /usr/bin/dpkg-query --show --showformat='${db:Status-Status}\n' $EXT_DEPS || already_installed=$?
  if [ $already_installed -eq 0 ]
  then
    echo -e "\e[33mSystem wide packages already installed, skipping their installation.\e[0m"
  else
    echo "System wide packages missing. Installing them..."
    sudo apt-get update
    sudo apt-get install $EXT_DEPS
  fi

  if [[ "$TRAVIS" ]]; then
    sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${OS_PYTHON_VERSION} 10
  fi
elif [[ "$OSTYPE" == "darwin"* ]]; then  # Mac OSX
  [[ -x `which realpath` ]] || brew install coreutils || echo "** Warning: failed 'brew install coreutils' -- continuing"
  [[ -x `which cmake` ]] || brew install cmake || echo "** Warning: failed 'brew install cmake' -- continuing"
  [[ -x `which python3` ]] || brew install python3 || echo "** Warning: failed 'brew install python3' -- continuing"
  `python3 -c "import tkinter" > /dev/null 2>&1` || brew install tcl-tk || echo "** Warning: failed 'brew install tcl-tk' -- continuing"
  [[ -x `which clang++` ]] || die "Clang not found. Please install or upgrade XCode and run the command-line developer tools"
  [[ -x `which curl` ]] || brew install curl || echo "** Warning: failed 'brew install curl' -- continuing"
  curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
  python3 get-pip.py
  pip3 install virtualenv
else
  echo "The OS '$OSTYPE' is not supported (Only Linux and MacOS is). " \
       "Feel free to contribute the install for a new OS."
  exit 1
fi
