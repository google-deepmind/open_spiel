#!/bin/bash

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

# This file is called by the wheels workflow .github/workflows/wheels.yml.
set -e
set -x

if [ "$2" = "" ];
then
  echo "Usage: test_wheel <mode (full|basic)> <project main dir> [python binary]"
  echo ""
  echo "Basic mode tests only the python functionaly (no ML libraries)"
  echo "Full mode installs the extra ML libraries and the wheel. (requires Python >= 3.7 for JAX)."
  exit -1
fi

MODE=$1
PROJDIR=$2

uname -a
OS=`uname -a | awk '{print $1}'`

# If it's full mode on Linux, we have to install Python 3.9 and make it the default.
if [[ "$MODE" = "full" && "$OS" = "Linux" && "$OS_PYTHON_VERSION" = "3.9" ]]; then
  echo "Linux detected and Python 3.9 requested. Installing Python 3.9 and setting as default."
  sudo apt-get install python3.9 python3.9-dev
  sudo update-alternatives --install /usr/bin/python python /usr/bin/python3.9 1
  sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1
fi

# Setting of PYBIN is complicated because of all the different environments this is run from.
if [[ "$3" != "" ]]; then
  PYBIN=$3
else
  PYBIN=${PYBIN:-"python3"}
fi

PYBIN=`which $PYBIN`
$PYBIN -m pip install --upgrade setuptools
$PYBIN -m pip install --upgrade -r $PROJDIR/requirements.txt -q

if [[ "$MODE" = "full" ]]; then
  echo "Full mode. Installing Python extra deps libraries."
  source $PROJDIR/open_spiel/scripts/python_extra_deps.sh $PYBIN
  $PYBIN -m pip install --upgrade $OPEN_SPIEL_PYTHON_JAX_DEPS
  $PYBIN -m pip install --upgrade $OPEN_SPIEL_PYTHON_PYTORCH_DEPS
  $PYBIN -m pip install --upgrade $OPEN_SPIEL_PYTHON_MISC_DEPS
fi

if [[ "$MODE" = "full" ]]; then
  # Special case: Skip full tests on macOS Python 3.13 (only cp314 wheel built)
  if [[ "$OS" = "Darwin" && "$OS_PYTHON_VERSION" = "3.13" ]]; then
    echo "Skipping full tests on macOS Python 3.13 (only Python 3.14 wheel available)"
    exit 0
  fi
  
  # Dynamically detect Python version and install matching wheel
  PYTHON_VERSION=$(${PYBIN} --version 2>&1 | awk '{print $2}' | cut -d. -f1,2 | tr -d '.')
  echo "Detected Python version: ${PYTHON_VERSION}, installing matching wheel for ${OS}"
  
  if [[ "$OS" = "Linux" ]]; then
    # Match both x86_64 and aarch64 wheels
    WHEEL_FILE=$(ls wheelhouse/open_spiel-*-cp${PYTHON_VERSION}-cp${PYTHON_VERSION}-manylinux*.whl 2>/dev/null | head -1)
  elif [[ "$OS" = "Darwin" ]]; then
    WHEEL_FILE=$(ls wheelhouse/open_spiel-*-cp${PYTHON_VERSION}-cp${PYTHON_VERSION}-*.whl 2>/dev/null | head -1)
  else
    echo "ERROR: Unsupported OS: ${OS}"
    exit 1
  fi
  
  if [[ -z "$WHEEL_FILE" ]]; then
    echo "ERROR: No wheel found for Python ${PYTHON_VERSION} on ${OS}"
    echo "Available wheels:"
    ls wheelhouse/*.whl 2>/dev/null || echo "No wheels found in wheelhouse/"
    exit 1
  fi
  
  echo "Installing wheel: $WHEEL_FILE"
  ${PYBIN} -m pip install "$WHEEL_FILE"
fi

export OPEN_SPIEL_BUILDING_WHEEL="ON"
export OPEN_SPIEL_BUILD_WITH_HANABI="ON"
export OPEN_SPIEL_BUILD_WITH_ACPC="ON"

rm -rf build && mkdir build && cd build
cmake -DPython3_EXECUTABLE=${PYBIN} $PROJDIR/open_spiel

NPROC="nproc"
if [[ "$OS" == "darwin"* || "$OS" == "Darwin"* ]]; then
  NPROC="sysctl -n hw.physicalcpu"
fi

MAKE_NUM_PROCS=$(${NPROC})
let TEST_NUM_PROCS=4*${MAKE_NUM_PROCS}

ctest -j$TEST_NUM_PROCS --output-on-failure -R "^python/*" ../open_spiel
