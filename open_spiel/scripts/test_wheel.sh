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
set -e
set -x

if [ "$1" = "" ];
then
  echo "Usage: test_wheel <project main dir>"
fi

uname -a

#OS=`uname -a | awk '{print $1}'`
#if [[ "$OS" = "Linux" && "$OS_PYTHON_VERSION" = "3.9" ]]; then
#  echo "Linux detected and Python 3.9 requested. Installing Python 3.9 and setting as default."
#  sudo apt-get install python3.9 python3.9-dev
#  sudo update-alternatives --install /usr/bin/python python /usr/bin/python3.9 1
#  sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1
#fi

PYBIN=${PYBIN:-"python"}
PYBIN=`which $PYBIN`
         
$PYBIN -m pip install --upgrade setuptools
$PYBIN -m pip install --upgrade -r requirements.txt -q
source $1/open_spiel/scripts/python_extra_deps.sh
$PYBIN -m pip install --upgrade $OPEN_SPIEL_PYTHON_JAX_DEPS $OPEN_SPIEL_PYTHON_PYTORCH_DEPS $OPEN_SPIEL_PYTHON_TENSORFLOW_DEPS $OPEN_SPIEL_PYTHON_MISC_DEPS

#if [[ "$OS" = "Linux" ]]; then
#  ${PYBIN} -m pip install wheelhouse/open_spiel-0.3.1-cp39-cp39-manylinux2014_x86_64.whl
#else
#  ${PYBIN} -m pip install wheelhouse/open_spiel-0.3.1-cp39-cp39-macosx_10_9_x86_64.whl
#fi

export OPEN_SPIEL_BUILDING_WHEEL="ON"
export OPEN_SPIEL_BUILD_WITH_HANABI="ON"
export OPEN_SPIEL_BUILD_WITH_ACPC="ON"

rm -rf build && mkdir build && cd build
cmake -DPython3_EXECUTABLE=${PYBIN} $1/open_spiel

NPROC="nproc"
if [[ "$OS" == "darwin"* ]]; then
  NPROC="sysctl -n hw.physicalcpu"
fi

MAKE_NUM_PROCS=$(${NPROC})
let TEST_NUM_PROCS=4*${MAKE_NUM_PROCS}

ctest -j$TEST_NUM_PROCS --output-on-failure -R "^python/*" ../open_spiel
