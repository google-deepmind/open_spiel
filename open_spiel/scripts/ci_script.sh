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

# Python 3.9 not default on Ubuntu yet.
OS=`uname -a | awk '{print $1}'`
if [[ "$OS" = "Linux" && "$OS_PYTHON_VERSION" = "3.9" ]]; then
  echo "Linux detected and Python 3.9 requested. Installing Python 3.9 and setting as default."
  sudo apt-get install python3.9 python3.9-dev
  sudo update-alternatives --install /usr/bin/python python /usr/bin/python3.9 1
  sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1
fi

PYBIN=${PYBIN:-"python3"}
PYBIN=`which $PYBIN`

source ./open_spiel/scripts/python_extra_deps.sh

${PYBIN} -m pip install --upgrade pip
${PYBIN} -m pip install --upgrade setuptools
${PYBIN} -m pip install --force-reinstall virtualenv==20.0.23

virtualenv -p ${PYBIN} ./venv
source ./venv/bin/activate

python3 --version
pip3 install --upgrade -r requirements.txt

[[ "$OPEN_SPIEL_ENABLE_JAX" = "ON" ]] && pip3 install --upgrade $OPEN_SPIEL_PYTHON_JAX_DEPS
[[ "$OPEN_SPIEL_ENABLE_PYTORCH" = "ON" ]] && pip3 install --upgrade $OPEN_SPIEL_PYTHON_PYTORCH_DEPS
[[ "$OPEN_SPIEL_ENABLE_TENSORFLOW" = "ON" ]] && pip3 install --upgrade $OPEN_SPIEL_PYTHON_TENSORFLOW_DEPS
[[ "$OPEN_SPIEL_ENABLE_PYTHON_MISC" = "ON" ]] && pip3 install --upgrade $OPEN_SPIEL_PYTHON_MISC_DEPS

./open_spiel/scripts/build_and_run_tests.sh

deactivate
