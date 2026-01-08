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

PYBIN=${PYBIN:-"python${OS_PYTHON_VERSION}"}
PYBIN=${PYBIN:-"python"}
PYBIN=${PYBIN:-"python3"}
PYBIN=`which $PYBIN`

source ./open_spiel/scripts/python_extra_deps.sh $PYBIN

${PYBIN} -m venv ./venv
source ./venv/bin/activate

pip install --upgrade pip
pip install --upgrade setuptools

# Can use python and pip directly after here because we're in the virtual env

python --version
pip install --upgrade -r requirements.txt

[[ "$OPEN_SPIEL_ENABLE_JAX" = "ON" ]] && pip install --no-cache-dir --upgrade $OPEN_SPIEL_PYTHON_JAX_DEPS
[[ "$OPEN_SPIEL_ENABLE_PYTORCH" = "ON" ]] && pip install --no-cache-dir --upgrade $OPEN_SPIEL_PYTHON_PYTORCH_DEPS --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple
[[ "$OPEN_SPIEL_ENABLE_PYTHON_MISC" = "ON" ]] && pip install --no-cache-dir --upgrade $OPEN_SPIEL_PYTHON_MISC_DEPS

# We need PYBIN to be python on its own so that the build and run script
# finds the one from the virtual environment.
PYBIN="python" ./open_spiel/scripts/build_and_run_tests.sh --github_ci=true

deactivate
