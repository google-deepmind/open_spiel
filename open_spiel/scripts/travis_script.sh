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

if [ ! $TRAVIS_USE_NOX -eq 0 ]; then
  # Build and run tests using nox
  sudo -H pip3 install nox
  PWD=`pwd`  # normally defined, but just in case!
  PYTHONPATH="$PYTHONPATH:$PWD:$PWD/build:$PWD/build/python" nox -s tests
  exit 0
fi

sudo -H pip3 install --upgrade pip
sudo -H pip3 install --upgrade setuptools
sudo -H pip3 install --force-reinstall virtualenv==20.0.23

virtualenv -p python3 ./venv
source ./venv/bin/activate

python3 --version

if [[ "$OPEN_SPIEL_ENABLE_JAX" = "ON" ]]
then
  echo "Jax enabled, installing Jax dependencies..."
  pip3 install --upgrade jax==0.2.7 jaxlib==0.1.57 dm-haiku==0.0.3 optax==0.0.2 chex==0.0.3 -q
fi

if [[ "$OPEN_SPIEL_ENABLE_PYTORCH" = "ON" ]]
then
  echo "PyTorch enabled, installing PyTorch dependencies..."
  pip3 install --upgrade torch==1.7.0 -q
fi

if [[ "$OPEN_SPIEL_ENABLE_TENSORFLOW" = "ON" ]]
then
  echo "Tensorflow enabled, installing Tensorflow dependencies..."
  pip3 install --upgrade tensorflow==2.4.1 tensorflow-probability\<0.8.0 tensorflow-probability\>=0.7.0 -q
fi

pip3 install --upgrade -r requirements.txt -q

./open_spiel/scripts/build_and_run_tests.sh

deactivate
