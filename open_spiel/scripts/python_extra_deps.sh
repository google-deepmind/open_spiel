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

# These are extra packages that are not strictly required to run the OpenSpiel
# Python API, but are required by certain algorithms or tools. Packages here
# are for testing purposes: they are not installed by any of the install
# scripts, and are referred to only in the testing scripts run on GitHub, so
# they must be installed separately. The versions are pinned to ensure that
# tests are covering only those versions supported by the algorithms that use
# them, but could work for other versions too.
#
# To enable specific tests, please use the environment variables found in
# scripts/global_variables.sh

# This script depends on the Python version, which it gets from $PYBIN or
# $CI_PYBIN passed in as $1. If it's not defined, Python 3.9 is assumed.

PY_VER="3.9"
if [ "$1" != "" ]; then
  PY_VER=`$1 --version | awk '{print $2}'`
  if [ "$PY_VER" = "" ]; then
    PY_VER="3.9"
  fi
fi

verlte() {
  stuff=`echo -e "$1\n$2" | sort -V | head -n1`
  [ "$1" = "$stuff" ]
}

verlt() {
  [ "$1" = "$2" ] && return 1 || verlte $1 $2
}

#
# Python-version dependent versions
#

echo "Set Python version: $PY_VER"
if verlt $PY_VER 3.10; then
  echo "Detected Python version < 3.10"
  export OPEN_SPIEL_PYTHON_PYTORCH_DEPS="torch==1.13.1"
  export OPEN_SPIEL_PYTHON_JAX_DEPS="jax==0.4.6 jaxlib==0.4.6 dm-haiku==0.0.10 optax==0.1.7 chex==0.1.7 rlax==0.1.5 distrax==0.1.3"
  export OPEN_SPIEL_PYTHON_TENSORFLOW_DEPS="numpy==1.23.5 tensorflow==2.13.1 tensorflow-probability==0.19.0 tensorflow_datasets==4.9.2 keras==2.13.1"
  export OPEN_SPIEL_PYTHON_MISC_DEPS="IPython==5.8.0 networkx==2.4 matplotlib==3.5.2 mock==4.0.2 nashpy==0.0.19 scipy==1.10.1 testresources==2.0.1 cvxopt==1.3.1 cvxpy==1.2.0 ecos==2.0.10 osqp==0.6.2.post5 clu==0.0.6 flax==0.5.3"
elif verlt $PY_VER 3.12; then
  echo "Detected Python version in {3.10, 3.11}"
  export OPEN_SPIEL_PYTHON_PYTORCH_DEPS="torch==2.1.0"
  export OPEN_SPIEL_PYTHON_JAX_DEPS="jax==0.4.20 jaxlib==0.4.20 dm-haiku==0.0.10 optax==0.1.7 chex==0.1.84 rlax==0.1.6 distrax==0.1.4"
  export OPEN_SPIEL_PYTHON_TENSORFLOW_DEPS="numpy==1.26.1 tensorflow==2.14.0 tensorflow-probability==0.22.1 tensorflow_datasets==4.9.2 keras==2.14.0"
  export OPEN_SPIEL_PYTHON_MISC_DEPS="IPython==5.8.0 networkx==3.2 matplotlib==3.5.2 mock==4.0.2 nashpy==0.0.19 scipy==1.11.3 testresources==2.0.1 cvxopt==1.3.1 cvxpy==1.4.1 ecos==2.0.10 osqp==0.6.2.post5 clu==0.0.6 flax==0.5.3"
else
  echo "Detected Python version >= 3.12"
  export OPEN_SPIEL_PYTHON_PYTORCH_DEPS="torch==2.2.2"
  export OPEN_SPIEL_PYTHON_JAX_DEPS="jax==0.4.26 jaxlib==0.4.26 dm-haiku==0.0.12 optax==0.2.2 chex==0.1.86 rlax==0.1.6 distrax==0.1.5"
  export OPEN_SPIEL_PYTHON_TENSORFLOW_DEPS="numpy==1.26.4 tensorflow==2.16.1 tensorflow_datasets==4.9.4 keras==3.1.1"
  export OPEN_SPIEL_PYTHON_MISC_DEPS="IPython==8.23.0 networkx==3.3 matplotlib==3.8.4 mock==5.1.0 nashpy==0.0.41 scipy==1.11.4 testresources==2.0.1 cvxopt==1.3.2 cvxpy==1.4.2 ecos==2.0.13 osqp==0.6.5 clu==0.0.11 flax==0.8.2"
fi



