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
export OPEN_SPIEL_PYTHON_JAX_DEPS="jax==0.3.24 jaxlib==0.3.24 dm-haiku==0.0.8 optax==0.1.3 chex==0.1.5 rlax==0.1.5 distrax==0.1.3"
export OPEN_SPIEL_PYTHON_PYTORCH_DEPS="torch==1.13.1"
export OPEN_SPIEL_PYTHON_TENSORFLOW_DEPS="numpy==1.23.5 tensorflow==2.12.0 tensorflow-probability==0.19.0 tensorflow_datasets==4.9.2 keras==2.12.0"
export OPEN_SPIEL_PYTHON_MISC_DEPS="IPython==5.8.0 networkx==2.4 matplotlib==3.5.2 mock==4.0.2 nashpy==0.0.19 scipy==1.10.1 testresources==2.0.1 cvxopt==1.3.1 cvxpy==1.2.0 ecos==2.0.10 osqp==0.6.2.post5 clu==0.0.6 flax==0.5.3"
