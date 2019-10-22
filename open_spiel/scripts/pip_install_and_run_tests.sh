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

# The following builds open_spiel and executes the tests using the `python`
# command. The version under 'python' is automatically detected.
#
# It assumes:
#   - we are at the root of the project
#   - `install.sh` has been run
#
# Using a virtualenv is recommended but not mandatory.
#
set -e  # exit when any command fails
set -x

PYVERSION=$(python3 -c 'import sys; print(sys.version.split(" ")[0])')

echo "Building and testing in $PWD using 'python' (version $PYVERSION)."

python3 -m pip install .
if python3 setup.py test; then
  echo -e "\033[32mAll tests passed. Nicely done!\e[0m"
else
  echo -e "\033[31mAt least one test failed.\e[0m"
  exit 1
fi
