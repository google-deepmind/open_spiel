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

read -r -d '' TESTSCRIPT << EOT
import tensorflow as tf
print(tf.__version__)
EOT

PY_EXEC=$(which $1)
if [[ ! -x $PY_EXEC ]]
then
  echo "Python executable: $PY_EXEC not found or not executable."
  exit -1
fi

echo "$TESTSCRIPT" | $PY_EXEC
exit $?
