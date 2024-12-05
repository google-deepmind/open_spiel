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

# Only use for Github Actions CI!
OS=`uname -a | awk '{print $1}'`
if [[ "$OS" = "Darwin" ]]; then
  # This seems to be necessary to install python via brew in Github Actions
  rm -f /usr/local/bin/2to3-${OS_PYTHON_VERSION}
  rm -f /usr/local/bin/idle${OS_PYTHON_VERSION}
  rm -f /usr/local/bin/pydoc${OS_PYTHON_VERSION}
  rm -f /usr/local/bin/python${OS_PYTHON_VERSION}
  rm -f /usr/local/bin/python${OS_PYTHON_VERSION}*
fi

