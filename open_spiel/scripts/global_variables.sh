#!/bin/sh

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

# Generates a playthrough for a new game with optional parameters.
# This script exists mainly as a reminder for the command to run.

# This file contains the global variables that control conditional dependencies.
# It is being used to know whether we should:
# (a) download a dependency (done in install.sh)
# (b) build it and link against it during the `cmake` build process
#
# Note that we do not change the value of the constants if they are already
# defined by an enclosing scope (useful for command line overrides).

# We add a single flag, to enable/disable all conditional dependencies, in
# particular to be able to use that in the Travis CI test.
export DEFAULT_OPTIONAL_DEPENDENCY=${DEFAULT_OPTIONAL_DEPENDENCY:-"OFF"}

export BUILD_WITH_HANABI=${BUILD_WITH_HANABI:-$DEFAULT_OPTIONAL_DEPENDENCY}
export BUILD_WITH_ACPC=${BUILD_WITH_ACPC:-$DEFAULT_OPTIONAL_DEPENDENCY}
