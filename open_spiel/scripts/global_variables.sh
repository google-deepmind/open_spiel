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

# Building the Python API can be disabled by setting this to OFF.
export BUILD_WITH_PYTHON=${BUILD_WITH_PYTHON:-"ON"}

# Each optional dependency has their own flag, that defaults to the global
# "$DEFAULT_OPTIONAL_DEPENDENCY" if undefined. To enable an optional dependency,
# we recomment defining the associated environment variable in your bashrc or
# your virtualenv bashrc, e.g. export BUILD_WITH_HANABI="ON"
export BUILD_WITH_HANABI=${BUILD_WITH_HANABI:-$DEFAULT_OPTIONAL_DEPENDENCY}
export BUILD_WITH_ACPC=${BUILD_WITH_ACPC:-$DEFAULT_OPTIONAL_DEPENDENCY}
export BUILD_WITH_JULIA=${BUILD_WITH_JULIA:-$DEFAULT_OPTIONAL_DEPENDENCY}
export BUILD_WITH_EIGEN=${BUILD_WITH_EIGEN:-$DEFAULT_OPTIONAL_DEPENDENCY}
export BUILD_WITH_XINXIN=${BUILD_WITH_XINXIN:-$DEFAULT_OPTIONAL_DEPENDENCY}
export BUILD_WITH_ROSHAMBO=${BUILD_WITH_ROSHAMBO:-$DEFAULT_OPTIONAL_DEPENDENCY}

# Download precompiled binaries for libtorch (PyTorch C++ API).
# See https://pytorch.org/cppdocs/ for C++ documentation.
# This dependency is currently not supported by Travis CI test.
#
# From PyTorch documentation:
#
# > If you would prefer to write Python, and can afford to write Python, we
# > recommend using the Python interface to PyTorch. However, if you would
# > prefer to write C++, or need to write C++ (because of multithreading,
# > latency or deployment requirements), the C++ frontend to PyTorch provides
# > an API that is approximately as convenient, flexible, friendly and intuitive
# > as its Python counterpart.
#
# You can find an example usage in open_spiel/libtorch/torch_integration_test.cc
export BUILD_WITH_LIBTORCH="${BUILD_WITH_LIBTORCH:-"OFF"}"

# Turn off public states as it is Work-In-Progress.
export BUILD_WITH_PUBLIC_STATES="${BUILD_WITH_PUBLIC_STATES:-"OFF"}"

# Enable integration with GAMUT game generator (see games/gamut).
# Requires java and GAMUT, so disabled by default.
export BUILD_WITH_GAMUT="${BUILD_WITH_GAMUT:-"OFF"}"

# Flag to enable building with OR-Tools to get C++ optimization routines.
# Disabled by default as it requires installation of third party software.
# See algorithms/ortools/CMakeLists.txt for specific instructions.
export BUILD_WITH_ORTOOLS="${BUILD_WITH_ORTOOLS:-"OFF"}"
# You may want to replace this URL according to your system.
# Use version 8 at minimum, due to compatibility between absl library versions
# used in OpenSpiel and in OrTools.
export BUILD_WITH_ORTOOLS_DOWNLOAD_URL="${BUILD_WITH_ORTOOLS_DOWNLOAD_URL:-"https://github.com/google/or-tools/releases/download/v8.0/or-tools_ubuntu-18.04_v8.0.8283.tar.gz"}"
