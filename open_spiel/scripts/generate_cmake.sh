#!/bin/bash
# Copyright 2026 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Regenerates the CMake files generated from the Bazel BUILD files.
#
# Run this after editing any BUILD file. Needs only python3 (no Bazel).
#
#   open_spiel/scripts/generate_cmake.sh          # rewrite generated files
#   open_spiel/scripts/generate_cmake.sh --check  # fail if any is stale

set -euo pipefail

SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPEN_SPIEL_DIR="$(dirname "${SCRIPTS_DIR}")"
PYTHON="${PYTHON:-python3}"


PYTHONPATH="$(dirname "${OPEN_SPIEL_DIR}")${PYTHONPATH:+:${PYTHONPATH}}" exec \
  "${PYTHON}" -m open_spiel.scripts.bazel_to_cmake.bazel_to_cmake \
  --root "${OPEN_SPIEL_DIR}" "$@"
